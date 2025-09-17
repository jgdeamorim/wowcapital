from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
import json
import asyncio
import time
from backend.core.contracts import OrderRequest, OrderAck
from backend.execution.gateway import OrderRouter
from backend.orchestrator.plugin_manager import PluginManager
from backend.orchestrator.loader import load_manifests_from_dir
from backend.orchestrator.service import OrchestratorService
from backend.orchestrator.promotion import PromotionManager
from backend.common.config import load_yaml
from pathlib import Path
from backend.observability.metrics import metrics_app, HTTP_REQUESTS, HTTP_LATENCY
from backend.audit.worm import AsyncWormLog
from backend.storage.mongo import MongoStore
from backend.marketdata.app import md_app
from backend.marketdata.ws_public import PublicWS
from backend.observability.pubsub_redis import RedisPubSub
from backend.storage.redis_client import RedisClient
from backend.cofre.ledger import CofreLedger
from backend.cofre.executor_binance import BinanceCofreExecutor
from backend.cofre.executor_bybit import BybitCofreExecutor
from backend.cofre.executor_kraken import KrakenCofreExecutor
from backend.common.runtime_config import RuntimeConfig
from backend.observability.perf import PerfAggregator
from backend.orchestrator.routing import PluginRouting
from backend.shadow.runner import ShadowRunner
from backend.common.config import load_models
import httpx
import hashlib
from backend.common.auth import require_scopes
from backend.orchestrator.manifest_schema import StrategyManifest
from backend.simulation.backtester import backtest_strategy
from backend.orchestrator.allocator import BanditAllocator
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp
import os as _os

app = FastAPI(title="WOWCAPITAL API Gateway (MVP)")
pm = PluginManager()
router = OrderRouter(pm)
worm = AsyncWormLog("backend/var/audit/trading.ndjson")
mongo = MongoStore()
orch = OrchestratorService(pm)
pub = RedisPubSub(RedisClient())
ledger = CofreLedger()
try:
    _cofre_exec_binance = BinanceCofreExecutor()
except Exception:
    _cofre_exec_binance = None  # type: ignore
try:
    _cofre_exec_bybit = BybitCofreExecutor()
except Exception:
    _cofre_exec_bybit = None  # type: ignore
_cofre_exec_kraken = KrakenCofreExecutor()
import yaml as _yaml
try:
    _orch_cfg = load_yaml("backend/config/orchestrator.yaml")
    _gates = (_orch_cfg.get("gates_config") or {})
    _stage_mult = (_orch_cfg.get("stage_multipliers") or {})
except Exception:
    _gates = {}
    _stage_mult = {}
prom = PromotionManager(gates=_gates)
runtime = RuntimeConfig(RedisClient())
perf = PerfAggregator(RedisClient(), mongo)
_md_ws: PublicWS | None = None
route = PluginRouting(RedisClient())
_shadow: ShadowRunner | None = None


class _SecurityHeaders(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if _os.getenv("ENABLE_HSTS", "0") in ("1", "true", "TRUE") and request.url.scheme == "https":
            resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return resp

# CORS configurável
_origins = [o.strip() for o in (_os.getenv("CORS_ORIGINS", "").split(",")) if o.strip()]
if _os.getenv("CORS_ALLOW_ALL", "0") in ("1", "true", "TRUE"):
    _origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_origins or [], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(_SecurityHeaders)


class _HTTPMetrics(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:
            # record as 500 and re-raise
            route = getattr(request.scope.get("route"), "path", request.url.path)
            method = request.method
            HTTP_REQUESTS.labels(route, method, "500").inc()
            raise
        dur = max(0.0, time.time() - start)
        route = getattr(request.scope.get("route"), "path", request.url.path)
        method = request.method
        status = str(getattr(response, "status_code", 200))
        try:
            HTTP_REQUESTS.labels(route, method, status).inc()
            HTTP_LATENCY.labels(route, method).observe(dur)
        except Exception:
            pass
        return response

app.add_middleware(_HTTPMetrics)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


app.mount("/metrics", metrics_app)
app.include_router(md_app, prefix="/md")


class PlaceOrderIn(BaseModel):
    venue: str
    order: OrderRequest


@app.post("/exec/place", response_model=OrderAck)
async def exec_place(body: PlaceOrderIn, _: None = Depends(require_scopes(["trade"]))):
    try:
        # Audit decision/order
        await worm.append({
            "ts_ns": int(time.time() * 1e9),
            "event": "order_submit",
            "payload": body.order.model_dump(),
        })
        if mongo.enabled():
            await mongo.record_order(body.order.model_dump())
        # Register intent mapping for cofre policy resolution
        router.register_intent(body.order.idempotency_key, body.order.client_id, account=body.order.meta.get("account") if isinstance(body.order.meta, dict) else None)
        try:
            await pub.publish("orders.submit", body.order.model_dump())
        except Exception:
            pass
        ack = await router.place(body.venue, body.order)
        # Audit ack
        await worm.append({
            "ts_ns": int(time.time() * 1e9),
            "event": "order_ack",
            "payload": ack.model_dump(),
        })
        try:
            await pub.publish("orders.ack", ack.model_dump())
        except Exception:
            pass
        if mongo.enabled():
            await mongo.record_ack(ack.model_dump())
        return ack
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# --- Generic exchange test (health + balances quick) ---
class ExchangeTestIn(BaseModel):
    venue: str
    account: str | None = None


@app.post("/exchanges/test")
async def exchanges_test(body: ExchangeTestIn, _: None = Depends(require_scopes(["read"]))):
    try:
        health = await exec_health()
    except Exception:
        health = {}
    try:
        bals = await router.balances(body.venue, account=body.account)
    except Exception as e:
        bals = {"error": str(e)}
    return {"health": health, "balances": bals}


# --- Minimal OpenAI completion endpoint ---
class LLMIn(BaseModel):
    prompt: str
    model: str | None = None
    system: str | None = None
    temperature: float | None = None


@app.post("/llm/complete")
async def llm_complete(body: LLMIn):
    cfg = load_models()
    model = body.model or ((cfg.get("models") or {}).get("gpt41mini_primary") or {}).get("model", "gpt-4.1-mini")
    temperature = body.temperature
    if temperature is None:
        temperature = ((cfg.get("models") or {}).get("gpt41mini_primary") or {}).get("temperature", 0.2)
    rc = RedisClient()
    cache_key = None
    if rc.enabled():
        key_src = json.dumps({"m": model, "p": body.prompt, "s": body.system or "", "t": float(temperature)}, separators=(",", ":"))
        cache_key = f"llm:openai:{model}:{hashlib.sha256(key_src.encode()).hexdigest()}"
        cached = await rc.get_json(cache_key)
        if cached:
            return {"model": model, "cached": True, **cached}
    # Call OpenAI
    try:
        from backend.common.secure_env import get_secret
        api_key = get_secret("OPENAI_API_KEY")
    except Exception:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    messages.append({"role": "user", "content": body.prompt})
    payload = {"model": model, "messages": messages, "temperature": float(temperature)}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            data = r.json()
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=str(data))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            out = {"content": content, "usage": data.get("usage", {}), "id": data.get("id")}
            if rc.enabled() and cache_key:
                await rc.set_json(cache_key, out, ex=60)
            return {"model": model, **out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/exec/health")
async def exec_health() -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    tasks = []
    for name, adapter in router.adapters.items():
        tasks.append(adapter.healthcheck())
    oks = await asyncio.gather(*tasks, return_exceptions=True)
    for (name, _), ok in zip(router.adapters.items(), oks):
        results[name] = bool(ok) and not isinstance(ok, Exception)
    return results


class CancelIn(BaseModel):
    venue: str
    broker_order_id: str


@app.post("/exec/cancel")
async def exec_cancel(body: CancelIn, account: str | None = None, _: None = Depends(require_scopes(["trade"]))):
    ok = await router.cancel(body.venue, body.broker_order_id, account=account)
    return {"ok": ok}


class RuntimeSetIn(BaseModel):
    ns: str  # risk | cofre
    data: Dict
    plugin_id: str | None = None


@app.get("/runtime/overrides")
async def get_runtime_overrides(ns: str, plugin_id: str | None = None, _: None = Depends(require_scopes(["config"]))):
    return await runtime.get_overrides(ns, plugin_id)


@app.post("/runtime/overrides")
async def set_runtime_overrides(body: RuntimeSetIn, _: None = Depends(require_scopes(["config"]))):
    if body.ns not in ("risk", "cofre"):
        raise HTTPException(status_code=400, detail="invalid namespace")
    await runtime.set_overrides(body.ns, body.data, body.plugin_id)
    return {"ok": True}


# --- Accounts Admin (CRUD + Test) ---
import yaml
from backend.common.accounts import Accounts as AccountsFile
from backend.common.schemas import AccountsConfig, ModelsConfig


class AccountIn(BaseModel):
    venue: str
    account_id: str
    mode: str | None = None           # binance: spot|futures
    category: str | None = None       # bybit: spot|linear|inverse
    api_key_env: str | None = None
    api_secret_env: str | None = None
    testnet: bool | None = None       # bybit testnet hint for testing
    execution_mode: str | None = None # LIVE | SHADOW_REAL | SHADOW_DEMO (future use)


def _accounts_path() -> str:
    return "backend/config/accounts.yaml"


def _load_accounts_yaml() -> dict:
    p = Path(_accounts_path())
    if p.exists():
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:
            return {}
    return {}


def _save_accounts_yaml(data: dict) -> None:
    p = Path(_accounts_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))


@app.get("/accounts")
async def accounts_list(_: None = Depends(require_scopes(["config"]))):
    data = _load_accounts_yaml()
    return data.get("exchanges", {})


@app.post("/accounts")
async def accounts_upsert(body: AccountIn, _: None = Depends(require_scopes(["config"]))):
    data = _load_accounts_yaml()
    ex = data.setdefault("exchanges", {}).setdefault(body.venue, {}).setdefault("accounts", {})
    acc = ex.setdefault(body.account_id, {})
    if body.mode:
        acc["mode"] = body.mode
    if body.category:
        acc["category"] = body.category
    if body.api_key_env:
        acc["api_key"] = f"${{{body.api_key_env}:-}}"
    if body.api_secret_env:
        acc["api_secret"] = f"${{{body.api_secret_env}:-}}"
    # Validate entire structure before saving
    AccountsConfig(**{"exchanges": {k: {"accounts": v.get("accounts", {})} for k, v in data.get("exchanges", {}).items()}})
    _save_accounts_yaml(data)
    return {"ok": True, "account": {body.venue: {body.account_id: acc}}}


class AccountTestIn(BaseModel):
    venue: str
    account_id: str
    api_key: str | None = None     # optional direct test
    api_secret: str | None = None  # optional direct test
    mode: str | None = None
    category: str | None = None
    testnet: bool | None = None


@app.post("/accounts/test")
async def accounts_test(body: AccountTestIn, _: None = Depends(require_scopes(["config"]))):
    venue = body.venue.lower()
    tos = load_yaml("env/policies/exchange_tos.yaml") if Path("env/policies/exchange_tos.yaml").exists() else {"exchanges": {}}
    from backend.exchanges.binance import BinanceAdapter
    from backend.exchanges.bybit import BybitAdapter
    from backend.exchanges.kraken import KrakenAdapter
    api_key = body.api_key
    api_secret = body.api_secret
    mode = body.mode
    category = body.category
    if not api_key or not api_secret or not mode or (venue == "bybit" and not category):
        accs = AccountsFile(_accounts_path())
        keys = accs.get_keys(venue, body.account_id)
        api_key = api_key or keys.get("api_key")
        api_secret = api_secret or keys.get("api_secret")
        mode = mode or keys.get("mode")
        category = category or keys.get("category")
    if not api_key or not api_secret:
        raise HTTPException(status_code=400, detail="missing credentials for test")
    adapter = None
    if venue == "binance":
        import os as _os
        adapter = BinanceAdapter(tos, mode=(mode or _os.getenv("BINANCE_MODE", "spot")), api_key=api_key, api_secret=api_secret)
    elif venue == "bybit":
        import os as _os
        if body.testnet:
            _os.environ["BYBIT_TESTNET"] = "1"
        adapter = BybitAdapter(tos, category=(category or _os.getenv("BYBIT_CATEGORY", "spot")), api_key=api_key, api_secret=api_secret)
    elif venue == "kraken":
        import base64
        try:
            base64.b64decode(api_secret)
        except Exception:
            api_secret = base64.b64encode(api_secret.encode()).decode()
        adapter = KrakenAdapter(tos, api_key=api_key, api_secret_b64=api_secret)
    else:
        raise HTTPException(status_code=400, detail="unsupported venue")
    try:
        ok = await adapter.healthcheck()
        bals = await asyncio.wait_for(adapter.balances(), timeout=1.0)
        brief = {k: {"free": float(getattr(v, "free", 0.0)), "total": float(getattr(v, "total", 0.0))} for k, v in bals.items()}
        return {"ok": bool(ok), "balances": brief}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/perf/stats")
async def perf_stats(plugin_id: str, symbol: str | None = None, limit: int = 200, _: None = Depends(require_scopes(["read"]))):
    if not mongo.enabled():
        return {}
    q: Dict[str, Any] = {"plugin_id": plugin_id}
    if symbol:
        q["symbol"] = symbol
    cur = mongo.db.perf_samples.find(q).sort("_id", -1).limit(min(limit, 1000))
    n = 0
    slip_sum = 0.0
    score_sum = 0.0
    has_score = 0
    min_slip = None
    max_slip = None
    async for doc in cur:
        n += 1
        s = float(doc.get("abs_slippage_bps", 0.0))
        slip_sum += s
        if min_slip is None or s < min_slip:
            min_slip = s
        if max_slip is None or s > max_slip:
            max_slip = s
        if doc.get("score") is not None:
            score_sum += float(doc.get("score"))
            has_score += 1
    avg_slip = (slip_sum / n) if n else 0.0
    avg_score = (score_sum / has_score) if has_score else None
    return {"plugin_id": plugin_id, "symbol": symbol, "samples": n, "avg_abs_slippage_bps": avg_slip,
            "min_abs_slippage_bps": min_slip, "max_abs_slippage_bps": max_slip, "avg_score": avg_score}


@app.get("/perf/samples")
async def perf_samples(plugin_id: str, symbol: str | None = None, limit: int = 50):
    if not mongo.enabled():
        return []
    q: Dict[str, Any] = {"plugin_id": plugin_id}
    if symbol:
        q["symbol"] = symbol
    cur = mongo.db.perf_samples.find(q).sort("_id", -1).limit(min(limit, 1000))
    out = []
    async for doc in cur:
        doc["_id"] = str(doc.get("_id"))
        out.append(doc)
    return out


@app.get("/exec/balances")
async def exec_balances(venue: str, account: str | None = None):
    return await router.balances(venue, account=account)


@app.get("/exec/positions")
async def exec_positions(venue: str, account: str | None = None):
    return await router.positions(venue, account=account)


@app.get("/cofre")
async def get_cofre_state(_: None = Depends(require_scopes(["cofre"]))):
    return router.cofre_snapshot()


@app.get("/plugins")
async def list_plugins(_: None = Depends(require_scopes(["read"]))):
    return orch.list_plugins()


class DecideIn(BaseModel):
    plugin_id: str
    snapshot: Dict


@app.post("/orchestrator/decide")
async def orchestrator_decide(body: DecideIn, _: None = Depends(require_scopes(["trade"]))):
    return orch.decide(body.plugin_id, body.snapshot)


class DecidePlaceIn(BaseModel):
    plugin_id: str
    venue: str
    snapshot: Dict


@app.post("/orchestrator/place", response_model=OrderAck)
async def orchestrator_place(body: DecidePlaceIn, _: None = Depends(require_scopes(["trade"]))):
    # include venue in snapshot for indicator enrichment
    snap = dict(body.snapshot)
    snap.setdefault("venue", body.venue)
    # plugin auto-selection by routing weights
    plugin_id = body.plugin_id
    if plugin_id == "auto":
        sym = snap.get("symbol") or snap.get("symbol_id") or "BTCUSDT"
        candidates = list(pm.loaded.keys())
        pool = [p for p in candidates if p.startswith("jerico.")]
        default = "jerico.16ppr.v1" if "jerico.16ppr.v1" in pool else (pool[0] if pool else candidates[0])
        plugin_id = await route.pick_best(sym, pool or candidates, default)
    decision = orch.decide(plugin_id, snap)
    order = OrderRequest(**decision)
    await worm.append({"ts_ns": int(time.time()*1e9), "event": "strategy_decision", "payload": decision})
    try:
        await pub.publish("strategy.decisions", decision)
    except Exception:
        pass
    # record decision metadata for perf join (best-effort)
    try:
        score = decision.get("score") if isinstance(decision, dict) else None
        features = decision.get("features") if isinstance(decision, dict) else None
        await perf.record_decision(plugin_id, order.idempotency_key, symbol=order.symbol, venue=body.venue, score=score, features=features)
    except Exception:
        pass
    router.register_intent(order.idempotency_key, order.client_id, account=order.meta.get("account") if isinstance(order.meta, dict) else None)
    ack = await router.place(body.venue, order)
    await worm.append({"ts_ns": int(time.time()*1e9), "event": "order_ack", "payload": ack.model_dump()})
    try:
        await pub.publish("orders.ack", ack.model_dump())
    except Exception:
        pass
    return ack


class AssignIn(BaseModel):
    symbol: str
    plugin_id: str


@app.post("/orchestrator/assign")
async def orchestrator_assign(body: AssignIn, _: None = Depends(require_scopes(["config"]))):
    await route.assign(body.symbol, body.plugin_id)
    return {"ok": True}


@app.get("/orchestrator/assignments")
async def orchestrator_assignments(symbols: str | None = None, _: None = Depends(require_scopes(["config"]))):
    out = {}
    if not symbols:
        return out
    for s in [x.strip() for x in symbols.split(",") if x.strip()]:
        out[s] = await route.get_assignment(s)
    return out


class WeightIn(BaseModel):
    plugin_id: str
    symbol: str
    weight: float


@app.post("/orchestrator/weights")
async def orchestrator_set_weight(body: WeightIn, _: None = Depends(require_scopes(["config"]))):
    await route.set_weight(body.plugin_id, body.symbol, body.weight)
    return {"ok": True}


@app.get("/db/orders")
async def db_orders(limit: int = 100, _: None = Depends(require_scopes(["read"]))):
    if not mongo.enabled():
        return []
    cur = mongo.db.orders.find().sort("_id", -1).limit(min(limit, 1000))
    return [ {**doc, "_id": str(doc.get("_id"))} async for doc in cur ]


@app.get("/db/fills")
async def db_fills(limit: int = 100, _: None = Depends(require_scopes(["read"]))):
    if not mongo.enabled():
        return []
    cur = mongo.db.fills.find().sort("_id", -1).limit(min(limit, 1000))
    return [ {**doc, "_id": str(doc.get("_id"))} async for doc in cur ]


@app.get("/cofre/ledger")
async def cofre_ledger(limit: int = 100, _: None = Depends(require_scopes(["read"]))):
    if not mongo.enabled():
        return []
    cur = mongo.db.cofre_ledger.find().sort("_id", -1).limit(min(limit, 1000))
    return [ {**doc, "_id": str(doc.get("_id"))} async for doc in cur ]


class SweepIn(BaseModel):
    venue: str
    amount_usdt: float
    account: str | None = None
    reason: str | None = "manual"


@app.post("/cofre/sweep")
async def cofre_sweep(body: SweepIn, request: Request, _: None = Depends(require_scopes(["cofre"]))):
    venue = body.venue.lower()
    ok = False
    if venue == "binance" and _cofre_exec_binance:
        ok = await _cofre_exec_binance.sweep(body.amount_usdt, reason=body.reason or "manual", account=body.account)
    elif venue == "bybit" and _cofre_exec_bybit:
        ok = await _cofre_exec_bybit.sweep(body.amount_usdt, reason=body.reason or "manual", account=body.account)
    elif venue == "kraken":
        # Kraken: criar pendência (two-man rule) em Mongo se disponível
        if mongo.enabled():
            try:
                who = request.headers.get("X-API-Key") or "unknown"
                doc = {
                    "venue": "kraken",
                    "amount_usdt": float(body.amount_usdt),
                    "account": body.account or "acc#1",
                    "reason": body.reason or "manual",
                    "status": "pending",
                    "approvals": [who],
                    "requested_by": who,
                    "ts": int(time.time()*1e3),
                }
                ins = await mongo.db.cofre_pending.insert_one(doc)
                await ledger.record({"event": "cofre_sweep_request", **doc, "_id": str(ins.inserted_id)})
            except Exception:
                pass
        else:
            _add_pending_sweep("kraken", body.amount_usdt, body.account or "acc#1", body.reason or "manual")
        ok = False
    else:
        raise HTTPException(status_code=400, detail="unsupported venue for sweep or executor not configured")
    return {"ok": ok}

# --- Pending approvals (two-man) ---
import json as _json
from pathlib import Path as _Path

_PENDING_FILE = _Path("backend/var/cofre/pending.json")


def _load_pending():
    if _PENDING_FILE.exists():
        try:
            return _json.loads(_PENDING_FILE.read_text())
        except Exception:
            return []
    return []


def _save_pending(items):
    _PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    # atomic write
    tmp = _PENDING_FILE.with_suffix(".tmp")
    tmp.write_text(_json.dumps(items, separators=(",", ":")))
    tmp.replace(_PENDING_FILE)


def _add_pending_sweep(venue: str, amount_usdt: float, account: str, reason: str):
    items = _load_pending()
    entry = {"venue": venue, "amount_usdt": amount_usdt, "account": account, "reason": reason}
    items.append(entry)
    _save_pending(items)
    return entry


@app.get("/cofre/pending")
async def cofre_pending(_: None = Depends(require_scopes(["cofre"]))):
    if mongo.enabled():
        cur = mongo.db.cofre_pending.find({"status": "pending"}).sort("ts", -1).limit(1000)
        return [{**doc, "_id": str(doc.get("_id"))} async for doc in cur]
    return _load_pending()


class ApproveIn(BaseModel):
    index: int | None = None
    id: str | None = None


@app.post("/cofre/approve")
async def cofre_approve(body: ApproveIn, request: Request, _: None = Depends(require_scopes(["cofre"]))):
    # Mongo path com aprovação dupla
    if mongo.enabled() and body.id:
        try:
            from bson import ObjectId  # type: ignore
        except Exception:
            ObjectId = None  # type: ignore
        if not ObjectId:
            raise HTTPException(status_code=500, detail="bson not available")
        who = request.headers.get("X-API-Key") or "unknown"
        doc = await mongo.db.cofre_pending.find_one({"_id": ObjectId(body.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="not found")
        approvals = set([str(x) for x in (doc.get("approvals") or [])])
        if who in approvals:
            raise HTTPException(status_code=400, detail="already approved by this user")
        approvals.add(who)
        status = "pending"
        if len(approvals) >= 2:
            status = "approved"
        await mongo.db.cofre_pending.update_one({"_id": doc["_id"]}, {"$set": {"approvals": list(approvals), "status": status}})
        try:
            await ledger.record({"event": "cofre_sweep_approved", "_id": str(doc.get("_id")), "approver": who, "status": status})
        except Exception:
            pass
        return {"ok": True, "id": body.id, "status": status, "approvals": list(approvals)}
    # Fallback arquivo
    items = _load_pending()
    if body.index is None or body.index < 0 or body.index >= len(items):
        raise HTTPException(status_code=400, detail="invalid index")
    entry = items.pop(body.index)
    _save_pending(items)
    try:
        await ledger.record({"event": "cofre_sweep_approved", **entry})
    except Exception:
        pass
    return {"ok": True, "approved": entry}


@app.get("/cofre/safety")
async def cofre_safety(_: None = Depends(require_scopes(["cofre"]))):
    return router._cofre.safety_snapshot()  # type: ignore


class ReplenishIn(BaseModel):
    account: str
    amount_usdt: float
    reason: str | None = "replenish"


@app.post("/cofre/replenish")
async def cofre_replenish(body: ReplenishIn, _: None = Depends(require_scopes(["cofre"]))):
    ok = await router._cofre.replenish_from_safety(body.account, body.amount_usdt)  # type: ignore
    if not ok:
        raise HTTPException(status_code=400, detail="insufficient safety pool")
    # Ledger best-effort
    try:
        await ledger.record({
            "venue": "internal",
            "account": body.account,
            "amount_usdt": body.amount_usdt,
            "event": "cofre_replenish",
            "reason": body.reason or "replenish",
        })
    except Exception:
        pass
    return {"ok": True}


class ShadowStartIn(BaseModel):
    accounts: list[Dict]


@app.post("/shadow/start")
async def shadow_start(body: ShadowStartIn):
    rc = RedisClient()
    if rc.enabled():
        await rc.client.set("shadow:accounts", json.dumps(body.accounts))  # type: ignore
    global _shadow
    if _shadow is None:
        from backend.shadow.runner import ShadowRunner
        _shadow = ShadowRunner(rc, worm, perf)
        await _shadow.start()
    return {"ok": True}


@app.post("/shadow/stop")
async def shadow_stop():
    rc = RedisClient()
    if rc.enabled():
        await rc.client.delete("shadow:accounts")  # type: ignore
    global _shadow
    if _shadow:
        await _shadow.stop()
        _shadow = None
    return {"ok": True}


@app.on_event("startup")
async def on_startup():
    # Load manifests from dir (plugins/examples and plugins/manifests if exist)
    import os
    cfg_path = Path("backend/config/orchestrator.yaml")
    if cfg_path.exists():
        try:
            cfg = load_yaml(str(cfg_path))
            manifests_dir = os.getenv("PLUGIN_MANIFESTS_DIR", cfg.get("examples_dir", "backend/plugins/examples"))
        except Exception:
            manifests_dir = os.getenv("PLUGIN_MANIFESTS_DIR", "backend/plugins/examples")
    else:
        manifests_dir = os.getenv("PLUGIN_MANIFESTS_DIR", "backend/plugins/examples")
    loaded = load_manifests_from_dir(pm, manifests_dir)
    # rebuild registry after manifests load
    global orch
    orch = OrchestratorService(pm)
    # Ensure Mongo indexes
    if mongo.enabled():
        try:
            await mongo.db.orders.create_index("idempotency_key")
            await mongo.db.acks.create_index("broker_order_id")
            await mongo.db.fills.create_index([("symbol", 1), ("ts", -1)])
            # TTL for microstructure snapshots (30 minutes)
            try:
                await mongo.db.micro.create_index("ts_dt", expireAfterSeconds=1800)
                await mongo.db.micro.create_index([("venue", 1), ("symbol", 1), ("ts_dt", -1)])
            except Exception:
                pass
            await mongo.db.cofre_pending.create_index([("status", 1), ("ts", -1)])
        except Exception:
            pass
    await router.start()
    # Start public MD WS if enabled
    try:
        import os
        if os.getenv("MD_WS_ENABLED", "1") in ("1", "true", "TRUE"):
            global _md_ws
            _md_ws = PublicWS(RedisClient())
            await _md_ws.start()
    except Exception:
        pass


@app.on_event("shutdown")
async def on_shutdown():
    await router.stop()
    try:
        if _md_ws:
            await _md_ws.stop()
    except Exception:
        pass
    try:
        if _shadow:
            await _shadow.stop()
    except Exception:
        pass


@app.get("/orchestrator/status")
async def orchestrator_status():
    return prom.all()


class PromoteIn(BaseModel):
    plugin_id: str
    stage: str


@app.post("/orchestrator/promote")
async def orchestrator_promote(body: PromoteIn):
    prom.set_stage(body.plugin_id, body.stage)
    await worm.append({"ts_ns": int(time.time()*1e9), "event": "plugin_promotion", "payload": body.__dict__})
    # Auto size multiplier by stage
    def _mult_for_stage(stage: str) -> float:
        default_map = {"pending": 0.0, "sandbox": 0.0, "shadow": 0.1, "active": 1.0}
        if stage in _stage_mult:
            try:
                return float(_stage_mult.get(stage))
            except Exception:
                pass
        return float(default_map.get(stage, 1.0))
    try:
        mult = _mult_for_stage(body.stage)
        await runtime.set_overrides("risk", {"size_multiplier": mult}, body.plugin_id)
        await worm.append({"ts_ns": int(time.time()*1e9), "event": "runtime_override", "payload": {"plugin_id": body.plugin_id, "ns": "risk", "size_multiplier": mult}})
        try:
            from backend.observability.metrics import SIZE_MULTIPLIER
            SIZE_MULTIPLIER.labels(body.plugin_id).set(mult)
        except Exception:
            pass
    except Exception:
        pass
    return {"ok": True, "status": prom.get_stage(body.plugin_id)}


# --- Internal: Strategy validate/backtest and allocator suggest (secured) ---
class StrategyValidateIn(BaseModel):
    manifest: Dict


@app.post("/internal/strategy/validate")
async def internal_strategy_validate(body: StrategyValidateIn, _: None = Depends(require_scopes(["config"]))):
    m = StrategyManifest.parse_obj(body.manifest)
    return {"ok": True, "entrypoint": m.entrypoint, "name": m.name, "version": m.version}


class StrategyBacktestIn(BaseModel):
    entrypoint: str
    symbol: str
    venue: str = "binance"
    candles: list[Dict]
    params: Dict | None = None
    spreads_bps: list[float] | None = None
    micro_timeline: list[Dict] | None = None


@app.post("/internal/strategy/backtest")
async def internal_strategy_backtest(body: StrategyBacktestIn, _: None = Depends(require_scopes(["config"]))):
    # Guard inputs
    if not body.candles or len(body.candles) < 10:
        raise HTTPException(status_code=400, detail="not enough candles")
    metrics = backtest_strategy(body.entrypoint, symbol=body.symbol, candles=body.candles, params=body.params,
                                venue=body.venue, spreads_bps=body.spreads_bps, micro_timeline=body.micro_timeline)
    return {"ok": True, "metrics": metrics}


class AllocatorSuggestIn(BaseModel):
    arms: list[str]
    budget: float = 1.0
    min_alloc: float | None = 0.0


@app.post("/internal/allocator/suggest")
async def internal_allocator_suggest(body: AllocatorSuggestIn, _: None = Depends(require_scopes(["config"]))):
    alloc = BanditAllocator(RedisClient())
    out = await alloc.suggest(body.arms, budget=float(body.budget), min_alloc=float(body.min_alloc or 0.0))
    return {"ok": True, "allocations": out}


# --- Internal: Global/Venue/Account pause controls (kill-switches) ---
class PauseIn(BaseModel):
    scope: str  # all | venue | account
    venue: str | None = None
    account: str | None = None
    on: bool = True


@app.post("/internal/exec/pause")
async def internal_exec_pause(body: PauseIn, _: None = Depends(require_scopes(["config"]))):
    rc = RedisClient()
    if not rc.enabled():
        raise HTTPException(status_code=500, detail="redis not enabled")
    if body.scope == "all":
        if body.on:
            await rc.client.set("exec:pause:all", "1")  # type: ignore
        else:
            await rc.client.delete("exec:pause:all")  # type: ignore
    elif body.scope == "venue":
        if not body.venue:
            raise HTTPException(status_code=400, detail="venue required")
        key = f"exec:pause:venue:{body.venue}"
        if body.on:
            await rc.client.set(key, "1")  # type: ignore
        else:
            await rc.client.delete(key)  # type: ignore
    elif body.scope == "account":
        if not body.venue or not body.account:
            raise HTTPException(status_code=400, detail="venue and account required")
        key = f"exec:pause:account:{body.venue}:{body.account}"
        if body.on:
            await rc.client.set(key, "1")  # type: ignore
        else:
            await rc.client.delete(key)  # type: ignore
    else:
        raise HTTPException(status_code=400, detail="invalid scope")
    return {"ok": True}
class PromotionProposeIn(BaseModel):
    plugin_id: str
    metrics: Dict


@app.post("/internal/promotion/propose")
async def internal_promotion_propose(body: PromotionProposeIn, _: None = Depends(require_scopes(["config"]))):
    res = prom.propose(body.plugin_id, body.metrics)
    await worm.append({"ts_ns": int(time.time()*1e9), "event": "promotion_propose", "payload": {"plugin_id": body.plugin_id, **res}})
    return res


class PromotionScheduleIn(BaseModel):
    plugin_id: str
    steps: list[Dict]


@app.post("/internal/promotion/schedule")
async def internal_promotion_schedule(body: PromotionScheduleIn, _: None = Depends(require_scopes(["config"]))):
    prom.schedule_ramp(body.plugin_id, body.steps)
    await worm.append({"ts_ns": int(time.time()*1e9), "event": "promotion_schedule", "payload": {"plugin_id": body.plugin_id, "steps": body.steps}})
    return {"ok": True}


class PromotionAdvanceIn(BaseModel):
    plugin_id: str
    metrics: Dict


@app.post("/internal/promotion/advance")
async def internal_promotion_advance(body: PromotionAdvanceIn, _: None = Depends(require_scopes(["config"]))):
    res = prom.advance(body.plugin_id, body.metrics)
    await worm.append({"ts_ns": int(time.time()*1e9), "event": "promotion_advance", "payload": {"plugin_id": body.plugin_id, **res}})
    # If advanced, set size_multiplier according to new stage
    try:
        if res.get("ok"):
            stage = str(res.get("stage"))
            def _mult_for_stage(stage: str) -> float:
                m = {"pending": 0.0, "sandbox": 0.0, "shadow": 0.1, "active": 1.0}
                return float(m.get(stage, 1.0))
            mult = _mult_for_stage(stage)
            await runtime.set_overrides("risk", {"size_multiplier": mult}, body.plugin_id)
            await worm.append({"ts_ns": int(time.time()*1e9), "event": "runtime_override", "payload": {"plugin_id": body.plugin_id, "ns": "risk", "size_multiplier": mult, "reason": "stage_advance"}})
            try:
                from backend.observability.metrics import SIZE_MULTIPLIER
                SIZE_MULTIPLIER.labels(body.plugin_id).set(mult)
            except Exception:
                pass
    except Exception:
        pass
    return res


@app.get("/internal/promotion/status")
async def internal_promotion_status(plugin_id: str | None = None, _: None = Depends(require_scopes(["config"]))):
    if not plugin_id:
        return {"status": prom.all()}
    return {"plugin_id": plugin_id, "stage": prom.get_stage(plugin_id)}
