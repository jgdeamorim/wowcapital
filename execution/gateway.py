from __future__ import annotations
import asyncio
import time
from typing import Dict
from backend.common.config import load_policies
from backend.common.accounts import Accounts
from backend.core.contracts import OrderRequest, OrderAck
from backend.exchanges.base import ExchangeAdapter
from backend.exchanges.binance import BinanceAdapter
from backend.exchanges.bybit import BybitAdapter
from backend.exchanges.kraken import KrakenAdapter
from backend.audit.worm import AsyncWormLog
from backend.tca.calculator import estimate_costs_bps
from backend.common.config import load_yaml
from backend.observability.metrics import AGGR_COFRE_BAL, AGGR_EXPLOSIONS, AGGR_RESETS, FEE_TO_EDGE
from backend.cofre.manager import CofreManager
from backend.cofre.executor_binance import BinanceCofreExecutor
from backend.cofre.executor_bybit import BybitCofreExecutor
from backend.orchestrator.plugin_manager import PluginManager
from backend.storage.mongo import MongoStore
from backend.risk.engine import RiskEngine
from backend.common.pairs import PairsConfig
from backend.storage.redis_client import RedisClient
from backend.observability.pubsub_redis import RedisPubSub
from backend.common.config import load_yaml as _load_yaml
import os
import aiohttp
from backend.common.runtime_config import RuntimeConfig
from backend.observability.perf import PerfAggregator
from backend.indicators.greeks_synth import SyntheticGreeks


class OrderRouter:
    def __init__(self, pm: PluginManager | None = None):
        import os
        tos = load_policies().get("exchange_tos", {})
        self._accounts = Accounts()
        binance_mode = os.getenv("BINANCE_MODE", "spot").lower()
        enabled = os.getenv("EXCHANGES_ENABLED", "binance,bybit,kraken")
        enabled_set = {e.strip().lower() for e in enabled.split(",") if e.strip()}
        adapters: Dict[str, ExchangeAdapter] = {}
        if "binance" in enabled_set:
            adapters["binance"] = BinanceAdapter(tos, mode=binance_mode)
        if "bybit" in enabled_set:
            adapters["bybit"] = BybitAdapter(tos)
        if "kraken" in enabled_set:
            adapters["kraken"] = KrakenAdapter(tos)
        self.adapters = adapters
        self._account_adapters: Dict[str, Dict[str, ExchangeAdapter]] = {"binance": {}, "bybit": {}, "kraken": {}}
        self._worm = AsyncWormLog("backend/var/audit/trading.ndjson")
        self._tos_cfg = tos
        self._binance_mode = binance_mode
        self._cofre = CofreManager()
        # Cofre executors (optional; only if secrets available)
        try:
            self._cofre_exec_binance = BinanceCofreExecutor()
        except Exception:
            self._cofre_exec_binance = None  # type: ignore
        try:
            self._cofre_exec_bybit = BybitCofreExecutor()
        except Exception:
            self._cofre_exec_bybit = None  # type: ignore
        self._pm = pm or PluginManager()  # manifests loaded externally; used for cofre policy lookup
        self._intent_map: Dict[str, str] = {}  # idempotency_key -> plugin_id
        self._mongo = MongoStore()
        self._ack_ts: Dict[str, float] = {}  # clientOrderId -> ack timestamp (seconds)
        self._risk = RiskEngine()
        self._pairs = PairsConfig()
        self._redis = RedisClient()
        self._pub = RedisPubSub(self._redis)
        self._runtime = RuntimeConfig(self._redis)
        self._perf = PerfAggregator(self._redis, self._mongo)
        self._greeks = SyntheticGreeks(self._redis)
        # circuit breaker memory fallback
        self._cb_block_until: Dict[str, float] = {}

    async def _cb_is_blocked(self, venue: str) -> bool:
        now = time.time()
        # memory check
        t = self._cb_block_until.get(venue)
        if t and now < t:
            return True
        if self._redis.enabled():
            try:
                v = await self._redis.client.get(f"cb:block:{venue}")  # type: ignore
                if v:
                    return True
            except Exception:
                pass
        return False

    async def _cb_trip(self, venue: str, block_s: float = 20.0) -> None:
        until = time.time() + block_s
        self._cb_block_until[venue] = until
        if self._redis.enabled():
            try:
                await self._redis.client.setex(f"cb:block:{venue}", int(block_s), "1")  # type: ignore
            except Exception:
                pass
        await self._worm.append({
            "ts_ns": int(time.time() * 1e9),
            "event": "circuit_breaker_trip",
            "payload": {"venue": venue, "block_s": block_s},
        })

    async def _cb_record_fail(self, venue: str, window_s: int = 30, threshold: int = 5) -> None:
        if not self._redis.enabled():
            # memory fallback: trip directly after threshold single-process
            key = f"{venue}:_memfail"
            cnt = int(getattr(self, key, 0)) + 1
            setattr(self, key, cnt)
            if cnt >= threshold:
                await self._cb_trip(venue)
            return
        try:
            key = f"cb:fail:{venue}"
            pipe = self._redis.client.pipeline()  # type: ignore
            pipe.incr(key, 1)
            pipe.expire(key, window_s)
            res = await pipe.execute()
            cnt = int(res[0] or 0)
            if cnt >= threshold:
                await self._cb_trip(venue)
        except Exception:
            pass

    async def place(self, venue: str, req: OrderRequest, timeout_ms: int = 600) -> OrderAck:
        # Global pause (Redis key exec:pause:all)
        try:
            if self._redis.enabled():
                v = await self._redis.client.get("exec:pause:all")  # type: ignore
                if v:
                    return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="PAUSED_ALL", ts_ns=int(time.time()*1e9))
        except Exception:
            pass
        # Circuit breaker check
        if await self._cb_is_blocked(venue):
            from backend.observability.metrics import CB_BLOCKS
            try:
                CB_BLOCKS.labels(venue).inc()
            except Exception:
                pass
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="CB_BLOCKED", ts_ns=int(time.time()*1e9))
        account = None
        try:
            if isinstance(req.meta, dict):
                account = req.meta.get("account")
        except Exception:
            account = None
        if venue == "auto":
            venue = self._select_venue(req.symbol, account)
        # Venue/account pause flags
        try:
            if self._redis.enabled():
                if await self._redis.client.get(f"exec:pause:venue:{venue}"):  # type: ignore
                    return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="PAUSED_VENUE", ts_ns=int(time.time()*1e9))
                if account and await self._redis.client.get(f"exec:pause:account:{venue}:{account}"):  # type: ignore
                    return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="PAUSED_ACCOUNT", ts_ns=int(time.time()*1e9))
        except Exception:
            pass
        adapter = self._get_adapter(venue, account)
        # Risk pre-trade inputs
        ref_mid, spread_bps = await self._fetch_mid_and_spread_bps(venue, req.symbol)
        # (defer adjust until after potential size multiplier)
        try:
            balances = await asyncio.wait_for(adapter.balances(), timeout=0.3)
        except Exception:
            balances = {}
        # detect perps and funding
        is_perp = False
        try:
            if venue == "binance" and getattr(adapter, "_mode", "") == "futures":
                is_perp = True
            if venue == "bybit" and getattr(adapter, "_category", "") in ("linear", "inverse"):
                is_perp = True
        except Exception:
            is_perp = False
        funding_bps = await self._fetch_funding_bps(venue, req.symbol, adapter) if is_perp else 0.0
        # Resolve plugin id for runtime overrides (via intent map)
        plugin_id = None
        try:
            if req.idempotency_key and req.idempotency_key in self._intent_map:
                plugin_id = self._intent_map.get(req.idempotency_key)
        except Exception:
            plugin_id = None
        risk_over = await self._runtime.get_overrides("risk", plugin_id)
        # Optional size multiplier (ramp/partial_live sizing)
        try:
            mult = float(risk_over.get("size_multiplier", 1.0))
        except Exception:
            mult = 1.0
        if mult != 1.0:
            try:
                d = req.model_dump()
                d["qty"] = float(d.get("qty", 0.0)) * mult
                req = OrderRequest(**d)
            except Exception:
                pass
        # Now adjust to venue precision/steps/min rules
        req = await self._adjust_order(venue, req, ref_mid)
        # Dynamic overrides from synthetic greeks proxies (best-effort)
        dyn_over: Dict[str, float] = {}
        try:
            proxies = await self._greeks.compute(venue, req.symbol, funding_rate_bps=funding_bps if is_perp else 0.0)
            # heuristics: tighten guards on adverse convexity/vol-of-vol; relax on favorable trend and tight spread
            if proxies.get("vanna_hat", 0.0) > 0.6:
                # lower leverage cap under vol-of-vol
                ml = float(risk_over.get("max_leverage", 10.0))
                dyn_over["max_leverage"] = max(2.0, min(ml, 6.0))
            if proxies.get("gamma_hat", 0.0) < -0.4:
                dyn_over["price_guard_bps"] = float(risk_over.get("price_guard_bps", 80.0)) * 0.5
                dyn_over["slippage_cap_bps"] = float(risk_over.get("slippage_cap_bps", 40.0)) * 0.7
            if proxies.get("delta_hat", 0.0) > 0.6 and proxies.get("gamma_hat", 0.0) > 0.0 and spread_bps and spread_bps < 10.0:
                # allow slightly looser guard in favorable microtrend with tight spreads
                dyn_over["price_guard_bps"] = max(float(risk_over.get("price_guard_bps", 80.0)), 120.0)
        except Exception:
            pass
        # merge dyn_over into risk_over without mutating stored runtime
        risk_over = {**risk_over, **dyn_over}
        # Risk evaluation (fail-safe: block on error)
        try:
            allowed, reason = self._risk.evaluate(req.model_dump(), ref_price=ref_mid, spread_bps=spread_bps,
                                                  balances=balances, positions=None, funding_rate_bps=funding_bps, is_perp=is_perp,
                                                  overrides=risk_over)
        except Exception as e:
            await self._worm.append({
                "ts_ns": int(time.time() * 1e9),
                "event": "risk_error_block",
                "payload": {"symbol": req.symbol, "err": str(e), "client_id": req.client_id},
            })
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="RISK_ERROR", ts_ns=int(time.time() * 1e9))
        if not allowed:
            # Audit risk decision
            await self._worm.append({
                "ts_ns": int(time.time() * 1e9),
                "event": "risk_pretrade_block",
                "payload": {"symbol": req.symbol, "reason": reason, "client_id": req.client_id},
            })
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=f"RISK_BLOCKED:{reason}", ts_ns=int(time.time() * 1e9))
        # Idempotency guard (best-effort with Redis)
        cache_key = None
        if self._redis.enabled() and req.idempotency_key:
            cache_key = f"exec:ack:{req.idempotency_key}"
            idem_key = f"exec:idem:{venue}:{req.idempotency_key}"
            try:
                set_ok = await self._redis.client.set(idem_key, "1", ex=120, nx=True)  # type: ignore
                if not set_ok:
                    cached = await self._redis.get_json(cache_key)
                    if cached:
                        return OrderAck(**cached)
                    return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="DUPLICATE_IDEMPOTENCY", ts_ns=int(time.time()*1e9))
            except Exception:
                pass
        # In-memory idempotency fallback when Redis is disabled
        mem_cache_key = None
        if (not self._redis.enabled()) and req.idempotency_key:
            # simple in-proc TTL cache
            now = time.time()
            if not hasattr(self, "_idem_mem"):
                self._idem_mem: Dict[str, tuple[dict, float]] = {}
            mem_cache_key = req.idempotency_key
            v = self._idem_mem.get(mem_cache_key)
            if v and now < float(v[1]):
                try:
                    return OrderAck(**v[0])
                except Exception:
                    pass
        # Place with minimal retries on transient failures
        attempts = 2
        last_ack = None
        for i in range(attempts):
            try:
                ack = await asyncio.wait_for(adapter.place_order(req, timeout_ms=timeout_ms), timeout=timeout_ms / 1000.0)
            except asyncio.TimeoutError:
                ack = OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="ADAPTER_TIMEOUT", ts_ns=int(time.time()*1e9))
            except Exception as e:
                ack = OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=f"ADAPTER_ERROR:{str(e)}", ts_ns=int(time.time()*1e9))
            last_ack = ack
            if ack.accepted or i == attempts - 1:
                break
            # simple backoff
            await asyncio.sleep(0.05 * (2 ** i))
        ack = last_ack if last_ack is not None else OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="NO_ACK", ts_ns=int(time.time()*1e9))
        if not ack.accepted:
            await self._cb_record_fail(venue)
        else:
            # success path: optionally reset memory counter
            try:
                setattr(self, f"{venue}:_memfail", 0)
            except Exception:
                pass
        # Cache ack for idempotent lookup
        if cache_key and self._redis.enabled():
            try:
                await self._redis.set_json(cache_key, ack.model_dump(), ex=120)
            except Exception:
                pass
        # In-memory idempotency cache set (fallback)
        if (not self._redis.enabled()) and mem_cache_key:
            try:
                now = time.time()
                # prune occasionally to avoid unbounded growth
                if not hasattr(self, "_idem_mem"):
                    self._idem_mem = {}
                if len(self._idem_mem) > 2000:
                    # drop expired entries
                    self._idem_mem = {k: v for k, v in self._idem_mem.items() if now < float(v[1])}
                self._idem_mem[mem_cache_key] = (ack.model_dump(), now + 120.0)
            except Exception:
                pass
        # store ack ts for ack->fill latency (idempotency as client order id)
        try:
            from backend.observability.metrics import ACKS_TOTAL
            ACKS_TOTAL.labels(venue).inc()
        except Exception:
            pass
        if ack.accepted:
            cid = req.idempotency_key
            if cid:
                self._ack_ts[cid] = time.time()
        return ack

    async def start(self) -> None:
        # Start user-data streams for default adapters
        for name, adapter in self.adapters.items():
            start = getattr(adapter, "start", None)
            if callable(start):
                handler = getattr(adapter, "set_fill_handler", None)
                if callable(handler):
                    handler(self._on_fill)
                try:
                    await start()
                except Exception:
                    continue
        # Start user-data streams for account-specific adapters discovered via accounts.yaml
        # Pre-create adapters for configured accounts and start their streams
        ex_cfg = (self._accounts.cfg.get("exchanges") or {})
        for venue, vcfg in ex_cfg.items():
            accounts = (vcfg.get("accounts") or {}).keys()
            for acc in accounts:
                try:
                    adp = self._get_adapter(venue, acc)
                    start = getattr(adp, "start", None)
                    if callable(start):
                        handler = getattr(adp, "set_fill_handler", None)
                        if callable(handler):
                            handler(self._on_fill)
                        await start()
                except Exception:
                    continue

    async def stop(self) -> None:
        # Stop default adapters
        for name, adapter in self.adapters.items():
            stop = getattr(adapter, "stop", None)
            if callable(stop):
                try:
                    await stop()
                except Exception:
                    continue
        # Stop account-specific adapters
        try:
            for venue, amap in self._account_adapters.items():
                for acc, adapter in amap.items():
                    stop = getattr(adapter, "stop", None)
                    if callable(stop):
                        try:
                            await stop()
                        except Exception:
                            continue
        except Exception:
            pass

    async def balances(self, venue: str, account: str | None = None):
        adapter = self._get_adapter(venue, account)
        return await adapter.balances()

    async def positions(self, venue: str, account: str | None = None):
        adapter = self._get_adapter(venue, account)
        return await adapter.positions()

    async def cancel(self, venue: str, broker_order_id: str, account: str | None = None, timeout_ms: int = 200) -> bool:
        adapter = self._get_adapter(venue, account)
        return await asyncio.wait_for(adapter.cancel(broker_order_id, timeout_ms=timeout_ms), timeout=timeout_ms / 1000.0)

    def cofre_snapshot(self):
        return self._cofre.snapshot()

    def register_intent(self, idempotency_key: str, plugin_id: str, account: str | None = None) -> None:
        # Store mapping to resolve plugin policy at fill time
        if idempotency_key:
            self._intent_map[idempotency_key] = plugin_id

    def _get_adapter(self, venue: str, account: str | None) -> ExchangeAdapter:
        if not account:
            return self.adapters[venue]
        # cache per account
        cache = self._account_adapters.setdefault(venue, {})
        if account in cache:
            return cache[account]
        keys = self._accounts.get_keys(venue, account)
        tos = load_policies().get("exchange_tos", {})
        if venue == "binance":
            mode = keys.get("mode") or self._binance_mode
            adp = BinanceAdapter(tos, mode=mode, api_key=keys.get("api_key"), api_secret=keys.get("api_secret"))
        elif venue == "bybit":
            adp = BybitAdapter(tos, category=(keys.get("category") or "spot"), api_key=keys.get("api_key"), api_secret=keys.get("api_secret"))
        elif venue == "kraken":
            adp = KrakenAdapter(tos, api_key=keys.get("api_key"), api_secret_b64=keys.get("api_secret"))
        else:
            adp = self.adapters[venue]
        try:
            setattr(adp, "_account_label", account)
        except Exception:
            pass
        cache[account] = adp
        return adp

    def _select_venue(self, symbol: str, account: str | None) -> str:
        allowed = self._pairs.allowed_venues(symbol)
        ex_cfg = (self._accounts.cfg.get("exchanges") or {})
        for v in allowed:
            if account:
                if ex_cfg.get(v, {}).get("accounts", {}).get(account):
                    return v
            else:
                if ex_cfg.get(v, {}).get("accounts"):
                    return v
        return allowed[0] if allowed else "binance"

    def _get_instr_rules(self, symbol: str, venue: str) -> dict:
        ins = _load_yaml(os.getenv("INSTRUMENTS_FILE", "backend/config/instruments.yaml"))
        rules: dict = {}
        for it in ins.get("symbols", []):
            if it.get("symbol") == symbol:
                pr = it.get("precision", {}) or {}
                steps = it.get("steps", {}) or {}
                rules.update({
                    "price_dp": pr.get("price_dp"),
                    "qty_dp": pr.get("qty_dp"),
                    "price_step": steps.get("price"),
                    "qty_step": steps.get("qty"),
                    "min_qty": it.get("min_qty"),
                    "min_notional_usd": it.get("min_notional_usd"),
                })
                vrs = (it.get("venues_rules") or {}).get(venue, {})
                # venue-specific overrides
                for k in ("price_step","qty_step","min_qty","min_notional_usd","price_dp","qty_dp"):
                    if vrs.get(k) is not None:
                        rules[k] = vrs.get(k)
                break
        return rules

    def _quantize_to_step(self, value: float, step: float) -> float:
        if not step or step <= 0:
            return value
        return round(round(value / step) * step, 12)

    def _ceil_to_step(self, value: float, step: float) -> float:
        if not step or step <= 0:
            return value
        import math
        return round(math.ceil(value / step) * step, 12)

    async def _adjust_order(self, venue: str, req: OrderRequest, ref_mid: float | None) -> OrderRequest:
        rules = self._get_instr_rules(req.symbol, venue)
        d = req.model_dump()
        qty = float(d.get("qty", 0))
        px = d.get("price")
        # Round by steps or dp
        qty_step = rules.get("qty_step")
        price_step = rules.get("price_step")
        if qty_step:
            qty = self._quantize_to_step(qty, float(qty_step))
        elif rules.get("qty_dp") is not None:
            qty = round(qty, int(rules["qty_dp"]))
        if px is not None:
            pxv = float(px)
            if price_step:
                pxv = self._quantize_to_step(pxv, float(price_step))
            elif rules.get("price_dp") is not None:
                pxv = round(pxv, int(rules["price_dp"]))
            d["price"] = pxv
        # Enforce min_qty
        min_qty = rules.get("min_qty")
        if min_qty is not None and qty < float(min_qty):
            qty = float(min_qty)
            if qty_step:
                qty = self._ceil_to_step(qty, float(qty_step))
        # Enforce min_notional_usd
        min_notional = rules.get("min_notional_usd")
        if min_notional and ref_mid and ref_mid > 0:
            need_qty = float(min_notional) / float(ref_mid)
            if qty < need_qty:
                qty = need_qty
                if qty_step:
                    qty = self._ceil_to_step(qty, float(qty_step))
                elif rules.get("qty_dp") is not None:
                    qty = round(qty + (10 ** -int(rules["qty_dp"])), int(rules["qty_dp"]))
        d["qty"] = qty
        return OrderRequest(**d)

    async def _fetch_mid_and_spread_bps(self, venue: str, symbol: str) -> tuple[float, float]:
        # Hot-path: use Redis cache only; avoid blocking HTTP here.
        try:
            if self._redis.enabled():
                key = f"md:quote:{venue}:{symbol}"
                cached = await self._redis.get_json(key)
                if cached:
                    bid = float(cached.get("bid", 0))
                    ask = float(cached.get("ask", 0))
                    mid = (bid + ask) / 2.0 if bid and ask else 0.0
                    spread_bps = ((ask - bid) / mid * 10_000) if mid else 0.0
                    return mid, spread_bps
        except Exception:
            pass
        # Miss: return neutral values; Risk Engine will handle conservative checks.
        return 0.0, 0.0

    def _venue_symbol(self, symbol: str, venue: str) -> str:
        ins = _load_yaml(os.getenv("INSTRUMENTS_FILE", "backend/config/instruments.yaml"))
        for it in ins.get("symbols", []):
            if it.get("symbol") == symbol:
                return (it.get("venues") or {}).get(venue, symbol)
        return symbol

    async def _fetch_funding_bps(self, venue: str, symbol: str, adapter: ExchangeAdapter) -> float:
        try:
            if venue == "binance" and getattr(adapter, "_mode", "") == "futures":
                vsym = self._venue_symbol(symbol, venue)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
                    r = await s.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": vsym})
                    data = await r.json()
                    fr = float(data.get("lastFundingRate", 0.0))
                    return fr * 10_000.0
            if venue == "bybit" and getattr(adapter, "_category", "") in ("linear", "inverse"):
                vsym = self._venue_symbol(symbol, venue)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
                    # v5 tickers returns fundingRate for linear/inverse
                    cat = getattr(adapter, "_category", "linear")
                    r = await s.get("https://api.bybit.com/v5/market/tickers", params={"category": cat, "symbol": vsym})
                    data = await r.json()
                    lst = (data.get("result") or {}).get("list", [])
                    it = lst[0] if lst else {}
                    fr = float(it.get("fundingRate", 0.0))
                    return fr * 10_000.0
            if venue == "kraken":
                # Kraken spot adapter; if futures added later, plug funding endpoint here
                return 0.0
        except Exception:
            return 0.0
        return 0.0

    async def _on_fill(self, ev: Dict) -> None:
        # Audit to WORM
        await self._worm.append({
            "ts_ns": int(time.time() * 1e9),
            "event": "fill",
            "payload": ev,
        })
        try:
            await self._pub.publish("fills", ev)
        except Exception:
            pass
        if self._mongo.enabled():
            await self._mongo.record_fill({"symbol": ev.get("s") or ev.get("o", {}).get("s"), "ts": int(time.time()*1e3), **ev})
        # TCA estimate (Binance example)
        fees_cfg = self._tos_cfg.get("exchanges", {}).get("binance", {}).get("fees", {})
        tca = estimate_costs_bps(ev, fees_cfg, market=self._binance_mode)
        await self._worm.append({
            "ts_ns": int(time.time() * 1e9),
            "event": "tca_estimate",
            "payload": tca,
        })
        try:
            await self._pub.publish("tca", {"symbol": symbol, **tca})
        except Exception:
            pass
        # Metrics: venue detection + fee_to_edge placeholder (edge unknown, set to fee_bps only)
        etype = ev.get("e") or ev.get("event")
        if etype == "executionReport":
            venue_lbl = "binance"
        elif etype == "BYBIT_ORDER":
            venue_lbl = "bybit"
        elif etype == "KRAKEN_TRADE":
            venue_lbl = "kraken"
        else:
            venue_lbl = "binance"
        symbol = (ev.get("s") or ev.get("o", {}).get("s") or "").upper()
        FEE_TO_EDGE.labels("jerico", symbol).set(tca.get("fee_bps", 0.0) / max(1.0, 100.0))
        # Fills total and ack->fill latency
        try:
            from backend.observability.metrics import FILLS_TOTAL, ACK_TO_FILL
            side = (ev.get("S") or ev.get("o", {}).get("S") or "").upper()
            FILLS_TOTAL.labels(venue_lbl, symbol, side or "").inc()
            client_order_id = ev.get("c") or ev.get("o", {}).get("c")
            if client_order_id and client_order_id in self._ack_ts:
                dt = max(0.0, time.time() - float(self._ack_ts[client_order_id]))
                ACK_TO_FILL.labels(venue_lbl, symbol).observe(dt)
                # cleanup to avoid leak
                self._ack_ts.pop(client_order_id, None)
        except Exception:
            pass
        # Cofre sweep (using client order id as proxy to plugin id if available)
        client_order_id = ev.get("c") or ev.get("o", {}).get("c")  # clientOrderId
        cofre_policy = {}
        plugin_id = None
        # resolve plugin id by intent map
        if client_order_id and client_order_id in self._intent_map:
            plugin_id = self._intent_map.get(client_order_id)
        # fallback: some clients may set client_id == manifest id
        if not plugin_id:
            plugin_id = ev.get("C") or ev.get("o", {}).get("C")  # origClientOrderId, optional
        if plugin_id and plugin_id in self._pm.loaded:
            cofre_policy = self._pm.loaded[plugin_id].cofre_policy
        # Query balances on the same venue and apply sweep
        try:
            adapter = self.adapters.get(venue_lbl)
            bal = {}
            if adapter:
                try:
                    bal = await asyncio.wait_for(adapter.balances(), timeout=0.3)
                except Exception:
                    bal = {}
            usdt = 0.0
            for ccy, b in bal.items():
                if ccy.upper() in ("USDT", "BUSD"):
                    usdt += float(b.total)
            cofre_over = await self._runtime.get_overrides("cofre", plugin_id)
            sweep_amt = await self._cofre.sweep_if_needed(account="acc#1", policy=cofre_policy, balance_usdt=usdt, overrides=cofre_over)
            if sweep_amt > 0:
                AGGR_EXPLOSIONS.labels("acc#1", "binance").inc()
                from backend.observability.metrics import COFRE_SWEEP_USDT
                COFRE_SWEEP_USDT.inc(sweep_amt)
                # Executa sweep real na venue correspondente (best effort; audita ledger de qualquer forma)
                try:
                    if venue_lbl == "binance" and self._cofre_exec_binance:
                        await self._cofre_exec_binance.sweep(sweep_amt, reason="auto")
                    elif venue_lbl == "bybit" and self._cofre_exec_bybit:
                        await self._cofre_exec_bybit.sweep(sweep_amt, reason="auto")
                except Exception:
                    pass
            AGGR_COFRE_BAL.set(self._cofre.snapshot().get("acc#1", {}).get("cofre_usdt", 0.0))
        except Exception:
            AGGR_RESETS.labels("acc#1", "sweep_error").inc()
        # Perf sample (best-effort)
        try:
            price = float(ev.get("L") or ev.get("p") or ev.get("o", {}).get("ap") or 0.0)
            qty = float(ev.get("l") or ev.get("q") or ev.get("o", {}).get("l") or 0.0)
            side = (ev.get("S") or ev.get("o", {}).get("S") or "").upper()
            idem = client_order_id
            if price and qty and side:
                await self._perf.record_fill(venue=venue_lbl, symbol=symbol, side=side, price=price, qty=qty, idem=idem)
        except Exception:
            pass
