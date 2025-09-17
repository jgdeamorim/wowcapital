from __future__ import annotations
from typing import Dict, Optional, Callable, Awaitable
from time import time
import asyncio
import hmac
import hashlib
import aiohttp
import json
from urllib.parse import urlencode
from backend.core.contracts import OrderRequest, OrderAck, Position, Balance
from .base import ExchangeAdapter
import os
from backend.common.rate_limiter import TokenBucket
from backend.common.ratelimit_redis import RedisRateLimiter


class BinanceAdapter(ExchangeAdapter):
    name = "binance"

    def __init__(self, tos: Dict, mode: str = "spot", api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self._tos = tos.get("exchanges", {}).get("binance", {})
        self._mode = mode  # "spot" | "futures"
        if self._mode == "futures":
            self._base = "https://fapi.binance.com"
        else:
            self._base = "https://api.binance.com"
        # Lazy credentials: pull from params or env if present; enforce only on signed calls
        self._api_key: Optional[str] = api_key or os.getenv("BINANCE_API_KEY")
        self._api_secret: Optional[str] = api_secret or os.getenv("BINANCE_API_SECRET")
        rps = float(self._tos.get("rate_limit_rps", 10))
        self._rps = rps
        self._bucket = TokenBucket(rate_per_sec=self._rps, burst=int(self._rps * 2))
        self._rl_dist = RedisRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._listen_key: Optional[str] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._on_fill: Optional[Callable[[Dict], Awaitable[None]]] = None
        self._seen: set[str] = set()

    def _rl_key(self, route: str) -> str:
        acct = getattr(self, "_account_label", "default")
        return f"{self.name}:{acct}:{route}"

    def _rl_weight(self, route: str) -> int:
        try:
            w = (self._tos.get("weights") or {}).get(route)
            if w is not None:
                return int(w)
        except Exception:
            pass
        defaults = {"order": 1, "cancel": 1, "balances": 5, "positions": 10}
        return defaults.get(route, 1)

    async def _server_time(self, session: aiohttp.ClientSession) -> int:
        path = "/fapi/v1/time" if self._mode == "futures" else "/api/v3/time"
        async with session.get(self._base + path, timeout=5) as r:
            data = await r.json()
            return int(data["serverTime"])

    def _sign(self, params: Dict[str, str]) -> str:
        if not self._api_secret:
            raise RuntimeError("Missing BINANCE_API_SECRET for signed request")
        query = urlencode(params)
        return hmac.new(self._api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    async def healthcheck(self) -> bool:
        session = await self._get_session()
        ts = await self._server_time(session)
        return ts > 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-MBX-APIKEY": self._api_key} if self._api_key else None
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def place_order(self, req: OrderRequest, *, timeout_ms: int) -> OrderAck:
        path = "/fapi/v1/order" if self._mode == "futures" else "/api/v3/order"
        if not self._api_key or not self._api_secret:
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="missing api credentials", ts_ns=int(time() * 1e9))
        order_type = req.order_type
        params: Dict[str, str] = {
            "symbol": req.symbol,
            "side": req.side,
            "type": "MARKET" if order_type == "MARKET" else "LIMIT",
            "newClientOrderId": req.idempotency_key,
            "recvWindow": "5000",
        }
        if order_type == "LIMIT" and req.price is not None:
            params["price"] = str(req.price)
            tif = req.tif if req.tif in ("GTC", "IOC", "FOK") else "GTC"
            params["timeInForce"] = tif
        params["quantity"] = str(req.qty)

        session = await self._get_session()
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("order"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("order"))
        ts = await self._server_time(session)
        params["timestamp"] = str(ts)
        params["signature"] = self._sign(params)
        try:
            async with session.post(self._base + path, data=params, timeout=timeout_ms / 1000.0) as r:
                data = await r.json()
                if r.status >= 400:
                    return OrderAck(
                        client_id=req.client_id,
                        broker_order_id=None,
                        accepted=False,
                        message=str(data),
                        ts_ns=int(time() * 1e9),
                    )
                boid: Optional[str] = data.get("orderId") or data.get("clientOrderId")
                return OrderAck(
                    client_id=req.client_id,
                    broker_order_id=str(boid) if boid else None,
                    accepted=True,
                    message="",
                    ts_ns=int(time() * 1e9),
                )
        except Exception as e:
            return OrderAck(
                client_id=req.client_id,
                broker_order_id=None,
                accepted=False,
                message=str(e),
                ts_ns=int(time() * 1e9),
            )

    async def cancel(self, broker_order_id: str, *, timeout_ms: int) -> bool:
        path = "/fapi/v1/order" if self._mode == "futures" else "/api/v3/order"
        if not self._api_key or not self._api_secret:
            return False
        params: Dict[str, str] = {
            "origClientOrderId": broker_order_id,
            "recvWindow": "5000",
        }
        session = await self._get_session()
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("cancel"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("cancel"))
        ts = await self._server_time(session)
        params["timestamp"] = str(ts)
        params["signature"] = self._sign(params)
        async with session.delete(self._base + path, params=params, timeout=timeout_ms / 1000.0) as r:
            return r.status < 400

    async def positions(self) -> Dict[str, Position]:
        if self._mode != "futures":
            return {}
        if not self._api_key or not self._api_secret:
            return {}
        path = "/fapi/v2/account"
        session = await self._get_session()
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("positions"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("positions"))
        ts = await self._server_time(session)
        params = {"timestamp": str(ts), "signature": self._sign({"timestamp": str(ts)})}
        async with session.get(self._base + path, params=params, timeout=5) as r:
            data = await r.json()
            out: Dict[str, Position] = {}
            for p in data.get("positions", []):
                qty = float(p.get("positionAmt", 0))
                if qty == 0:
                    continue
                sym = p.get("symbol")
                out[sym] = Position(symbol=sym, qty=abs(qty), side=("LONG" if qty > 0 else "SHORT"))
            return out

    async def balances(self) -> Dict[str, Balance]:
        path = "/api/v3/account" if self._mode == "spot" else "/fapi/v2/account"
        if not self._api_key or not self._api_secret:
            return {}
        session = await self._get_session()
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("balances"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("balances"))
        ts = await self._server_time(session)
        params = {"timestamp": str(ts)}
        params["signature"] = self._sign(params)
        async with session.get(self._base + path, params=params, timeout=5) as r:
            data = await r.json()
            out: Dict[str, Balance] = {}
            if self._mode == "spot":
                for b in data.get("balances", []):
                    ccy = b.get("asset")
                    free = float(b.get("free", 0))
                    locked = float(b.get("locked", 0))
                    out[ccy] = Balance(ccy=ccy, free=free, used=locked, total=free + locked)
            else:
                for a in data.get("assets", []):
                    ccy = a.get("asset")
                    wallet = float(a.get("walletBalance", 0))
                    out[ccy] = Balance(ccy=ccy, free=wallet, used=0.0, total=wallet)
            return out

    # --- User Data Stream (fills) ---
    async def start(self) -> None:
        """Start user-data stream and keepalive tasks."""
        session = await self._get_session()
        # Create listenKey
        path = "/fapi/v1/listenKey" if self._mode == "futures" else "/api/v3/userDataStream"
        await self._bucket.acquire()
        async with session.post(self._base + path, timeout=5) as r:
            data = await r.json()
            self._listen_key = data.get("listenKey")
        if not self._listen_key:
            return
        # Start WS task and keepalive
        self._ws_task = asyncio.create_task(self._ws_consume())
        self._keepalive_task = asyncio.create_task(self._keepalive())

    async def stop(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._ws_task:
            self._ws_task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def set_fill_handler(self, cb: Callable[[Dict], Awaitable[None]]):
        self._on_fill = cb

    async def _keepalive(self):
        session = await self._get_session()
        path = "/fapi/v1/listenKey" if self._mode == "futures" else "/api/v3/userDataStream"
        while True:
            try:
                await asyncio.sleep(30 * 60)
                if not self._listen_key:
                    continue
                await self._bucket.acquire()
                async with session.put(self._base + path, params={"listenKey": self._listen_key}, timeout=5) as _:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _ws_consume(self):
        if not self._listen_key:
            return
        if self._mode == "futures":
            ws_url = f"wss://fstream.binance.com/ws/{self._listen_key}"
        else:
            ws_url = f"wss://stream.binance.com:9443/ws/{self._listen_key}"
        session = await self._get_session()
        backoff = 1.0
        while True:
            try:
                async with session.ws_connect(ws_url, heartbeat=15) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self._handle_ws_event(data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

    async def _handle_ws_event(self, data: Dict):
        # Spot: event type "executionReport"; Futures: "ORDER_TRADE_UPDATE"
        et = data.get("e") or data.get("eventType") or data.get("event")
        if et in ("executionReport",):
            # Minimal parse: only fills
            status = data.get("X")  # EXECUTION_TYPE or orderStatus
            trade_id = str(data.get("t") or data.get("i") or "")
            key = f"spot:{trade_id}"
            if status == "FILLED" and self._on_fill and trade_id:
                if key in self._seen:
                    return
                self._seen.add(key)
                if len(self._seen) > 5000:
                    # prune
                    self._seen = set(list(self._seen)[-2000:])
                await self._on_fill(data)
        elif et in ("ORDER_TRADE_UPDATE",):
            od = data.get("o", {})
            status = od.get("X")
            trade_id = str(od.get("t") or od.get("i") or "")
            key = f"fut:{trade_id}"
            if status == "FILLED" and self._on_fill and trade_id:
                if key in self._seen:
                    return
                self._seen.add(key)
                if len(self._seen) > 5000:
                    self._seen = set(list(self._seen)[-2000:])
                await self._on_fill(data)
