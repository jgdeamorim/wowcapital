from __future__ import annotations
from typing import Dict, Optional, Callable, Awaitable
import os
import hmac
import hashlib
import aiohttp
import asyncio
from time import time
from urllib.parse import urlencode
from backend.core.contracts import OrderRequest, OrderAck, Position, Balance
from .base import ExchangeAdapter
from backend.common.rate_limiter import TokenBucket
from backend.common.ratelimit_redis import RedisRateLimiter
from backend.common.secure_env import get_secret


class BybitAdapter(ExchangeAdapter):
    name = "bybit"

    def __init__(self, tos: Dict, category: Optional[str] = None, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self._tos = tos.get("exchanges", {}).get("bybit", {})
        # Lazy credentials: prefer params, then env; only required on signed calls
        self._api_key: Optional[str] = api_key or os.getenv("BYBIT_API_KEY")
        self._api_secret: Optional[str] = api_secret or os.getenv("BYBIT_API_SECRET")
        testnet = os.getenv("BYBIT_TESTNET", "0")
        self._base = "https://api-testnet.bybit.com" if testnet in ("1", "true", "TRUE") else "https://api.bybit.com"
        self._category = (category or os.getenv("BYBIT_CATEGORY", "spot")).lower()  # spot|linear|inverse
        rps = float(self._tos.get("rate_limit_rps", 10))
        self._rps = rps
        self._bucket = TokenBucket(rate_per_sec=self._rps, burst=int(self._rps * 2))
        self._rl_dist = RedisRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._on_fill: Optional[Callable[[Dict], Awaitable[None]]] = None

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

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _sign(self, ts_ms: str, recv_window: str, query_or_body: str) -> str:
        if not self._api_key or not self._api_secret:
            raise RuntimeError("Missing BYBIT credentials for signed request")
        to_sign = ts_ms + self._api_key + recv_window + query_or_body
        return hmac.new(self._api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()

    async def healthcheck(self) -> bool:
        if not self._api_key or not self._api_secret:
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="missing api credentials", ts_ns=int(time() * 1e9))
        session = await self._get_session()
        async with session.get(self._base + "/v5/market/time", timeout=5) as r:
            return r.status == 200

    async def place_order(self, req: OrderRequest, *, timeout_ms: int) -> OrderAck:
        if not self._api_key or not self._api_secret:
            return False
        session = await self._get_session()
        path = "/v5/order/create"
        recv_window = "5000"
        ts_ms = str(int(time() * 1000))
        body = {
            "category": self._category,
            "symbol": req.symbol,
            "side": "Buy" if req.side.upper() == "BUY" else "Sell",
            "orderType": "Market" if req.order_type == "MARKET" else "Limit",
            "qty": str(req.qty),
            "orderLinkId": req.idempotency_key,
        }
        if req.order_type == "LIMIT" and req.price is not None:
            body["price"] = str(req.price)
            tif = req.tif if req.tif in ("GTC", "IOC", "FOK") else "GTC"
            body["timeInForce"] = tif
        body_str = json_dumps(body)
        sign = self._sign(ts_ms, recv_window, body_str)
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts_ms,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("order"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("order"))
        try:
            async with session.post(self._base + path, data=body_str, headers=headers, timeout=timeout_ms / 1000.0) as r:
                data = await r.json()
                ret = data.get("retCode", data.get("ret_code", -1))
                if ret != 0:
                    return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=str(data), ts_ns=int(time() * 1e9))
                boid = data.get("result", {}).get("orderId") or data.get("result", {}).get("orderLinkId")
                return OrderAck(client_id=req.client_id, broker_order_id=str(boid) if boid else None, accepted=True, message="", ts_ns=int(time() * 1e9))
        except Exception as e:
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=str(e), ts_ns=int(time() * 1e9))

    async def cancel(self, broker_order_id: str, *, timeout_ms: int) -> bool:
        if not self._api_key or not self._api_secret:
            return {}
        session = await self._get_session()
        path = "/v5/order/cancel"
        recv_window = "5000"
        ts_ms = str(int(time() * 1000))
        body = {
            "category": self._category,
            "orderLinkId": broker_order_id,
        }
        body_str = json_dumps(body)
        sign = self._sign(ts_ms, recv_window, body_str)
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts_ms,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("cancel"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("cancel"))
        async with session.post(self._base + path, data=body_str, headers=headers, timeout=timeout_ms / 1000.0) as r:
            data = await r.json()
            return (data.get("retCode", -1) == 0)

    async def positions(self) -> Dict[str, Position]:
        if self._category == "spot":
            return {}
        session = await self._get_session()
        path = "/v5/position/list"
        recv_window = "5000"
        ts_ms = str(int(time() * 1000))
        query = {"category": self._category}
        query_str = urlencode(query)
        sign = self._sign(ts_ms, recv_window, query_str)
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts_ms,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": sign,
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("positions"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("positions"))
        async with session.get(self._base + path + "?" + query_str, headers=headers, timeout=5) as r:
            data = await r.json()
            out: Dict[str, Position] = {}
            for it in data.get("result", {}).get("list", []) or []:
                sym = it.get("symbol")
                qty = float(it.get("size", 0))
                if not qty:
                    continue
                side = it.get("side", "").upper()
                out[sym] = Position(symbol=sym, qty=qty, side=("LONG" if side == "BUY" else "SHORT"))
            return out

    async def balances(self) -> Dict[str, Balance]:
        session = await self._get_session()
        path = "/v5/account/wallet-balance"
        recv_window = "5000"
        ts_ms = str(int(time() * 1000))
        query = {"accountType": "UNIFIED"}
        query_str = urlencode(query)
        sign = self._sign(ts_ms, recv_window, query_str)
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts_ms,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": sign,
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("balances"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("balances"))
        async with session.get(self._base + path + "?" + query_str, headers=headers, timeout=5) as r:
            data = await r.json()
            out: Dict[str, Balance] = {}
            for acct in data.get("result", {}).get("list", []) or []:
                for c in acct.get("coin", []) or []:
                    ccy = c.get("coin")
                    free = float(c.get("availableToWithdraw", 0))
                    total = float(c.get("walletBalance", 0))
                    out[ccy] = Balance(ccy=ccy, free=free, used=max(0.0, total - free), total=total)
            return out

    # --- WS Private (skeleton) ---
    def set_fill_handler(self, cb: Callable[[Dict], Awaitable[None]]):
        self._on_fill = cb

    async def start(self) -> None:
        # Bybit private WS auth and order subscription
        # Docs: wss://stream.bybit.com/v5/private (or testnet) with auth op
        if not self._api_key or not self._api_secret:
            return
        url = "wss://stream.bybit.com/v5/private"
        if "testnet" in self._base:
            url = "wss://stream-testnet.bybit.com/v5/private"
        session = await self._get_session()
        recv_window = "5000"
        backoff = 1.0
        async def _run():
            nonlocal backoff
            while True:
                try:
                    async with session.ws_connect(url, heartbeat=15) as ws:
                        ts_ms = str(int(time() * 1000))
                        sign = self._sign(ts_ms, recv_window, "")
                        auth_msg = {"op": "auth", "args": [self._api_key, ts_ms, sign, recv_window]}
                        await ws.send_json(auth_msg)
                        sub = {"op": "subscribe", "args": ["order"]}
                        await ws.send_json(sub)
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = msg.json()
                                if isinstance(data, dict) and data.get("topic") == "order":
                                    for it in data.get("data", []) or []:
                                        status = (it.get("orderStatus") or "").lower()
                                        if status == "filled" and self._on_fill:
                                            await self._on_fill({
                                                "e": "BYBIT_ORDER",
                                                "s": it.get("symbol"),
                                                "S": (it.get("side") or "").upper(),
                                                "c": it.get("orderLinkId"),
                                                "p": it.get("avgPrice") or it.get("price"),
                                                "q": it.get("qty") or it.get("leavesQty"),
                                                "t": it.get("orderId"),
                                            })
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
        self._ws_task = asyncio.create_task(_run())

    async def stop(self) -> None:
        try:
            if self._ws_task:
                self._ws_task.cancel()
        except Exception:
            pass


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"))
