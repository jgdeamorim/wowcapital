from __future__ import annotations
from typing import Dict, Optional, Callable, Awaitable
import os
import base64
import hmac
import hashlib
from time import time
import aiohttp
import asyncio
from urllib.parse import urlencode
from backend.core.contracts import OrderRequest, OrderAck, Position, Balance
from .base import ExchangeAdapter
from backend.common.rate_limiter import TokenBucket
from backend.common.ratelimit_redis import RedisRateLimiter
from backend.common.secure_env import get_secret


class KrakenAdapter(ExchangeAdapter):
    name = "kraken"

    def __init__(self, tos: Dict, api_key: Optional[str] = None, api_secret_b64: Optional[str] = None):
        self._tos = tos.get("exchanges", {}).get("kraken", {})
        # Lazy credentials: prefer params/env; enforce presence on signed calls
        self._api_key: Optional[str] = api_key or os.getenv("KRAKEN_API_KEY")
        self._api_secret_b64: Optional[str] = api_secret_b64 or os.getenv("KRAKEN_API_SECRET")
        self._base = "https://api.kraken.com"
        rps = float(self._tos.get("rate_limit_rps", 5))
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

    async def healthcheck(self) -> bool:
        session = await self._get_session()
        async with session.get(self._base + "/0/public/Time", timeout=5) as r:
            return r.status == 200

    def _sign(self, path: str, data: Dict[str, str]) -> str:
        if not self._api_secret_b64:
            raise RuntimeError("Missing KRAKEN_API_SECRET for signed request")
        postdata = urlencode(data)
        sha256 = hashlib.sha256((data["nonce"] + postdata).encode()).digest()
        message = path.encode() + sha256
        secret = base64.b64decode(self._api_secret_b64)
        sig = hmac.new(secret, message, hashlib.sha512)
        return base64.b64encode(sig.digest()).decode()

    async def place_order(self, req: OrderRequest, *, timeout_ms: int) -> OrderAck:
        if not self._api_key or not self._api_secret_b64:
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message="missing api credentials", ts_ns=int(time() * 1e9))
        session = await self._get_session()
        path = "/0/private/AddOrder"
        nonce = str(int(time() * 1000))
        data = {
            "nonce": nonce,
            "ordertype": "market" if req.order_type == "MARKET" else "limit",
            "type": "buy" if req.side.upper() == "BUY" else "sell",
            "volume": str(req.qty),
            "pair": req.symbol,
        }
        if req.order_type == "LIMIT" and req.price is not None:
            data["price"] = str(req.price)
        headers = {
            "API-Key": self._api_key,
            "API-Sign": self._sign(path, data),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("order"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("order"))
        try:
            async with session.post(self._base + path, data=urlencode(data), headers=headers, timeout=timeout_ms / 1000.0) as r:
                resp = await r.json()
                if resp.get("error"):
                    return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=str(resp["error"]), ts_ns=int(time() * 1e9))
                txids = resp.get("result", {}).get("txid", [])
                boid = txids[0] if txids else None
                return OrderAck(client_id=req.client_id, broker_order_id=boid, accepted=True, message="", ts_ns=int(time() * 1e9))
        except Exception as e:
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=str(e), ts_ns=int(time() * 1e9))

    async def cancel(self, broker_order_id: str, *, timeout_ms: int) -> bool:
        if not self._api_key or not self._api_secret_b64:
            return False
        session = await self._get_session()
        path = "/0/private/CancelOrder"
        nonce = str(int(time() * 1000))
        data = {"nonce": nonce, "txid": broker_order_id}
        headers = {
            "API-Key": self._api_key,
            "API-Sign": self._sign(path, data),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("cancel"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("cancel"))
        async with session.post(self._base + path, data=urlencode(data), headers=headers, timeout=timeout_ms / 1000.0) as r:
            resp = await r.json()
            return not resp.get("error")

    async def positions(self) -> Dict[str, Position]:
        return {}

    async def balances(self) -> Dict[str, Balance]:
        if not self._api_key or not self._api_secret_b64:
            return {}
        session = await self._get_session()
        path = "/0/private/Balance"
        nonce = str(int(time() * 1000))
        data = {"nonce": nonce}
        headers = {
            "API-Key": self._api_key,
            "API-Sign": self._sign(path, data),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        await self._bucket.acquire()
        await self._rl_dist.acquire(self._rl_key("balances"), rate_per_sec=self._rps, burst=int(self._rps * 2), weight=self._rl_weight("balances"))
        async with session.post(self._base + path, data=urlencode(data), headers=headers, timeout=5) as r:
            resp = await r.json()
            out: Dict[str, Balance] = {}
            if resp.get("error"):
                return out
            for ccy, amt in (resp.get("result") or {}).items():
                total = float(amt)
                out[ccy] = Balance(ccy=ccy, free=total, used=0.0, total=total)
            return out

    # --- WS Private (skeleton) ---
    def set_fill_handler(self, cb: Callable[[Dict], Awaitable[None]]):
        self._on_fill = cb

    async def start(self) -> None:
        # Kraken private WS requires token from REST GetWebSocketsToken, then subscribe ownTrades
        if not self._api_key or not self._api_secret:
            return
        session = await self._get_session()
        # Get token
        path = "/0/private/GetWebSocketsToken"
        nonce = str(int(time() * 1000))
        data = {"nonce": nonce}
        headers = {
            "API-Key": self._api_key,
            "API-Sign": self._sign(path, data),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        await self._bucket.acquire()
        async with session.post(self._base + path, data=urlencode(data), headers=headers, timeout=5) as r:
            resp = await r.json()
            token = (resp.get("result") or {}).get("token")
        if not token:
            return
        url = "wss://ws.kraken.com/"
        backoff = 1.0
        async def _run():
            nonlocal backoff
            while True:
                try:
                    async with session.ws_connect(url, heartbeat=15) as ws:
                        sub = {"event": "subscribe", "subscription": {"name": "ownTrades", "token": token}}
                        await ws.send_json(sub)
                        backoff = 1.0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = msg.json()
                                if isinstance(data, list) and len(data) >= 4 and data[2] == "ownTrades":
                                    trades = data[1]
                                    if isinstance(trades, dict):
                                        for _, t in trades.items():
                                            if self._on_fill:
                                                await self._on_fill({
                                                    "e": "KRAKEN_TRADE",
                                                    "s": t.get("pair"),
                                                    "S": (t.get("type") or "").upper(),
                                                    "p": t.get("price"),
                                                    "q": t.get("vol"),
                                                    "t": t.get("ordertxid"),
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
