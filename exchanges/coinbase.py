from __future__ import annotations
from typing import Dict, Optional, Callable, Awaitable
import os
import asyncio
import aiohttp
import json
import logging
from time import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
import base64

from backend.core.contracts import OrderRequest, OrderAck, Position, Balance
from .base import ExchangeAdapter
from backend.common.rate_limiter import TokenBucket
from backend.common.ratelimit_redis import RedisRateLimiter


class CoinbaseAdapter(ExchangeAdapter):
    """Adapter para Coinbase Advanced Trade (Spot)."""

    name = "coinbase"

    def __init__(self, tos: Dict, api_key: Optional[str] = None, private_key_path: Optional[str] = None, private_key_pem: Optional[str] = None):
        self._tos = tos.get("exchanges", {}).get("coinbase", {})
        self._logger = logging.getLogger(__name__)
        key_env = (
            os.getenv("COINBASE_CLIENT_KEY")
            or os.getenv("COINBASE_API_KEY")
            or os.getenv("COINBASE_KEY_ID")
        )
        self._api_key: Optional[str] = api_key or key_env

        pem_data = private_key_pem or os.getenv("COINBASE_PRIVATE_KEY_PEM")
        key_path = private_key_path or os.getenv("COINBASE_PRIVATE_KEY_PATH")
        key_file = os.getenv("COINBASE_KEY_FILE")

        if not pem_data and key_path:
            expanded = os.path.expanduser(key_path)
            if os.path.exists(expanded):
                with open(expanded, "rb") as f:
                    pem_data = f.read().decode()
            else:
                self._logger.warning("Coinbase private key path not found: %s", expanded)

        if not pem_data and key_file:
            json_path = os.path.expanduser(key_file)
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                pem_data = data.get("privateKey") or data.get("private_key")
                if not self._api_key:
                    name = data.get("name") or data.get("apiKey")
                    if name:
                        self._api_key = name
            else:
                self._logger.warning("Coinbase key file not found: %s", json_path)
        self._private_key = None
        if pem_data:
            self._private_key = serialization.load_pem_private_key(pem_data.encode(), password=None)

        self._base = os.getenv("COINBASE_BASE_URL", "https://api.coinbase.com/api/v3")
        from urllib.parse import urlparse
        parsed = urlparse(self._base)
        self._api_prefix = parsed.path.rstrip("/") or ""
        rps = float(self._tos.get("rate_limit_rps", 8))
        self._rps = rps
        self._bucket = TokenBucket(rate_per_sec=self._rps, burst=int(self._rps * 2))
        self._rl_dist = RedisRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._on_fill: Optional[Callable[[Dict], Awaitable[None]]] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _require_credentials(self):
        if not self._api_key or self._private_key is None:
            raise RuntimeError("Missing Coinbase credentials for signed request")

    def _sign(self, method: str, path: str, body: str, timestamp: str) -> str:
        self._require_credentials()
        method_u = method.upper()
        path_for_sig = path
        if self._api_prefix and not path.startswith(self._api_prefix):
            path_for_sig = f"{self._api_prefix}{path}" if path.startswith("/") else f"{self._api_prefix}/{path}"
        message = timestamp + method_u + path_for_sig + body
        signature = self._private_key.sign(message.encode(), ec.ECDSA(hashes.SHA256()))  # type: ignore[arg-type]
        r, s = decode_dss_signature(signature)
        size = 32
        r_bytes = r.to_bytes(size, byteorder="big")
        s_bytes = s.to_bytes(size, byteorder="big")
        return base64.b64encode(r_bytes + s_bytes).decode()

    async def _request(self, method: str, path: str, *, body: Optional[dict] = None, timeout_s: float = 5.0) -> Dict:
        session = await self._get_session()
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        timestamp = str(int(time()))
        headers = {
            "Content-Type": "application/json",
        }
        method_u = method.upper()
        path_for_sig = path
        if self._api_prefix and not path.startswith(self._api_prefix):
            path_for_sig = f"{self._api_prefix}{path}" if path.startswith("/") else f"{self._api_prefix}/{path}"
        needs_auth = "brokerage" in path_for_sig
        if needs_auth:
            self._require_credentials()
            headers.update({
                "CB-ACCESS-KEY": self._api_key,
                "CB-ACCESS-SIGNATURE": self._sign(method_u, path, body_str, timestamp),
                "CB-ACCESS-TIMESTAMP": timestamp,
            })

        url = self._base.rstrip("/") + path
        await self._bucket.acquire()
        await self._rl_dist.acquire(f"{self.name}:{path}", rate_per_sec=self._rps, burst=int(self._rps * 2))
        if method_u == "GET":
            async with session.get(url, headers=headers, timeout=timeout_s) as resp:
                resp.raise_for_status()
                return await resp.json()
        elif method_u == "POST":
            async with session.post(url, headers=headers, data=body_str or None, timeout=timeout_s) as resp:
                resp.raise_for_status()
                return await resp.json()
        else:
            async with session.request(method.upper(), url, headers=headers, data=body_str or None, timeout=timeout_s) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def healthcheck(self) -> bool:
        try:
            data = await self._request("GET", "/brokerage/products", timeout_s=4.0)
            return bool(data.get("products"))
        except Exception:
            return False

    def _product_id(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol.replace("/", "-")
        quotes = ["USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP"]
        for q in quotes:
            if symbol.endswith(q):
                base = symbol[:-len(q)]
                return f"{base}-{q}"
        if len(symbol) > 3:
            return f"{symbol[:-3]}-{symbol[-3:]}"
        return symbol

    async def place_order(self, req: OrderRequest, *, timeout_ms: int) -> OrderAck:
        self._require_credentials()
        product = self._product_id(req.symbol)
        order_config: Dict[str, Dict[str, str]]
        if req.order_type == "MARKET":
            order_config = {
                "market_market_ioc": {
                    "base_size": f"{req.qty:.8f}",
                }
            }
        else:
            price = req.price if req.price is not None else 0.0
            order_config = {
                "limit_limit_gtc": {
                    "base_size": f"{req.qty:.8f}",
                    "limit_price": f"{price:.2f}",
                    "post_only": False,
                }
            }
        body = {
            "client_order_id": req.idempotency_key or req.client_id,
            "product_id": product,
            "side": req.side.upper(),
            "order_configuration": order_config,
        }
        try:
            data = await self._request("POST", "/brokerage/orders", body=body, timeout_s=timeout_ms / 1000.0)
            success = data.get("success", False)
            order_id = data.get("order_id")
            message = "" if success else json.dumps(data)
            return OrderAck(client_id=req.client_id, broker_order_id=order_id, accepted=bool(success), message=message, ts_ns=int(time() * 1e9))
        except Exception as exc:
            return OrderAck(client_id=req.client_id, broker_order_id=None, accepted=False, message=str(exc), ts_ns=int(time() * 1e9))

    async def cancel(self, broker_order_id: str, *, timeout_ms: int) -> bool:
        self._require_credentials()
        body = {"order_ids": [broker_order_id]}
        try:
            data = await self._request("POST", "/brokerage/orders/batch_cancel", body=body, timeout_s=timeout_ms / 1000.0)
            results = data.get("results") or []
            for res in results:
                if res.get("success"):
                    return True
            return False
        except Exception:
            return False

    async def positions(self) -> Dict[str, Position]:
        # Spot não mantém posições alavancadas
        return {}

    async def balances(self) -> Dict[str, Balance]:
        self._require_credentials()
        try:
            data = await self._request("GET", "/brokerage/accounts", timeout_s=5.0)
        except Exception:
            return {}
        out: Dict[str, Balance] = {}
        for acct in data.get("accounts", []) or []:
            currency = acct.get("currency")
            avail = acct.get("available_balance", {}).get("value")
            hold = acct.get("hold", {}).get("value")
            total = acct.get("ready", {}).get("value")
            try:
                free = float(avail)
            except (TypeError, ValueError):
                free = 0.0
            try:
                used = float(hold)
            except (TypeError, ValueError):
                used = 0.0
            try:
                tot = float(total)
            except (TypeError, ValueError):
                tot = free + used
            if currency:
                out[currency] = Balance(ccy=currency, free=free, used=used, total=tot)
        return out


def json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))
