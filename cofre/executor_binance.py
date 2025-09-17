from __future__ import annotations
from typing import Optional, Dict, Any
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
from time import time
from backend.common.secure_env import get_secret
from backend.cofre.ledger import CofreLedger
from backend.cofre.executor_base import CofreExecutorBase
from backend.common.config import load_yaml


class BinanceCofreExecutor(CofreExecutorBase):
    """Executa transferências internas (universal transfer) para o cofre da Binance.
    Precisa de BINANCE_API_KEY/SECRET com permissão SAPI, sem withdraw externo.
    """

    def __init__(self, base: str = "https://api.binance.com"):
        self._base = base
        self._api_key = get_secret("BINANCE_API_KEY")
        self._api_secret = get_secret("BINANCE_API_SECRET").encode()
        cfg = load_yaml("backend/config/cofre.yaml")
        super().__init__(min_sweep_usdt=float(cfg.get("min_sweep_usdt", 100)))
        self._route = ((cfg.get("venues") or {}).get("binance") or {}).get("route", "MAIN_FUNDING")
        self._asset = ((cfg.get("venues") or {}).get("binance") or {}).get("asset", "USDT")

    def _sign(self, params: Dict[str, str]) -> str:
        query = urlencode(params)
        return hmac.new(self._api_secret, query.encode(), hashlib.sha256).hexdigest()

    async def sweep(self, amount_usdt: float, *, reason: str = "auto", account: Optional[str] = None) -> bool:
        if not self.min_allowed(amount_usdt):
            return False
        params: Dict[str, str] = {
            "type": self._route,
            "asset": self._asset,
            "amount": str(round(float(amount_usdt), 2)),
            "timestamp": str(int(time() * 1000)),
            "recvWindow": "5000",
        }
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        async with aiohttp.ClientSession(headers=headers) as s:
            async with s.post(self._base + "/sapi/v1/asset/transfer", params=params, timeout=10) as r:
                data = await r.json()
                ok = (r.status == 200) and (data.get("success") in (True, "true", 1))
                await self.record({
                    "venue": "binance",
                    "account": account or "acc#1",
                    "amount_usdt": amount_usdt,
                    "route": self._route,
                    "asset": self._asset,
                    "api_status": r.status,
                    "api_resp": data,
                    "reason": reason,
                })
                return ok
