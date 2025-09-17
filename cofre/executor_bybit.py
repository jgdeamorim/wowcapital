from __future__ import annotations
from typing import Optional, Dict, Any
import hmac
import hashlib
from urllib.parse import urlencode
import aiohttp
from time import time
from backend.common.secure_env import get_secret
from backend.cofre.ledger import CofreLedger
from backend.common.config import load_yaml
from backend.cofre.executor_base import CofreExecutorBase


class BybitCofreExecutor(CofreExecutorBase):
    """Executa transferências internas (v5 asset transfer) na Bybit para o cofre.
    Requer BYBIT_API_KEY/BYBIT_API_SECRET com permissão de transfer.
    """

    def __init__(self, base: str = "https://api.bybit.com"):
        self._base = base
        self._api_key = get_secret("BYBIT_API_KEY")
        self._api_secret = get_secret("BYBIT_API_SECRET")
        cfg = load_yaml("backend/config/cofre.yaml")
        super().__init__(min_sweep_usdt=float(cfg.get("min_sweep_usdt", 100)))
        bybit_cfg = (cfg.get("venues") or {}).get("bybit") or {}
        self._coin = bybit_cfg.get("asset", "USDT")
        # Defaults: SPOT -> UNIFIED
        self._from = bybit_cfg.get("from_account", "SPOT").upper()
        self._to = bybit_cfg.get("to_account", "UNIFIED").upper()

    def _sign(self, ts_ms: str, recv_window: str, query_or_body: str) -> str:
        to_sign = ts_ms + self._api_key + recv_window + query_or_body
        return hmac.new(self._api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()

    async def sweep(self, amount_usdt: float, *, reason: str = "auto", account: Optional[str] = None) -> bool:
        if not self.min_allowed(amount_usdt):
            return False
        path = "/v5/asset/transfer"
        recv_window = "5000"
        ts_ms = str(int(time() * 1000))
        body = {
            "transferId": f"cofre-{ts_ms}",
            "coin": self._coin,
            "amount": str(round(float(amount_usdt), 2)),
            "fromAccountType": self._from,
            "toAccountType": self._to,
        }
        import json
        body_str = json.dumps(body, separators=(",", ":"))
        sign = self._sign(ts_ms, recv_window, body_str)
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts_ms,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(self._base + path, data=body_str, headers=headers, timeout=10) as r:
                try:
                    data = await r.json()
                except Exception:
                    data = {"raw": await r.text()}
                ok = (data.get("retCode") == 0)
                await self.record({
                    "venue": "bybit",
                    "account": account or "acc#1",
                    "amount_usdt": amount_usdt,
                    "from": self._from,
                    "to": self._to,
                    "coin": self._coin,
                    "api_status": r.status,
                    "api_resp": data,
                    "reason": reason,
                })
                return ok
