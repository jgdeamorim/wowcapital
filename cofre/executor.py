from __future__ import annotations
from typing import Optional
from backend.audit.worm import AsyncWormLog


class SweepExecutor:
    """Stub for executing USDT transfers to a secure cofre wallet.
    For MVP, only logs the intent to WORM; integration with wallet API can be added later.
    """

    def __init__(self, worm: Optional[AsyncWormLog] = None):
        self._worm = worm or AsyncWormLog("backend/var/audit/trading.ndjson")

    async def transfer(self, account: str, amount_usdt: float, dest: str = "cofre#primary") -> bool:
        await self._worm.append({
            "ts_ns": 0,
            "event": "cofre_sweep_intent",
            "payload": {"account": account, "amount_usdt": amount_usdt, "dest": dest},
        })
        return True
