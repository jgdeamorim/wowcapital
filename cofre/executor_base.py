from __future__ import annotations
from typing import Optional, Dict, Any
from backend.cofre.ledger import CofreLedger


class CofreExecutorBase:
    def __init__(self, min_sweep_usdt: float = 100.0):
        self._ledger = CofreLedger()
        self._min = float(min_sweep_usdt)

    def min_allowed(self, amt: float) -> bool:
        return float(amt) >= self._min

    async def record(self, entry: Dict[str, Any]) -> None:
        await self._ledger.record(entry)

