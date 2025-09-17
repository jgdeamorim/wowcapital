from __future__ import annotations
from typing import Any, Dict
from time import time
from backend.audit.worm import AsyncWormLog
from backend.storage.mongo import MongoStore


class CofreLedger:
    def __init__(self, worm: AsyncWormLog | None = None, mongo: MongoStore | None = None):
        self._worm = worm or AsyncWormLog("backend/var/audit/trading.ndjson")
        self._mongo = mongo or MongoStore()

    async def record(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("ts_ns", int(time() * 1e9))
        entry.setdefault("event", "cofre_sweep_exec")
        await self._worm.append(entry)
        if self._mongo.enabled():
            await self._mongo.db.cofre_ledger.insert_one(entry)  # type: ignore
