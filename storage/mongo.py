from __future__ import annotations
import os
from typing import Optional, Dict, Any
from datetime import datetime, timezone

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except Exception:  # pragma: no cover
    AsyncIOMotorClient = None  # type: ignore


class MongoStore:
    def __init__(self, uri: Optional[str] = None, db_name: str = "wowcapital"):
        self.uri = uri or os.getenv("MONGO_URI")
        self.db_name = db_name
        self.client = None
        self.db = None
        if self.uri and AsyncIOMotorClient:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.db_name]

    def enabled(self) -> bool:
        return self.db is not None

    async def record_order(self, order: Dict[str, Any]) -> None:
        if not self.enabled():
            return
        await self.db.orders.insert_one(order)

    async def record_ack(self, ack: Dict[str, Any]) -> None:
        if not self.enabled():
            return
        await self.db.acks.insert_one(ack)

    async def record_fill(self, fill: Dict[str, Any]) -> None:
        if not self.enabled():
            return
        await self.db.fills.insert_one(fill)

    async def record_micro(self, micro: Dict[str, Any]) -> None:
        """Persist a microstructure snapshot with TTL-friendly timestamp.
        Expects micro to include venue, symbol, ts_ms (optional). Adds ts_dt (UTC now).
        """
        if not self.enabled():
            return
        doc = dict(micro)
        doc.setdefault("ts_ms", int(datetime.now(timezone.utc).timestamp() * 1000))
        doc["ts_dt"] = datetime.now(timezone.utc)
        await self.db.micro.insert_one(doc)
