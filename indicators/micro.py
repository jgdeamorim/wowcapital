from __future__ import annotations
from typing import Optional, Dict, Any
from backend.storage.redis_client import RedisClient


class MicroIndicators:
    def __init__(self, redis: Optional[RedisClient] = None):
        self._redis = redis or RedisClient()

    async def snapshot(self, venue: str, symbol: str) -> Dict[str, Any]:
        if not self._redis.enabled():
            return {}
        v = await self._redis.get_json(f"md:micro:{venue}:{symbol}")
        return v or {}

