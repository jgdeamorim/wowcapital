from __future__ import annotations
from typing import Any, Optional
import os
import json

try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class RedisClient:
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        self._client: Optional["redis.Redis"] = None if redis is None else redis.from_url(self.url, decode_responses=True)

    def enabled(self) -> bool:
        return self._client is not None

    @property
    def client(self):  # type: ignore
        return self._client

    async def get_json(self, key: str) -> Optional[dict[str, Any]]:
        if not self.enabled():
            return None
        v = await self._client.get(key)  # type: ignore
        if not v:
            return None
        try:
            return json.loads(v)
        except Exception:
            return None

    async def set_json(self, key: str, value: dict[str, Any], ex: int = 1) -> None:
        if not self.enabled():
            return
        await self._client.set(key, json.dumps(value, separators=(",", ":")), ex=ex)  # type: ignore

