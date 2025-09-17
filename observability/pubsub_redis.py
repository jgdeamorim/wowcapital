from __future__ import annotations
from typing import Any, Optional
import json
from backend.storage.redis_client import RedisClient


class RedisPubSub:
    def __init__(self, client: Optional[RedisClient] = None):
        self._rc = client or RedisClient()

    async def publish(self, channel: str, payload: dict[str, Any]) -> None:
        if not self._rc.enabled():
            return
        await self._rc.client.publish(channel, json.dumps(payload, separators=(",", ":")))  # type: ignore

