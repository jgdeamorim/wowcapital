from __future__ import annotations
from typing import Dict, Optional, Tuple
from backend.storage.redis_client import RedisClient


class PluginRouting:
    """Routing of plugins per symbol with dynamic weights.
    Redis keys:
      orch:plugin_for:{symbol} -> plugin_id (pinned assignment)
      orch:plugin_weight:{plugin_id}:{symbol} -> float weight (higher preferred)
    """

    def __init__(self, redis: Optional[RedisClient] = None):
        self._redis = redis or RedisClient()

    async def assign(self, symbol: str, plugin_id: str) -> None:
        if not self._redis.enabled():
            return
        await self._redis.client.set(f"orch:plugin_for:{symbol}", plugin_id)  # type: ignore

    async def unassign(self, symbol: str) -> None:
        if not self._redis.enabled():
            return
        await self._redis.client.delete(f"orch:plugin_for:{symbol}")  # type: ignore

    async def get_assignment(self, symbol: str) -> Optional[str]:
        if not self._redis.enabled():
            return None
        v = await self._redis.client.get(f"orch:plugin_for:{symbol}")  # type: ignore
        return v

    async def set_weight(self, plugin_id: str, symbol: str, weight: float) -> None:
        if not self._redis.enabled():
            return
        await self._redis.client.set(f"orch:plugin_weight:{plugin_id}:{symbol}", str(float(weight)))  # type: ignore

    async def get_weight(self, plugin_id: str, symbol: str) -> float:
        if not self._redis.enabled():
            return 0.0
        v = await self._redis.client.get(f"orch:plugin_weight:{plugin_id}:{symbol}")  # type: ignore
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    async def pick_best(self, symbol: str, candidates: list[str], default: str) -> str:
        # explicit assignment wins
        pinned = await self.get_assignment(symbol)
        if pinned and pinned in candidates:
            return pinned
        # choose highest weight
        best = default
        best_w = -1.0
        for p in candidates:
            w = await self.get_weight(p, symbol)
            if w > best_w:
                best = p
                best_w = w
        return best or default

