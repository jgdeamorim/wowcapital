from __future__ import annotations
import asyncio
import time
from typing import Optional
from backend.storage.redis_client import RedisClient


class RedisRateLimiter:
    """Distributed rate limiter with Lua token-bucket and weights.
    Falls back to a simple fixed-window if Lua/EVAL not available.
    """

    LUA_TB = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local weight = tonumber(ARGV[4])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then tokens = capacity end
if ts == nil then ts = now_ms end
local elapsed = now_ms - ts
if elapsed < 0 then elapsed = 0 end
local refill = (elapsed / 1000.0) * rate
tokens = tokens + refill
if tokens > capacity then tokens = capacity end
local allowed = 0
if tokens >= weight then
  tokens = tokens - weight
  allowed = 1
end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, 2000)
return allowed
"""

    def __init__(self, redis: Optional[RedisClient] = None):
        self._redis = redis or RedisClient()
        self._sha: Optional[str] = None

    async def _ensure_script(self):
        if not self._redis.enabled():
            return
        if self._sha is None:
            try:
                self._sha = await self._redis.client.script_load(self.LUA_TB)  # type: ignore
            except Exception:
                self._sha = None

    async def acquire(self, key: str, *, rate_per_sec: float, burst: int = 0, weight: int = 1, timeout_s: float = 1.0) -> None:
        if not self._redis.enabled():
            return
        await self._ensure_script()
        start = time.monotonic()
        capacity = int(max(burst, 0) + max(rate_per_sec, 0))
        if capacity <= 0:
            capacity = 1
        while True:
            allowed = 1
            if self._sha:
                try:
                    allowed = await self._redis.client.evalsha(self._sha, 1, f"rltb:{key}", int(time.time()*1000), float(rate_per_sec), int(capacity), int(weight))  # type: ignore
                except Exception:
                    allowed = 1  # fail-open minimally
            else:
                # fallback fixed-window counter
                window = int(time.time())
                rkey = f"rl:{key}:{window}"
                limit = int(rate_per_sec + burst)
                pipe = self._redis.client.pipeline()  # type: ignore
                pipe.incrby(rkey, weight)
                pipe.expire(rkey, 2)
                cnt, _ = await pipe.execute()
                allowed = 1 if int(cnt) <= limit else 0

            if int(allowed) == 1:
                return
            await asyncio.sleep(0.02)
            if time.monotonic() - start > timeout_s:
                return
