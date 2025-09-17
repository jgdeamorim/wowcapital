from __future__ import annotations
import asyncio
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int | None = None):
        self.rate = max(rate_per_sec, 0.1)
        self.capacity = burst if burst is not None else int(max(rate_per_sec * 2, 2))
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # need to wait
            wait_time = (1.0 - self.tokens) / self.rate
        await asyncio.sleep(wait_time)
        # recurse once to consume after sleep
        await self.acquire()

