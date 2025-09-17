from __future__ import annotations
from typing import Dict, List, Tuple
import math
from backend.storage.redis_client import RedisClient


class BanditAllocator:
    """Simple UCB-style allocator over arms (account,pair,strategy).
    Stores per-arm stats in Redis as JSON: {n, mean, var}
    """

    def __init__(self, rc: RedisClient | None = None):
        self._rc = rc or RedisClient()

    async def record(self, arm_id: str, reward: float) -> None:
        if not self._rc.enabled():
            return
        key = f"alloc:arm:{arm_id}"
        cur = await self._rc.get_json(key) or {"n": 0, "mean": 0.0, "m2": 0.0}
        n = int(cur.get("n", 0)) + 1
        mean = float(cur.get("mean", 0.0))
        delta = reward - mean
        mean += delta / n
        m2 = float(cur.get("m2", 0.0)) + delta * (reward - mean)
        await self._rc.set_json(key, {"n": n, "mean": mean, "m2": m2}, ex=24 * 3600)

    async def suggest(self, arms: List[str], budget: float = 1.0, min_alloc: float = 0.0) -> Dict[str, float]:
        if not arms:
            return {}
        stats: Dict[str, Tuple[int, float, float]] = {}
        for a in arms:
            cur = await self._rc.get_json(f"alloc:arm:{a}") if self._rc.enabled() else None
            n = int((cur or {}).get("n", 0))
            mean = float((cur or {}).get("mean", 0.0))
            var = 0.0
            stats[a] = (n, mean, var)
        # UCB score
        total_n = sum(max(1, s[0]) for s in stats.values())
        scores: Dict[str, float] = {}
        for a, (n, mean, _) in stats.items():
            ucb = mean + math.sqrt(2 * math.log(max(2, total_n)) / max(1, n))
            scores[a] = ucb
        # normalize to budget
        ssum = sum(max(0.0, v) for v in scores.values()) or 1.0
        out = {a: max(min_alloc, budget * (max(0.0, scores[a]) / ssum)) for a in arms}
        # ensure sum <= budget
        factor = budget / max(1e-9, sum(out.values()))
        out = {k: v * factor for k, v in out.items()}
        return out

