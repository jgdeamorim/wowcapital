from __future__ import annotations
from typing import Optional, Dict, Any
import time
from backend.storage.redis_client import RedisClient
from backend.storage.mongo import MongoStore


class PerfAggregator:
    """Aggregates decision/fill performance for learning and knobs.
    - Stores decisions (score/features) briefly in Redis keyed by idempotency key
    - Writes joined samples to Mongo (if enabled): slippage vs mid, score/features
    - Maintains lightweight counters in Redis per plugin for quick stats
    """

    def __init__(self, redis: Optional[RedisClient] = None, mongo: Optional[MongoStore] = None):
        self._redis = redis or RedisClient()
        self._mongo = mongo or MongoStore()

    async def record_decision(self, plugin_id: str, idem: str, *, symbol: str, venue: str,
                              score: Optional[float] = None, features: Optional[Dict[str, Any]] = None) -> None:
        # Cache decision metadata for short time to join on fill
        if self._redis.enabled() and idem:
            try:
                key = f"perf:decision:{idem}"
                await self._redis.set_json(key, {
                    "plugin_id": plugin_id,
                    "symbol": symbol,
                    "venue": venue,
                    "score": score,
                    "features": features or {},
                    "ts_ns": int(time.time() * 1e9),
                }, ex=600)
                # Update quick stats per plugin
                p = f"perf:plugin:{plugin_id}"
                pipe = self._redis.client.pipeline()  # type: ignore
                pipe.hincrby(p, "decisions", 1)
                if score is not None:
                    pipe.hincrbyfloat(p, "score_sum", float(score))
                await pipe.execute()
            except Exception:
                pass

    async def record_fill(self, *, venue: str, symbol: str, side: str, price: float, qty: float,
                           idem: Optional[str]) -> None:
        plugin_id = None
        score = None
        features: Dict[str, Any] = {}
        if self._redis.enabled() and idem:
            try:
                meta = await self._redis.get_json(f"perf:decision:{idem}")
                if meta:
                    plugin_id = meta.get("plugin_id")
                    score = meta.get("score")
                    features = meta.get("features") or {}
            except Exception:
                pass
        # Compute slippage vs cached mid
        mid = 0.0
        signed_slip_bps = 0.0
        abs_slip_bps = 0.0
        try:
            if self._redis.enabled():
                q = await self._redis.get_json(f"md:quote:{venue}:{symbol}")
                if q:
                    bid = float(q.get("bid", 0))
                    ask = float(q.get("ask", 0))
                    mid = (bid + ask) / 2.0 if bid and ask else 0.0
        except Exception:
            pass
        if mid:
            signed_slip_bps = (price - mid) / mid * 10_000.0
            if side.upper() == "SELL":
                signed_slip_bps = -signed_slip_bps
            abs_slip_bps = abs(signed_slip_bps)
        # Update Redis counters
        try:
            if self._redis.enabled():
                # Plugin-level counters
                if plugin_id:
                    p = f"perf:plugin:{plugin_id}"
                    pipe = self._redis.client.pipeline()  # type: ignore
                    pipe.hincrby(p, "fills", 1)
                    pipe.hincrbyfloat(p, "slip_sum_bps", float(signed_slip_bps))
                    pipe.hsetnx(p, "min_slip_bps", float(abs_slip_bps))
                    pipe.hsetnx(p, "max_slip_bps", float(abs_slip_bps))
                    # track min/max by compare (best-effort)
                    pipe.hget(p, "min_slip_bps")
                    pipe.hget(p, "max_slip_bps")
                    res = await pipe.execute()
                    try:
                        cur_min = float(res[-2] or 0)
                        cur_max = float(res[-1] or 0)
                        if abs_slip_bps and (abs_slip_bps < cur_min or cur_min == 0):
                            await self._redis.client.hset(p, mapping={"min_slip_bps": abs_slip_bps})  # type: ignore
                        if abs_slip_bps and abs_slip_bps > cur_max:
                            await self._redis.client.hset(p, mapping={"max_slip_bps": abs_slip_bps})  # type: ignore
                    except Exception:
                        pass
        except Exception:
            pass
        # Persist sample to Mongo
        try:
            if self._mongo.enabled():
                doc = {
                    "plugin_id": plugin_id,
                    "symbol": symbol,
                    "venue": venue,
                    "side": side,
                    "price": price,
                    "qty": qty,
                    "mid": mid,
                    "slippage_bps": signed_slip_bps,
                    "abs_slippage_bps": abs_slip_bps,
                    "score": score,
                    "features": features,
                    "idempotency_key": idem,
                    "ts_ns": int(time.time() * 1e9),
                }
                await self._mongo.db.perf_samples.insert_one(doc)  # type: ignore
        except Exception:
            pass

