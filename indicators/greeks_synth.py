from __future__ import annotations
from typing import Dict, Any, Optional
from backend.storage.redis_client import RedisClient


def _norm(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    x = (v - lo) / (hi - lo)
    return max(0.0, min(1.0, x))


class SyntheticGreeks:
    def __init__(self, redis: Optional[RedisClient] = None):
        self._redis = redis or RedisClient()

    async def compute(self, venue: str, symbol: str, *, funding_rate_bps: float | None = None) -> Dict[str, float]:
        """Compute synthetic proxies for Δ, Γ, Vanna, Charm blending quote + micro + L2 depth.
        Bounded in [-1,1]. Cached 1s in Redis.
        """
        delta_hat = 0.0
        gamma_hat = 0.0
        vanna_hat = 0.0
        charm_hat = 0.0
        cache_key = f"ind:greeks:{venue}:{symbol}"
        if self._redis.enabled():
            try:
                cached = await self._redis.get_json(cache_key)
                if cached:
                    return cached
            except Exception:
                pass
        else:
            return {"delta_hat": 0.0, "gamma_hat": 0.0, "vanna_hat": 0.0, "charm_hat": 0.0}
        try:
            cur = await self._redis.get_json(f"md:quote:{venue}:{symbol}") or {}
            prv = await self._redis.get_json(f"md:last:{venue}:{symbol}") or {}
            micro = await self._redis.get_json(f"md:micro:{venue}:{symbol}") or {}
            depth = await self._redis.get_json(f"md:depth:{venue}:{symbol}") or {}
            bid = float(cur.get("bid", 0) or 0)
            ask = float(cur.get("ask", 0) or 0)
            mid = (bid + ask) / 2.0 if bid and ask else 0.0
            spread = max(0.0, ask - bid)
            spread_bps = (spread / mid) * 10_000.0 if mid else 0.0
            p_bid = float(prv.get("bid", 0) or 0)
            p_ask = float(prv.get("ask", 0) or 0)
            p_mid = (p_bid + p_ask) / 2.0 if p_bid and p_ask else 0.0
            p_spread = max(0.0, p_ask - p_bid)
            p_spread_bps = (p_spread / p_mid) * 10_000.0 if p_mid else spread_bps
            # delta_hat: blend mid momentum and buy_ratio momentum
            if mid and p_mid:
                dm = (mid - p_mid) / p_mid * 10_000.0
                d1 = max(-1.0, min(1.0, dm / 5.0))
            else:
                d1 = 0.0
            try:
                br = float(micro.get("buy_ratio", 0.5))
                br_prev = float((await self._redis.get_json(f"md:last:micro:{venue}:{symbol}") or {}).get("buy_ratio", br))
                d2 = max(-1.0, min(1.0, (br - br_prev) * 3.0))
            except Exception:
                d2 = 0.0
            delta_hat = max(-1.0, min(1.0, 0.7 * d1 + 0.3 * d2))

            # gamma_hat: adverse convexity (spread widening) + L2 imbalance flip against delta
            d_spread = spread_bps - p_spread_bps
            g1 = max(-1.0, min(1.0, -d_spread / 10.0))
            try:
                imb_l5 = float(depth.get("imbalance_l5", 0.0))
                g2 = -abs(imb_l5) if (imb_l5 * delta_hat) < 0 else 0.0
            except Exception:
                g2 = 0.0
            gamma_hat = max(-1.0, min(1.0, 0.8 * g1 + 0.2 * g2))

            # vanna_hat: combine |funding| and spread level
            fr = abs(float(funding_rate_bps or 0.0))
            vanna_hat = max(0.0, min(1.0, _norm(fr, 5.0, 50.0) * 0.6 + _norm(spread_bps, 5.0, 50.0) * 0.4))

            # charm_hat: decay when spread expands and OFI opposes delta
            try:
                ofi = float(micro.get("ofi_rolling", 0.0))
                oppose = (ofi * delta_hat) < 0
            except Exception:
                oppose = False
            charm_hat = -min(1.0, max(0.0, (max(0.0, d_spread) / 10.0) * (0.5 + (0.5 if oppose else 0.0))))
        except Exception:
            pass
        out = {
            "delta_hat": float(delta_hat),
            "gamma_hat": float(gamma_hat),
            "vanna_hat": float(vanna_hat),
            "charm_hat": float(charm_hat),
        }
        if self._redis.enabled():
            try:
                await self._redis.set_json(cache_key, out, ex=1)
            except Exception:
                pass
        return out

