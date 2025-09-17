from __future__ import annotations
from typing import Dict, Optional, Tuple
from backend.storage.redis_client import RedisClient


class TechIndicators:
    def __init__(self, redis: Optional[RedisClient] = None):
        self._redis = redis or RedisClient()

    async def _get_mid(self, venue: str, symbol: str) -> tuple[float, float]:
        if not self._redis.enabled():
            return 0.0, 0.0
        cur = await self._redis.get_json(f"md:quote:{venue}:{symbol}") or {}
        prv = await self._redis.get_json(f"md:last:{venue}:{symbol}") or {}
        bid = float(cur.get("bid", 0) or 0)
        ask = float(cur.get("ask", 0) or 0)
        mid = (bid + ask) / 2.0 if bid and ask else 0.0
        p_bid = float(prv.get("bid", 0) or 0)
        p_ask = float(prv.get("ask", 0) or 0)
        p_mid = (p_bid + p_ask) / 2.0 if p_bid and p_ask else 0.0
        return mid, p_mid

    async def ema(self, venue: str, symbol: str, period: int) -> float:
        mid, _ = await self._get_mid(venue, symbol)
        if not self._redis.enabled() or not mid:
            return 0.0
        key = f"ind:ema:{venue}:{symbol}:{period}"
        prev = await self._redis.get_json(key) or {}
        prev_ema = float(prev.get("ema", 0.0))
        k = 2.0 / (period + 1.0)
        ema = mid if prev_ema == 0.0 else (mid - prev_ema) * k + prev_ema
        await self._redis.set_json(key, {"ema": ema}, ex=60)
        return float(ema)

    async def rsi(self, venue: str, symbol: str, period: int = 14) -> float:
        mid, p_mid = await self._get_mid(venue, symbol)
        if not self._redis.enabled() or not mid or not p_mid:
            return 50.0
        change = mid - p_mid
        gain = max(0.0, change)
        loss = max(0.0, -change)
        key = f"ind:rsi:{venue}:{symbol}:{period}"
        prev = await self._redis.get_json(key) or {}
        avg_gain = float(prev.get("avg_gain", gain))
        avg_loss = float(prev.get("avg_loss", loss))
        # Wilder's smoothing
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = (avg_gain / avg_loss) if avg_loss > 0 else 999.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        await self._redis.set_json(key, {"avg_gain": avg_gain, "avg_loss": avg_loss, "rsi": rsi}, ex=60)
        return float(rsi)

    async def atr_proxy(self, venue: str, symbol: str, period: int = 14) -> float:
        mid, p_mid = await self._get_mid(venue, symbol)
        if not self._redis.enabled() or not mid or not p_mid:
            return 0.0
        tr = abs(mid - p_mid)
        key = f"ind:atrp:{venue}:{symbol}:{period}"
        prev = await self._redis.get_json(key) or {}
        atr = float(prev.get("atr", tr))
        atr = (atr * (period - 1) + tr) / period
        await self._redis.set_json(key, {"atr": atr}, ex=60)
        return float(atr)

    async def macd(self, venue: str, symbol: str, fast: int = 12, slow: int = 26, signal_p: int = 9) -> Tuple[float, float, float]:
        efast = await self.ema(venue, symbol, fast)
        eslow = await self.ema(venue, symbol, slow)
        macd = efast - eslow
        key = f"ind:macd:{venue}:{symbol}:{fast}:{slow}:{signal_p}"
        prev = await self._redis.get_json(key) or {}
        sig_prev = float(prev.get("signal", 0.0))
        k = 2.0 / (signal_p + 1.0)
        signal = macd if sig_prev == 0.0 else (macd - sig_prev) * k + sig_prev
        hist = macd - signal
        await self._redis.set_json(key, {"signal": signal, "macd": macd, "hist": hist}, ex=60)
        return float(macd), float(signal), float(hist)

    async def bbands(self, venue: str, symbol: str, period: int = 20, mult: float = 2.0) -> Tuple[float, float, float]:
        mid, _ = await self._get_mid(venue, symbol)
        if not self._redis.enabled() or not mid:
            return 0.0, 0.0, 0.0
        key = f"ind:bb:{venue}:{symbol}:{period}:{mult}"
        prev = await self._redis.get_json(key) or {}
        mean = float(prev.get("mean", mid))
        var = float(prev.get("var", 0.0))
        # Welford EMA variance approx
        alpha = 2.0 / (period + 1.0)
        delta = mid - mean
        mean += alpha * delta
        var = (1 - alpha) * (var + alpha * delta * delta)
        std = max(0.0, var) ** 0.5
        upper = mean + mult * std
        lower = mean - mult * std
        await self._redis.set_json(key, {"mean": mean, "var": var, "upper": upper, "lower": lower}, ex=60)
        return float(lower), float(mean), float(upper)

    async def keltner(self, venue: str, symbol: str, period: int = 20, mult: float = 2.0) -> Tuple[float, float, float]:
        ema_c = await self.ema(venue, symbol, period)
        atr = await self.atr_proxy(venue, symbol, period)
        upper = ema_c + mult * atr
        lower = ema_c - mult * atr
        return float(lower), float(ema_c), float(upper)

    async def adx(self, venue: str, symbol: str, period: int = 14) -> Tuple[float, float, float]:
        mid, p_mid = await self._get_mid(venue, symbol)
        if not self._redis.enabled() or not mid or not p_mid:
            return 0.0, 0.0, 0.0
        up = max(0.0, mid - p_mid)
        dn = max(0.0, p_mid - mid)
        key = f"ind:adx:{venue}:{symbol}:{period}"
        prev = await self._redis.get_json(key) or {}
        tr = abs(mid - p_mid)
        atr = float(prev.get("atr", tr))
        atr = (atr * (period - 1) + tr) / period
        dm_pos = float(prev.get("dm_pos", up))
        dm_neg = float(prev.get("dm_neg", dn))
        dm_pos = (dm_pos * (period - 1) + up) / period
        dm_neg = (dm_neg * (period - 1) + dn) / period
        di_pos = (dm_pos / atr * 100.0) if atr > 0 else 0.0
        di_neg = (dm_neg / atr * 100.0) if atr > 0 else 0.0
        dx = (abs(di_pos - di_neg) / max(1e-9, (di_pos + di_neg))) * 100.0
        adx_prev = float(prev.get("adx", dx))
        adx = (adx_prev * (period - 1) + dx) / period
        await self._redis.set_json(key, {"atr": atr, "dm_pos": dm_pos, "dm_neg": dm_neg, "di_pos": di_pos, "di_neg": di_neg, "adx": adx}, ex=60)
        return float(di_pos), float(di_neg), float(adx)

    async def supertrend_proxy(self, venue: str, symbol: str, period: int = 10, mult: float = 3.0) -> float:
        # Simplified: trend = sign(mid - ema(period)) * 1 if |mid-ema| > mult*atr
        ema_c = await self.ema(venue, symbol, period)
        atr = await self.atr_proxy(venue, symbol, period)
        mid, _ = await self._get_mid(venue, symbol)
        if not mid or not ema_c:
            return 0.0
        thr = mult * atr
        if thr <= 0:
            return 0.0
        dev = mid - ema_c
        return 1.0 if dev > thr else (-1.0 if dev < -thr else 0.0)

    async def donchian_proxy(self, venue: str, symbol: str, window: int = 20) -> Tuple[float, float, float]:
        # Keep a small rolling window of mids in Redis; not perfect but useful
        mid, _ = await self._get_mid(venue, symbol)
        if not self._redis.enabled() or not mid:
            return 0.0, 0.0, 0.0
        key = f"ind:donch:{venue}:{symbol}:{window}"
        prev = await self._redis.get_json(key) or {"vals": []}
        vals = list(prev.get("vals", []))
        vals.append(float(mid))
        if len(vals) > window:
            vals = vals[-window:]
        hi = max(vals) if vals else 0.0
        lo = min(vals) if vals else 0.0
        midc = (hi + lo) / 2.0 if vals else 0.0
        await self._redis.set_json(key, {"vals": vals, "hi": hi, "lo": lo}, ex=120)
        return float(lo), float(midc), float(hi)

    async def obv_proxy(self, venue: str, symbol: str) -> float:
        # Proxy from microstructure: use buy_ratio and vol_qty ~ 10s horizon
        if not self._redis.enabled():
            return 0.0
        micro = await self._redis.get_json(f"md:micro:{venue}:{symbol}") or {}
        vol = float(micro.get("vol_qty", 0.0))
        br = float(micro.get("buy_ratio", 0.5))
        return float(vol * (2.0 * br - 1.0))

    async def snapshot(self, venue: str, symbol: str) -> Dict[str, float]:
        ema_fast = await self.ema(venue, symbol, period=12)
        ema_slow = await self.ema(venue, symbol, period=26)
        rsi = await self.rsi(venue, symbol, period=14)
        atrp = await self.atr_proxy(venue, symbol, period=14)
        _, _, macd_hist = await self.macd(venue, symbol)
        _, _, adx = await self.adx(venue, symbol)
        lo, mean, hi = await self.bbands(venue, symbol)
        return {
            "ema_fast": float(ema_fast),
            "ema_slow": float(ema_slow),
            "rsi": float(rsi),
            "atr_proxy": float(atrp),
            "macd_hist": float(macd_hist),
            "adx": float(adx),
            "bb_width": float(hi - lo),
        }
