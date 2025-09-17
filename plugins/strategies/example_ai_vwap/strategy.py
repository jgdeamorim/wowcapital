from __future__ import annotations
from typing import Dict, Any, List
from backend.plugins.base import IStrategy
from backend.core.contracts import MarketSnapshot


class Strategy(IStrategy):
    """Template AI-VWAP momentum strategy (simplified example).
    Safe defaults for shadow testing.
    """

    def __init__(self, manifest: Dict[str, Any] | None = None, params: Dict[str, Any] | None = None):
        self.manifest = manifest or {}
        p = params or {}
        self.window = int(p.get("window", 300))
        self.decay = float(p.get("decay", 0.7))
        self._series: Dict[str, List[float]] = {}

    @staticmethod
    def required_features() -> list[str]:
        return []

    @staticmethod
    def warmup_bars() -> int:
        return 300

    def _ai_vwap(self, arr: List[float]) -> float:
        if not arr:
            return 0.0
        w = [self.decay ** (len(arr) - 1 - i) for i in range(len(arr))]
        den = sum(w) or 1.0
        return sum(a * b for a, b in zip(arr, w)) / den

    def decide(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        sym = snapshot.symbol
        series = self._series.setdefault(sym, [])
        # use mid as proxy price
        px = float(snapshot.mid or 0.0) or float(snapshot.bid or 0.0) or float(snapshot.ask or 0.0)
        if px <= 0.0:
            return {}
        series.append(px)
        if len(series) < self.window:
            return {}
        window = series[-self.window:]
        vwap = self._ai_vwap(window)
        # simple momentum rule
        thr = 0.004
        if px > vwap * (1 + thr):
            side = "BUY"
        elif px < vwap * (1 - thr):
            side = "SELL"
        else:
            return {}
        qty = 0.001
        return {
            "symbol": sym,
            "side": side,
            "qty": qty,
            "order_type": "MARKET",
            "client_id": "example_ai_vwap",
            "idempotency_key": f"ai_vwap-{sym}-{len(series)}",
            "meta": {"class": snapshot.class_, "account": snapshot.features.get("account", "acc#1")},
        }

