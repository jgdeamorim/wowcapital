from __future__ import annotations
from typing import Dict


class Selector:
    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {"tier_A": 1.0, "tier_B": 0.8, "polygon": 0.5, "twelve_data": 0.4, "alpha_vantage": 0.3}

    def score(self, source: str, p95_ms: float, error_rate: float) -> float:
        base = self.weights.get(source, 0.1)
        lat_factor = max(0.0, min(1.0, (50.0 / max(p95_ms, 1.0))))
        err_factor = max(0.0, 1.0 - error_rate)
        return base * lat_factor * err_factor

