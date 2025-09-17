from __future__ import annotations
from typing import List


def vol_zscore(returns: List[float], window: int = 60) -> float:
    arr = returns[-window:]
    if len(arr) < 2:
        return 0.0
    import statistics
    mu = statistics.mean(arr)
    sigma = statistics.pstdev(arr) or 1.0
    return (arr[-1] - mu) / sigma

