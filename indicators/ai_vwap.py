from __future__ import annotations
from typing import List


def ai_vwap(prices: List[float], window: int = 300, decay: float = 0.7) -> float:
    if not prices:
        return 0.0
    arr = prices[-window:]
    if not arr:
        return 0.0
    w = [decay ** (len(arr) - 1 - i) for i in range(len(arr))]
    den = sum(w) or 1.0
    return sum(a * b for a, b in zip(arr, w)) / den

