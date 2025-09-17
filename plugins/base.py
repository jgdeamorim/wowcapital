from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from backend.core.contracts import MarketSnapshot


class IStrategy(ABC):
    @staticmethod
    def required_features() -> list[str]:
        return []

    @staticmethod
    def warmup_bars() -> int:
        return 0

    @abstractmethod
    def decide(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Returns a dict compatible with OrderRequest fields or a Signal.
        Deterministic, no I/O in hot path.
        """
        ...


class IIndicator(ABC):
    @abstractmethod
    def update(self, snapshot: MarketSnapshot) -> None: ...

    @abstractmethod
    def compute_batch(self, ohlcv: list[Dict[str, float]]) -> Dict[str, float]: ...

    @staticmethod
    def warmup_bars() -> int:
        return 0

