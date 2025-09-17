from __future__ import annotations
from typing import Dict, Any
from time import time
from backend.plugins.base import IStrategy
from backend.core.contracts import MarketSnapshot


class JericoBinanceV1(IStrategy):
    @staticmethod
    def required_features() -> list[str]:
        return []

    @staticmethod
    def warmup_bars() -> int:
        return 0

    def decide(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        # MVP: simple buy/sell flip based on mid move threshold
        # In production, use indicators/guards injected by orchestrator
        side = "BUY" if (snapshot.features.get("signal", 1.0) >= 0) else "SELL"
        qty = snapshot.features.get("qty", 0.001)
        return {
            "symbol": snapshot.symbol,
            "side": side,
            "qty": qty,
            "order_type": "MARKET",
            "client_id": "jerico.binance.v1",
            "idempotency_key": f"jerico-{snapshot.symbol}-{int(time())}",
            "meta": {"class": snapshot.class_, "account": snapshot.features.get("account", "acc#1")},
        }

