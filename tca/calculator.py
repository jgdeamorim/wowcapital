from __future__ import annotations
from typing import Dict


def _extract_spot_fill(event: Dict) -> tuple[float, float, float]:
    # Returns (price, qty, commission)
    # executionReport fields: L=lastExecPrice, l=lastExecQty, n=commission
    px = float(event.get("L", 0) or event.get("p", 0) or 0)
    qty = float(event.get("l", 0) or event.get("q", 0) or 0)
    commission = float(event.get("n", 0) or 0)
    return px, qty, commission


def _extract_futures_fill(event: Dict) -> tuple[float, float]:
    # ORDER_TRADE_UPDATE: o: {p=price, q=qty}
    o = event.get("o", {})
    px = float(o.get("ap", 0) or o.get("p", 0) or 0)  # ap=avgPrice
    qty = float(o.get("q", 0) or 0)
    return px, qty


def estimate_costs_bps(event: Dict, fees_cfg: Dict, market: str = "spot") -> Dict:
    """Estimate costs in bps from a Binance user-data event.
    Uses fee rates from policies when commission not present.
    """
    if market == "spot":
        px, qty, commission = _extract_spot_fill(event)
        notional = px * qty if px and qty else 0.0
        fee_rate = fees_cfg.get("spot", {}).get("taker", 0.001)
        fee_usd = commission if commission else notional * fee_rate
    else:
        px, qty = _extract_futures_fill(event)
        notional = px * qty if px and qty else 0.0
        fee_rate = fees_cfg.get("futures", {}).get("taker", 0.0004)
        fee_usd = notional * fee_rate
    fee_bps = (fee_usd / notional) * 10_000 if notional else 0.0
    # Slippage and spread estimation require expected price and book; set to 0 in MVP
    return {
        "notional_usd": notional,
        "fee_usd": fee_usd,
        "fee_bps": fee_bps,
        "slippage_bps": 0.0,
        "spread_bps": 0.0,
        "total_bps": fee_bps,
    }

