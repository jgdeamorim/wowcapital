from __future__ import annotations
from typing import Dict, Any, List, Tuple
import importlib
import importlib.util
import math
from backend.core.contracts import MarketSnapshot
from backend.common.config import load_yaml, load_policies
import os


def _load_strategy(entrypoint: str):
    """Load strategy class from entrypoint 'module:Class' or 'path.py:Class'."""
    mod, cls = entrypoint.split(":", 1)
    if mod.endswith(".py") or "/" in mod:
        spec = importlib.util.spec_from_file_location("plugin_mod", mod)
        if not spec or not spec.loader:
            raise ImportError(f"cannot import module from {mod}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
    else:
        module = importlib.import_module(mod)
    klass = getattr(module, cls)
    return klass


def _get_rules(symbol: str, venue: str) -> Dict[str, Any]:
    ins = load_yaml(os.getenv("INSTRUMENTS_FILE", "backend/config/instruments.yaml"))
    rules: Dict[str, Any] = {}
    for it in ins.get("symbols", []):
        if it.get("symbol") == symbol:
            pr = (it.get("precision") or {})
            st = (it.get("steps") or {})
            rules.update({
                "price_dp": pr.get("price_dp"),
                "qty_dp": pr.get("qty_dp"),
                "price_step": st.get("price"),
                "qty_step": st.get("qty"),
                "min_qty": it.get("min_qty"),
                "min_notional_usd": it.get("min_notional_usd"),
            })
            vr = (it.get("venues_rules") or {}).get(venue, {})
            for k in ("price_step","qty_step","min_qty","min_notional_usd","price_dp","qty_dp"):
                if vr.get(k) is not None:
                    rules[k] = vr.get(k)
            break
    return rules


def _quantize(value: float, step: float | None, dp: int | None) -> float:
    if step and step > 0:
        return round(round(value / step) * step, 12)
    if dp is not None:
        return round(value, int(dp))
    return value


def backtest_strategy(entrypoint: str, *, symbol: str, candles: List[Dict[str, float]], params: Dict[str, Any] | None = None,
                      fee_bps: float = 0.0, venue: str = "binance", spreads_bps: List[float] | None = None,
                      micro_timeline: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Very simple market-replay backtester for strategies exposing decide(MarketSnapshot).
    - Executes market orders at close with naive slippage model (bps by spread proxy)
    - Tracks PnL on a 1x notional basis, without leverage accounting
    candles: [{"ts": int_ms, "o":..., "h":..., "l":..., "c":...}]
    Returns metrics: sharpe, maxDD, winrate, pnl, trades
    """
    Strategy = _load_strategy(entrypoint)
    strat = Strategy({}, params or {})  # type: ignore
    cash = 0.0
    pos = 0.0
    entry_px = 0.0
    pnl_series: List[float] = []
    rets: List[float] = []
    wins = 0
    trades = 0
    last_equity = 0.0
    # Load fees from policies when not provided
    if fee_bps == 0.0:
        try:
            tos = load_policies().get("exchange_tos", {})
            fee_bps = float(((tos.get("exchanges") or {}).get(venue, {}).get("fees") or {}).get("taker_bps", 5.0))
        except Exception:
            fee_bps = 5.0
    rules = _get_rules(symbol, venue)
    for k in range(len(candles)):
        c = candles[k]
        ts_ns = int((c.get("ts") or 0) * 1e6)
        mid = float(c.get("c") or 0.0)
        if mid <= 0:
            continue
        snap = MarketSnapshot(class_="CRYPTO", symbol=symbol, ts_ns=ts_ns, bid=mid, ask=mid, mid=mid, spread=0.0, features={})
        try:
            d = strat.decide(snap) or {}
        except Exception:
            d = {}
        if d.get("side") in ("BUY", "SELL") and float(d.get("qty", 0)) > 0:
            side = d["side"]
            qty = float(d.get("qty", 0))
            price = mid
            # enforce min_notional/steps
            qty = _quantize(qty, rules.get("qty_step"), rules.get("qty_dp"))
            min_not = rules.get("min_notional_usd")
            if min_not and price > 0:
                need = float(min_not) / price
                if qty < need:
                    qty = need
                    qty = _quantize(qty, rules.get("qty_step"), rules.get("qty_dp"))
            # slippage proxy: use provided timeline if available
            if micro_timeline and k < len(micro_timeline):
                try:
                    mt = micro_timeline[k] or {}
                    base_spread = max(0.0, float(mt.get("spread_bps", 0.0)))
                    br = float(mt.get("buy_ratio", 0.5))
                    imb = float(mt.get("imbalance_l5", 0.0)) if mt.get("imbalance_l5") is not None else 0.0
                    # alignment: BUY with buy_ratio>0.5 hurts (more adverse), SELL with buy_ratio>0.5 helps
                    align = 1.0 if (side == "BUY" and br >= 0.5) or (side == "SELL" and br < 0.5) else -1.0
                    alpha = 0.6  # weight for buy_ratio deviation
                    beta = 0.4   # weight for depth imbalance influence
                    dev = abs(br - 0.5)
                    factor = 1.0 + align * alpha * dev + beta * abs(imb)
                    # clamp factor to reasonable bounds
                    factor = max(0.5, min(1.5, factor))
                    slip_bps = base_spread * factor
                except Exception:
                    slip_bps = 2.0
            elif spreads_bps and k < len(spreads_bps):
                slip_bps = max(0.0, float(spreads_bps[k]))
            else:
                slip_bps = 2.0
            if side == "BUY":
                fill_px = price * (1 + (slip_bps + fee_bps) / 10_000.0)
                cash -= fill_px * qty
                pos += qty
                if pos > 0 and entry_px == 0:
                    entry_px = fill_px
            else:
                fill_px = price * (1 - (slip_bps + fee_bps) / 10_000.0)
                cash += fill_px * qty
                pos -= qty
                if pos <= 0 and entry_px > 0:
                    # close long; PnL realized in cash
                    entry_px = 0.0
            trades += 1
        # mark-to-market equity
        equity = cash + pos * mid
        pnl_series.append(equity)
        if len(pnl_series) >= 2:
            rets.append(pnl_series[-1] - pnl_series[-2])
        if len(pnl_series) >= 2 and pnl_series[-1] > pnl_series[-2]:
            wins += 1
        last_equity = equity
    # Metrics
    import statistics
    winrate = (wins / trades) if trades else 0.0
    pnl = last_equity
    if len(rets) >= 2:
        mu = statistics.mean(rets)
        sigma = statistics.pstdev(rets) or 1.0
        sharpe = (mu / sigma) * (252 ** 0.5)
    else:
        sharpe = 0.0
    # Max drawdown
    peak = -1e18
    maxdd = 0.0
    for v in pnl_series:
        peak = max(peak, v)
        dd = (peak - v)
        maxdd = max(maxdd, dd)
    return {"sharpe": sharpe, "maxDD": maxdd, "winrate": winrate, "pnl": pnl, "trades": trades}
