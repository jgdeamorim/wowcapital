from __future__ import annotations
from typing import Dict, Any, Tuple
from backend.common.config import load_yaml
from datetime import datetime, timezone


class RiskEngine:
    def __init__(self, risk_path: str = "env/policies/risk.yaml"):
        try:
            self.cfg = load_yaml(risk_path)
        except Exception:
            self.cfg = {}

    def _symbol_allowed(self, symbol: str, meta: Dict[str, Any]) -> Tuple[bool, str]:
        wl = (self.cfg.get("symbol_whitelist") or [])
        bl = (self.cfg.get("symbol_blacklist") or [])
        if wl and symbol not in wl:
            return False, "SYMBOL_NOT_WHITELISTED"
        if bl and symbol in bl:
            return False, "SYMBOL_BLACKLISTED"
        return True, ""

    def _qty_limits(self, symbol: str, qty: float) -> Tuple[bool, str]:
        sym_cfg = {}
        for it in (self.cfg.get("instruments") or []):
            if it.get("symbol") == symbol:
                sym_cfg = it
                break
        min_qty = float(sym_cfg.get("min_qty", self.cfg.get("min_qty", 0)))
        max_qty = float(sym_cfg.get("max_qty", self.cfg.get("max_qty", 0)))
        if min_qty and qty < min_qty:
            return False, "MIN_QTY"
        if max_qty and qty > max_qty:
            return False, "MAX_QTY"
        return True, ""

    def evaluate(self, req: Dict[str, Any], *, ref_price: float | None = None, spread_bps: float | None = None,
                 balances: Dict[str, Any] | None = None, positions: Dict[str, Any] | None = None,
                 funding_rate_bps: float | None = None, is_perp: bool = False,
                 overrides: Dict[str, Any] | None = None) -> Tuple[bool, str]:
        """Pre-trade checks: whitelist/blacklist, qty limits, notional, slippage caps (basic).
        Returns (allowed, reason)
        """
        symbol = req.get("symbol")
        qty = float(req.get("qty", 0))
        meta = req.get("meta") or {}
        ok, reason = self._symbol_allowed(symbol, meta)
        if not ok:
            return False, reason
        ok, reason = self._qty_limits(symbol, qty)
        if not ok:
            return False, reason
        # Notional checks
        o = overrides or {}
        min_notional = float(o.get("min_notional_usd", self.cfg.get("min_notional_usd", 0.0)))
        max_notional = float(o.get("max_notional_usd", self.cfg.get("max_notional_usd", 0.0)))
        for it in (self.cfg.get("instruments") or []):
            if it.get("symbol") == symbol:
                min_notional = float(it.get("min_notional_usd", min_notional))
                max_notional = float(it.get("max_notional_usd", max_notional))
                break
        # Fail-closed when price is required but unavailable
        if (min_notional or max_notional) and qty and (not ref_price or ref_price <= 0):
            return False, "NO_REF_PRICE"
        if ref_price and qty:
            notional = ref_price * qty
            if min_notional and notional < min_notional:
                return False, "MIN_NOTIONAL"
            if max_notional and notional > max_notional:
                return False, "MAX_NOTIONAL"
        # Slippage cap (basic): if provided spread exceeds cap
        cap_bps = float(o.get("slippage_cap_bps", self.cfg.get("slippage_cap_bps", 0.0)))
        for it in (self.cfg.get("instruments") or []):
            if it.get("symbol") == symbol:
                cap_bps = float(it.get("slippage_cap_bps", cap_bps))
                break
        if cap_bps and spread_bps and spread_bps > cap_bps:
            return False, "SLIPPAGE_CAP"
        # Price guard (LIMIT): if price far from mid
        price_guard_bps = float(o.get("price_guard_bps", self.cfg.get("price_guard_bps", 0.0)))
        for it in (self.cfg.get("instruments") or []):
            if it.get("symbol") == symbol:
                price_guard_bps = float(it.get("price_guard_bps", price_guard_bps))
                break
        if req.get("order_type") == "LIMIT" and ref_price and price_guard_bps and req.get("price") is not None:
            px = float(req.get("price"))
            dev_bps = abs(px - ref_price) / ref_price * 10_000
            if dev_bps > price_guard_bps:
                return False, "PRICE_GUARD"
        # Macro no-trade windows
        if self._macro_block(symbol):
            return False, "MACRO_WINDOW"
        # Funding guard + leverage caps (perps)
        if is_perp:
            # funding rate cap
            fcap = float(o.get("funding_cap_bps", self.cfg.get("funding_cap_bps", 0.0)))
            for it in (self.cfg.get("instruments") or []):
                if it.get("symbol") == symbol:
                    fcap = float(it.get("funding_cap_bps", fcap))
                    break
            if fcap and funding_rate_bps and abs(funding_rate_bps) > fcap:
                return False, "FUNDING_CAP"
            # leverage cap (approx): notional <= free_usdt * max_leverage
            max_lev = float(o.get("max_leverage", self.cfg.get("max_leverage", 0.0)))
            for it in (self.cfg.get("instruments") or []):
                if it.get("symbol") == symbol:
                    max_lev = float(it.get("max_leverage", max_lev))
                    break
            if max_lev and ref_price and qty:
                notional = ref_price * qty
                free_usdt = 0.0
                if balances:
                    for ccy, b in balances.items():
                        try:
                            # b pode ser Pydantic Balance ou dict
                            free_usdt += float(getattr(b, "free", 0.0) if ccy.upper() in ("USDT","USD") else 0.0)
                        except Exception:
                            pass
                if free_usdt and notional > free_usdt * max_lev:
                    return False, "LEVERAGE_CAP"
        return True, ""

    def _macro_block(self, symbol: str) -> bool:
        try:
            macro = load_yaml("env/policies/macro.yaml")
        except Exception:
            return False
        rules = macro.get("no_trade") or []
        now = datetime.now(timezone.utc)
        day = now.strftime("%a")  # Mon, Tue, ...
        hm = now.strftime("%H:%M")
        for r in rules:
            syms = r.get("symbols") or ["*"]
            days = r.get("days") or [day]
            start = r.get("start") or "00:00"
            end = r.get("end") or "00:00"
            if day not in days:
                continue
            if ("*" in syms) or (symbol in syms):
                if start <= hm <= end:
                    return True
        return False
