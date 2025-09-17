from __future__ import annotations
import asyncio
from typing import Dict, Any
from pathlib import Path
import json


class CofreManager:
    """Manages cofre (USDT) balance, float thresholds and sweep events per account.
    Simple file-backed state for MVP.
    """

    def __init__(self, state_path: str = "backend/var/cofre/state.json"):
        self._lock = asyncio.Lock()
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return {}
        return {}

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.state)

    async def update_float(self, account: str, value_usd: float) -> None:
        async with self._lock:
            s = self.state.setdefault(account, {"float_usd": 0.0, "cofre_usdt": 0.0})
            s["float_usd"] = float(value_usd)
            self._write_atomic(self.state)

    async def sweep_if_needed(self, account: str, policy: Dict[str, Any], balance_usdt: float, overrides: Dict[str, Any] | None = None) -> float:
        """Apply sweep when float exceeds max threshold.
        Returns sweep amount performed.
        """
        async with self._lock:
            s = self.state.setdefault(account, {"float_usd": 0.0, "cofre_usdt": 0.0})
            ov = overrides or {}
            float_min = float(ov.get("account_float_min_usd", policy.get("account_float_min_usd", 300)))
            float_max = float(ov.get("account_float_max_usd", policy.get("account_float_max_usd", 500)))
            float_cap = ov.get("float_cap_usd")
            sweep_pct = float(ov.get("sweep_pct", policy.get("sweep_pct", 0.85)))
            current = balance_usdt
            s["float_usd"] = current
            sweep_amt = 0.0
            if float_cap is not None:
                try:
                    cap = float(float_cap)
                except Exception:
                    cap = float_min
                if current > cap:
                    sweep_amt = max(0.0, current - cap) * sweep_pct
            elif current > float_max:
                # sweep portion above min to cofre, keeping float_min..float_max band
                excess = max(0.0, current - float_min)
                sweep_amt = excess * sweep_pct
                # split into safety pool and main cofre
                safety_pct = float(ov.get("safety_pool_pct", policy.get("safety_pool_pct", 0.10)))
                safety_amt = sweep_amt * safety_pct
                main_amt = sweep_amt - safety_amt
                s["cofre_usdt"] = float(s.get("cofre_usdt", 0.0)) + main_amt
                s["safety_usdt"] = float(s.get("safety_usdt", 0.0)) + safety_amt
                s["float_usd"] = current - sweep_amt
            self._write_atomic(self.state)
            return sweep_amt

    def safety_snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"total": 0.0, "accounts": {}}
        for acc, st in self.state.items():
            amt = float(st.get("safety_usdt", 0.0))
            out["accounts"][acc] = amt
            out["total"] += amt
        return out

    async def replenish_from_safety(self, account: str, amount_usdt: float) -> bool:
        async with self._lock:
            s = self.state.setdefault(account, {"float_usd": 0.0, "cofre_usdt": 0.0, "safety_usdt": 0.0})
            if float(s.get("safety_usdt", 0.0)) < amount_usdt:
                return False
            s["safety_usdt"] = float(s.get("safety_usdt", 0.0)) - float(amount_usdt)
            s["float_usd"] = float(s.get("float_usd", 0.0)) + float(amount_usdt)
            self._write_atomic(self.state)
            return True

    def _write_atomic(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)
