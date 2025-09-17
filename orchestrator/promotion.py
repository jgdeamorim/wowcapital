from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
import json


class PromotionManager:
    def __init__(self, stages: list[str] | None = None, state_path: str = "backend/var/promotions.json",
                 gates: Dict[str, Any] | None = None):
        self.stages = stages or ["pending", "sandbox", "shadow", "active"]
        self.status: Dict[str, str] = {}
        self.history: list[Dict[str, Any]] = []
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self.gates = gates or {"min_sharpe": 0.8, "max_maxdd": 0.25, "min_trades": 30}
        # optional ramp schedule per plugin_id
        self.schedule: Dict[str, list[Dict[str, Any]]] = {}

    # persistence
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.status = data.get("status", {})
            self.history = data.get("history", [])
        except Exception:
            self.status = {}
            self.history = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"status": self.status, "history": self.history, "schedule": self.schedule}, separators=(",", ":")))
        tmp.replace(self.path)
        
    def schedule_ramp(self, plugin_id: str, steps: list[Dict[str, Any]]) -> None:
        """Define a ramp schedule: list of steps {stage, min_sharpe?, max_maxdd?, min_trades?}.
        Stages must be in self.stages order (e.g., sandbox -> shadow -> active)."""
        # basic validation
        seen = set()
        for s in steps:
            st = s.get("stage")
            if st not in self.stages:
                raise ValueError(f"unknown stage in ramp: {st}")
            if st in seen:
                raise ValueError("duplicate stage in ramp")
            seen.add(st)
        self.schedule[plugin_id] = steps
        self.history.append({"plugin_id": plugin_id, "event": "schedule", "steps": steps})
        self._save()

    def set_stage(self, plugin_id: str, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError(f"Invalid stage: {stage}")
        self.status[plugin_id] = stage
        self.history.append({"plugin_id": plugin_id, "event": "set_stage", "stage": stage})
        self._save()

    def get_stage(self, plugin_id: str) -> str:
        return self.status.get(plugin_id, self.stages[0])

    def all(self) -> Dict[str, str]:
        return dict(self.status)

    def check_gates(self, metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        min_sharpe = float(self.gates.get("min_sharpe", 0.8))
        max_maxdd = float(self.gates.get("max_maxdd", 0.25))
        min_trades = int(self.gates.get("min_trades", 30))
        s = float(metrics.get("sharpe", 0.0))
        dd = float(metrics.get("maxDD", 1e9))
        tr = int(metrics.get("trades", 0))
        checks = {
            "sharpe_ok": s >= min_sharpe,
            "maxdd_ok": dd <= max_maxdd,
            "trades_ok": tr >= min_trades,
        }
        ok = all(checks.values())
        checks.update({"min_sharpe": min_sharpe, "max_maxdd": max_maxdd, "min_trades": min_trades})
        return ok, checks

    def propose(self, plugin_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        ok, details = self.check_gates(metrics)
        self.history.append({"plugin_id": plugin_id, "event": "propose", "ok": ok, "metrics": metrics, "details": details})
        self._save()
        return {"ok": ok, "details": details}

    def advance(self, plugin_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Advance to next stage if metrics satisfy either global gates or scheduled gates for next stage."""
        cur = self.get_stage(plugin_id)
        # find next stage in order
        try:
            idx = self.stages.index(cur)
        except ValueError:
            idx = 0
        if idx + 1 >= len(self.stages):
            return {"ok": True, "stage": cur, "detail": "already at final stage"}
        next_stage = self.stages[idx + 1]
        # which gates to use
        step_gate = None
        for s in self.schedule.get(plugin_id, []):
            if s.get("stage") == next_stage:
                step_gate = s
                break
        gates = dict(self.gates)
        if step_gate:
            for k in ("min_sharpe", "max_maxdd", "min_trades"):
                if k in step_gate:
                    gates[k] = step_gate[k]
        # check
        saved = self.gates
        self.gates = gates
        ok, details = self.check_gates(metrics)
        self.gates = saved
        if not ok:
            self.history.append({"plugin_id": plugin_id, "event": "advance_block", "from": cur, "to": next_stage, "metrics": metrics, "details": details})
            self._save()
            return {"ok": False, "stage": cur, "next": next_stage, "details": details}
        self.set_stage(plugin_id, next_stage)
        self.history.append({"plugin_id": plugin_id, "event": "advanced", "to": next_stage, "metrics": metrics, "details": details})
        self._save()
        return {"ok": True, "stage": next_stage, "details": details}
