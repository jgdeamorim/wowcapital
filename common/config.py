from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}


def load_policies() -> Dict[str, Any]:
    cfg = {
        "execution": load_yaml("env/policies/execution.yaml"),
        "exchange_tos": load_yaml("env/policies/exchange_tos.yaml"),
    }
    return cfg


def load_instruments() -> Dict[str, Any]:
    return load_yaml("env/instruments.yaml")


def load_models() -> Dict[str, Any]:
    # models.yaml lives under backend/config by convention
    return load_yaml("backend/config/models.yaml")
