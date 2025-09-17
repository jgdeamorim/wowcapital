from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml


class PairsConfig:
    def __init__(self, path: str = "backend/config/pairs.yaml"):
        self.path = Path(path)
        self.cfg: Dict[str, List[str]] = {}
        if self.path.exists():
            data = yaml.safe_load(self.path.read_text()) or {}
            pairs = data.get("pairs", {})
            for sym, conf in pairs.items():
                self.cfg[sym] = conf.get("venues", [])

    def allowed_venues(self, symbol: str) -> List[str]:
        return self.cfg.get(symbol, ["binance"])  # default: binance

