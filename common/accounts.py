from __future__ import annotations
from typing import Optional, Dict, Any
from pathlib import Path
import os
import yaml


class Accounts:
    def __init__(self, path: str = "backend/config/accounts.yaml"):
        self.path = Path(path)
        self.cfg: Dict[str, Any] = {}
        if self.path.exists():
            self.cfg = yaml.safe_load(self.path.read_text()) or {}

    def get_keys(self, venue: str, account: str) -> Dict[str, str]:
        exch = (self.cfg.get("exchanges") or {}).get(venue, {})
        accs = exch.get("accounts") or {}
        acc = accs.get(account) or {}
        # Resolve env placeholders like ${VAR:-}
        def _resolve(val: Optional[str]) -> str:
            if not isinstance(val, str):
                return ""
            if val.startswith("${") and val.endswith("}"):
                # ${VAR:-}
                inner = val[2:-1]
                name = inner.split(":", 1)[0].strip("-")
                return os.getenv(name, "")
            return val
        out: Dict[str, str] = {}
        for k in ("api_key", "api_secret", "mode", "category"):
            if k in acc:
                out[k] = _resolve(acc[k])
        return out

