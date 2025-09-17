from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import importlib
import importlib.util
import yaml
from backend.orchestrator.manifest_schema import StrategyManifest


class PluginRegistry:
    def __init__(self, base_dir: str = "plugins/active"):
        self.base = Path(base_dir)
        self.plugins: Dict[str, Any] = {}

    def discover(self) -> None:
        if not self.base.exists():
            return
        for mf in self.base.rglob("*.yaml"):
            data = yaml.safe_load(mf.read_text())
            # Try new StrategyManifest format first
            entry = None
            pid = data.get("metadata", {}).get("id") or data.get("name") or mf.stem
            try:
                sm = StrategyManifest.parse_obj(data)
                entry = sm.entrypoint
            except Exception:
                entry = (data.get("runtime", {}) or {}).get("entrypoint")
            if not entry:
                continue
            mod, cls = entry.split(":")
            # Support module import or file path import
            if mod.endswith(".py") or "/" in mod or mod.startswith("backend/"):
                p = (mf.parent / mod) if not Path(mod).is_absolute() else Path(mod)
                spec = importlib.util.spec_from_file_location("plugin_mod", str(p))
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                obj = getattr(module, cls, None)
            else:
                obj = getattr(importlib.import_module(mod), cls, None)
            if obj is None:
                continue
            self.plugins[pid] = obj
