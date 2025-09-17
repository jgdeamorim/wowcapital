from __future__ import annotations
from typing import List
from pathlib import Path
from backend.orchestrator.plugin_manager import PluginManager


def load_manifests_from_dir(pm: PluginManager, directory: str) -> List[str]:
    p = Path(directory)
    loaded: List[str] = []
    if not p.exists():
        return loaded
    for path in p.rglob("*.yaml"):
        try:
            m = pm.load_manifest(str(path))
            loaded.append(m.metadata.get("id", path.name))
        except Exception:
            continue
    return loaded

