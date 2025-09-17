from __future__ import annotations
import yaml
from typing import Any, Dict
from backend.orchestrator.manifest_schema import ManifestModel


class Manifest:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.metadata = data.get("metadata", {})
        self.runtime = data.get("runtime", {})
        self.cofre_policy = data.get("cofre_policy", {})
        self.risk_controls = data.get("risk_controls", {})
        self.guards = data.get("guards", {})

    def tier(self) -> str:
        return self.metadata.get("tier") or self.data.get("tier", "1.6")


class PluginManager:
    def __init__(self):
        self.loaded: Dict[str, Manifest] = {}

    def load_manifest(self, path: str) -> Manifest:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        # strict validation via Pydantic
        mm = ManifestModel(**data)
        m = Manifest(mm.dict())
        self._validate(m)
        self.loaded[m.metadata.get("id", path)] = m
        return m

    def _validate(self, m: Manifest) -> None:
        assert m.metadata.get("name"), "metadata.name required"
        assert m.metadata.get("version"), "metadata.version required"
        cp = m.cofre_policy
        if cp:
            assert 0.5 <= float(cp.get("sweep_pct", 0.8)) <= 0.95
            assert 100 <= int(cp.get("account_float_min_usd", 300)) <= int(cp.get("account_float_max_usd", 500))
