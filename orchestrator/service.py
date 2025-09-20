from __future__ import annotations
from typing import Dict, Any
from backend.orchestrator.plugin_manager import PluginManager
from backend.indicators.tech import TechIndicators
from backend.indicators.greeks_synth import SyntheticGreeks
from backend.indicators.micro import MicroIndicators
from backend.storage.redis_client import RedisClient
from backend.plugins.registry import PluginRegistry
from backend.core.contracts import MarketSnapshot


class OrchestratorService:
    def __init__(self, pm: PluginManager):
        self.pm = pm
        import os
        base_dir = os.getenv("PLUGINS_DISCOVER_DIR", "backend/plugins/active")
        self.reg = PluginRegistry(base_dir)
        self.reg.discover()
        self.instances: Dict[str, Any] = {}
        self._redis = RedisClient()
        self._tech = TechIndicators(self._redis)
        self._greeks = SyntheticGreeks(self._redis)
        self._micro = MicroIndicators(self._redis)

    def list_plugins(self) -> Dict[str, Any]:
        return {k: v.__name__ for k, v in self.reg.plugins.items()}

    def ensure_instance(self, plugin_id: str) -> Any:
        if plugin_id in self.instances:
            return self.instances[plugin_id]
        # load from registry using plugin_id == manifest id
        cls = self.reg.plugins.get(plugin_id)
        if not cls:
            cls = self._load_entrypoint_from_manifest(plugin_id)
        if not cls:
            raise KeyError(f"Plugin not found: {plugin_id}")
        self.instances[plugin_id] = cls()
        return self.instances[plugin_id]

    def _manifest(self, plugin_id: str):
        return self.pm.loaded.get(plugin_id)

    def _load_entrypoint_from_manifest(self, plugin_id: str):
        manifest = self._manifest(plugin_id)
        if not manifest:
            return None
        runtime = manifest.runtime if isinstance(manifest.runtime, dict) else {}
        entry = None
        if runtime:
            entry = runtime.get("entrypoint") or runtime.get("entry_point")
        if not entry and isinstance(manifest.metadata, dict):
            entry = manifest.metadata.get("entrypoint")
        if not entry:
            return None
        module, _, cls_name = entry.partition(":")
        if not module or not cls_name:
            return None
        try:
            if module.endswith(".py"):
                from importlib.util import spec_from_file_location, module_from_spec
                from pathlib import Path
                spec = spec_from_file_location("manifest_mod", str(Path(module)))
                if not spec or not spec.loader:
                    return None
                mod = module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
            else:
                from importlib import import_module
                mod = import_module(module)
            obj = getattr(mod, cls_name, None)
            if obj:
                self.reg.plugins[plugin_id] = obj
            return obj
        except Exception:
            return None

    async def _enrich(self, plugin_id: str, snap: Dict[str, Any]) -> Dict[str, Any]:
        # Copy snapshot
        s = dict(snap)
        features = dict(s.get("features") or {})
        venue = s.get("venue") or features.get("venue") or "binance"
        symbol = s.get("symbol") or features.get("symbol") or "BTCUSDT"
        # Compute requested indicators from manifest
        m = self._manifest(plugin_id)
        inds = []
        if m and isinstance(m.runtime, dict):
            inds = m.runtime.get("indicators") or []
        tech = await self._tech.snapshot(venue, symbol)
        for k in ("ema_fast","ema_slow","rsi","atr_proxy"):
            if (not inds) or (k in inds):
                features[k] = tech.get(k)
        # Greeks proxies
        g = await self._greeks.compute(venue, symbol)
        for k, v in g.items():
            if (not inds) or (k in inds):
                features[k] = v
        # Microstructure (vwap, buy_ratio, imbalance_l1, convexity_proxy)
        micro = await self._micro.snapshot(venue, symbol)
        for k in ("vwap","buy_ratio","imbalance_l1","convexity_proxy"):
            if (not inds) or (k in inds):
                if k in micro:
                    features[k] = micro.get(k)
        s["features"] = features
        return s

    def decide(self, plugin_id: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # Backwards-compatible: orchestrator APIs are sync; run async enrichment via loop
        import asyncio
        try:
            snap = asyncio.get_event_loop().run_until_complete(self._enrich(plugin_id, snapshot))
        except Exception:
            snap = snapshot
        inst = self.ensure_instance(plugin_id)
        ms = MarketSnapshot(**snap)
        return inst.decide(ms)
