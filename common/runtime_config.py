from __future__ import annotations
from typing import Optional, Dict, Any
import json
from backend.storage.redis_client import RedisClient


class RuntimeConfig:
    """Lightweight runtime overrides store (Redis-backed).
    Namespaces: risk, cofre. Scopes: global or per plugin_id.
    Keys in Redis:
      rc:global:{ns} -> JSON dict
      rc:plugin:{plugin_id}:{ns} -> JSON dict
    """

    def __init__(self, redis: Optional[RedisClient] = None):
        self._redis = redis or RedisClient()
        self._mem: Dict[str, Dict[str, Dict[str, Any]]] = {"global": {}, "plugin": {}}

    def enabled(self) -> bool:
        return self._redis.enabled()

    async def _get_raw(self, ns: str, plugin_id: Optional[str]) -> Dict[str, Any]:
        if self.enabled():
            try:
                if plugin_id:
                    key = f"rc:plugin:{plugin_id}:{ns}"
                else:
                    key = f"rc:global:{ns}"
                v = await self._redis.client.get(key)  # type: ignore
                return json.loads(v) if v else {}
            except Exception:
                return {}
        # memory fallback
        scope = "plugin" if plugin_id else "global"
        sid = plugin_id or "__global__"
        return self._mem.get(scope, {}).get(f"{sid}:{ns}", {})

    async def _set_raw(self, ns: str, data: Dict[str, Any], plugin_id: Optional[str]) -> None:
        if self.enabled():
            try:
                if plugin_id:
                    key = f"rc:plugin:{plugin_id}:{ns}"
                else:
                    key = f"rc:global:{ns}"
                await self._redis.client.set(key, json.dumps(data, separators=(",", ":")))  # type: ignore
                return
            except Exception:
                pass
        scope = "plugin" if plugin_id else "global"
        sid = plugin_id or "__global__"
        self._mem.setdefault(scope, {})[f"{sid}:{ns}"] = dict(data)

    async def get_overrides(self, ns: str, plugin_id: Optional[str] = None) -> Dict[str, Any]:
        """Return merged overrides: global overlaid by plugin-specific."""
        g = await self._get_raw(ns, None)
        p = await self._get_raw(ns, plugin_id) if plugin_id else {}
        out = dict(g)
        out.update(p)
        return out

    async def set_overrides(self, ns: str, data: Dict[str, Any], plugin_id: Optional[str] = None) -> None:
        await self._set_raw(ns, data, plugin_id)

