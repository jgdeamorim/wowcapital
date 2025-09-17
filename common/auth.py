from __future__ import annotations
from typing import Dict, Optional, Callable, List
import os
import json
from fastapi import Header, HTTPException


def _keys_map() -> Dict[str, List[str]]:
    """Return API keys mapping to scopes from env.
    WOW_API_KEYS may be a JSON object: {"<key>": ["trade","config"], ...}
    If absent, falls back to single API_KEY that grants all scopes.
    """
    raw = os.getenv("WOW_API_KEYS")
    if raw:
        try:
            m = json.loads(raw)
            out: Dict[str, List[str]] = {}
            for k, v in m.items():
                if isinstance(v, list):
                    out[k] = [str(x) for x in v]
                elif isinstance(v, str):
                    out[k] = [x.strip() for x in v.split(",") if x.strip()]
            return out
        except Exception:
            pass
    k = os.getenv("API_KEY")
    if k:
        return {k: ["*"]}
    return {}


def require_scopes(required: List[str]):
    """FastAPI dependency factory to require API key with given scopes.
    Header: X-API-Key: <key>
    If WOW_API_KEYS not set and API_KEY not set, raises 401 for any protected endpoint.
    """

    async def _dep(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
        keys = _keys_map()
        if not keys:
            # Allow explicit bypass for local/dev if configured
            if os.getenv("WOW_AUTH_BYPASS", "0") in ("1", "true", "TRUE"):
                return
            raise HTTPException(status_code=401, detail="auth not configured")
        if not x_api_key or x_api_key not in keys:
            raise HTTPException(status_code=401, detail="invalid api key")
        scopes = keys.get(x_api_key, [])
        if "*" in scopes:
            return
        need = set(required)
        have = set(scopes)
        if not need.issubset(have):
            raise HTTPException(status_code=403, detail="insufficient scope")

    return _dep
