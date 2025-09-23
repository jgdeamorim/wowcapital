from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from common.runtime_config import RuntimeConfig

_AVAILABILITY_PATH = Path("config/exchange_availability.yaml")
_CACHE: Optional[Dict[str, Any]] = None


def _load_raw(path: Path = _AVAILABILITY_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_capabilities(country: str) -> Dict[str, Any]:
    """Return capability details for a given country (case-insensitive)."""

    global _CACHE
    if _CACHE is None:
        _CACHE = _load_raw()
    payload = _CACHE.get("paises", {}) if isinstance(_CACHE, dict) else {}
    return payload.get(country.lower(), {}) if isinstance(payload, dict) else {}


def list_countries() -> list[str]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_raw()
    payload = _CACHE.get("paises", {}) if isinstance(_CACHE, dict) else {}
    return sorted(payload.keys()) if isinstance(payload, dict) else []


async def sync_runtime_policy(runtime: RuntimeConfig, country: str) -> None:
    """Load country capabilities and store consolidated view in runtime config.

    The data is stored under the namespace ``exchange_policy`` in Redis/memory so
    other components can adjust trading modes (shadow vs live, feature toggles).
    """

    profile = get_capabilities(country)
    if not profile:
        await runtime.set_overrides(
            "exchange_policy",
            {"country": country.lower(), "exchanges": {}, "services": {}},
        )
        return

    services = profile.get("services", {}) if isinstance(profile.get("services"), dict) else {}
    latency = profile.get("latency_ms", {}) if isinstance(profile.get("latency_ms"), dict) else {}
    markets = profile.get("markets", {}) if isinstance(profile.get("markets"), dict) else {}
    regulation = profile.get("regulation", {}) if isinstance(profile.get("regulation"), dict) else {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for exchange, svc in services.items():
        ex_key = exchange.lower()
        reg = regulation.get(ex_key, {}) if isinstance(regulation, dict) else {}
        normalized[ex_key] = {
            "services": svc,
            "latency_ms": latency.get(ex_key),
            "markets": markets.get(ex_key),
            "regulation": reg,
        }

    await runtime.set_overrides(
        "exchange_policy",
        {
            "country": country.lower(),
            "exchanges": normalized,
            "services": services,
        },
    )

