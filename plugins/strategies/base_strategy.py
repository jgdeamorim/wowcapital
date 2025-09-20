from __future__ import annotations
from typing import Any, Dict, Optional


class BaseStrategy:
    """Base genérica para estratégias. Pode ser expandida futuramente."""

    def on_start(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Hook de inicialização opcional."""
        return None

    def on_stop(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Hook de finalização opcional."""
        return None

    def reset(self) -> None:
        """Permite reconfigurar estado interno."""
        return None
