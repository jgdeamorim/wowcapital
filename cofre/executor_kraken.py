from __future__ import annotations
from typing import Optional
from backend.cofre.executor_base import CofreExecutorBase
from backend.common.config import load_yaml


class KrakenCofreExecutor(CofreExecutorBase):
    """Placeholder executor for Kraken.
    Kraken API possui restrições para transferências internas; por padrão criamos uma requisição pendente
    para aprovação (two-man rule) via endpoints /cofre/pending e /cofre/approve.
    """

    def __init__(self):
        cfg = load_yaml("backend/config/cofre.yaml")
        super().__init__(min_sweep_usdt=float(cfg.get("min_sweep_usdt", 100)))

    async def sweep(self, amount_usdt: float, *, reason: str = "auto", account: Optional[str] = None) -> bool:
        # Indisponível por API: gerar pendência e retornar False
        return False

