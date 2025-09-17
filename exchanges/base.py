from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
from backend.core.contracts import OrderRequest, OrderAck, Position, Balance


class ExchangeAdapter(ABC):
    name: str

    @abstractmethod
    async def healthcheck(self) -> bool: ...

    @abstractmethod
    async def place_order(self, req: OrderRequest, *, timeout_ms: int) -> OrderAck: ...

    @abstractmethod
    async def cancel(self, broker_order_id: str, *, timeout_ms: int) -> bool: ...

    @abstractmethod
    async def positions(self) -> Dict[str, Position]: ...

    @abstractmethod
    async def balances(self) -> Dict[str, Balance]: ...
