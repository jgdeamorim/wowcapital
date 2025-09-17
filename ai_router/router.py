from __future__ import annotations
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import asyncio

from backend.core.contracts import OrderRequest
from backend.execution.gateway import OrderRouter


class ExecuteTradeArgs(BaseModel):
    account_id: str
    exchange: str
    symbol: str
    side: str
    size: float
    order_type: str
    price: Optional[float] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    idempotency_token: str

    def to_order_request(self) -> OrderRequest:
        side = self.side.upper()
        if side not in ("BUY", "SELL"):
            raise ValidationError([{"loc": ("side",), "msg": "invalid side", "type": "value_error"}], ExecuteTradeArgs)
        ot = self.order_type.upper()
        if ot not in ("MARKET", "LIMIT"):
            raise ValidationError([{"loc": ("order_type",), "msg": "invalid order_type", "type": "value_error"}], ExecuteTradeArgs)
        meta = {"account": self.account_id}
        return OrderRequest(symbol=self.symbol, side=side, qty=float(self.size), order_type=ot, price=self.price,
                            client_id="orchestrator", idempotency_key=self.idempotency_token, meta=meta)


async def execute_trade(args: ExecuteTradeArgs) -> Dict[str, Any]:
    """Validate and execute a trade via OrderRouter (risk enforced inside)."""
    req = args.to_order_request()
    router = OrderRouter()  # ephemeral router; for long-running, wire shared instance
    router.register_intent(req.idempotency_key, req.client_id, account=args.account_id)
    ack = await router.place(args.exchange.lower(), req)
    return ack.model_dump()


class ModelRouter:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.default = cfg.get("default", {}).get("router")
        self.models = cfg.get("models", {})

    def select(self, task: str = "default") -> Dict[str, Any]:
        return self.models.get(self.default, {})
