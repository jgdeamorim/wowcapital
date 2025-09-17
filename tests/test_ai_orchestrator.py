import asyncio
import pytest

from backend.ai_router.router import ExecuteTradeArgs


def test_execute_trade_args_validation_basic():
    with pytest.raises(Exception):
        ExecuteTradeArgs(
            account_id="acc#1", exchange="binance", symbol="BTCUSDT",
            side="HOLD", size=0.01, order_type="MARKET", idempotency_token="x"
        )
    with pytest.raises(Exception):
        ExecuteTradeArgs(
            account_id="acc#1", exchange="binance", symbol="BTCUSDT",
            side="BUY", size=0.01, order_type="X", idempotency_token="x"
        )
    ok = ExecuteTradeArgs(
        account_id="acc#1", exchange="binance", symbol="BTCUSDT",
        side="BUY", size=0.01, order_type="MARKET", idempotency_token="x"
    )
    orq = ok.to_order_request()
    assert orq.symbol == "BTCUSDT" and orq.side == "BUY"

