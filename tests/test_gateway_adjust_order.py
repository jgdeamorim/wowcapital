import os
import asyncio
import pytest

from backend.execution.gateway import OrderRouter
from backend.core.contracts import OrderRequest


@pytest.mark.asyncio
async def test_adjust_order_min_notional_and_steps(monkeypatch, tmp_path):
    # instruments config with steps and min_notional
    ins = {
        'symbols': [
            {
                'symbol': 'TESTUSDT',
                'precision': {'price_dp': 2, 'qty_dp': 3},
                'steps': {'price': 0.01, 'qty': 0.001},
                'min_notional_usd': 10,
                'venues_rules': {'binance': {'price_step': 0.01, 'qty_step': 0.001, 'min_notional_usd': 10}}
            }
        ]
    }
    f = tmp_path / "instruments.yaml"
    f.write_text(__import__('yaml').safe_dump(ins))
    monkeypatch.setenv('INSTRUMENTS_FILE', str(f))

    # set dummy secrets to construct router without error for binance/bybit
    monkeypatch.setenv('BINANCE_API_KEY','DUMMY')
    monkeypatch.setenv('BINANCE_API_SECRET','DUMMY')
    monkeypatch.setenv('BYBIT_API_KEY','DUMMY')
    monkeypatch.setenv('BYBIT_API_SECRET','DUMMY')
    r = OrderRouter()
    # qty too small at mid=1.0 => should adjust to reach min_notional 10
    req = OrderRequest(symbol='TESTUSDT', side='BUY', qty=0.5, order_type='LIMIT', price=1.00, client_id='t', idempotency_key='x')
    adj = await r._adjust_order('binance', req, ref_mid=1.0)  # type: ignore
    assert float(adj.qty) >= 10.0  # qty * mid
    # and obey step size 0.001
    assert abs((adj.qty / 0.001) - round(adj.qty / 0.001)) < 1e-6
