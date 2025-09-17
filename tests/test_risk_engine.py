import pytest
from backend.risk.engine import RiskEngine


def test_min_max_notional_blocks():
    rk = RiskEngine()
    # configure globals via monkeypatch cfg
    rk.cfg = {
        'min_notional_usd': 100.0,
        'max_notional_usd': 1000.0,
        'slippage_cap_bps': 50.0,
        'price_guard_bps': 100.0,
        'instruments': [
            {'symbol':'BTCUSDT','min_notional_usd': 200.0}
        ]
    }
    req = {'symbol':'BTCUSDT','qty':0.005,'order_type':'MARKET','meta':{}}
    allowed, reason = rk.evaluate(req, ref_price=10000.0, spread_bps=10.0)
    assert allowed is False and reason == 'MIN_NOTIONAL'
    req2 = {'symbol':'BTCUSDT','qty':2.0,'order_type':'MARKET','meta':{}}
    allowed2, reason2 = rk.evaluate(req2, ref_price=600.0, spread_bps=10.0)
    assert allowed2 is False and reason2 == 'MAX_NOTIONAL'


def test_slippage_and_price_guard():
    rk = RiskEngine()
    rk.cfg = {
        'slippage_cap_bps': 20.0,
        'price_guard_bps': 50.0,
        'instruments': []
    }
    req = {'symbol':'ETHUSDT','qty':1.0,'order_type':'MARKET','meta':{}}
    allowed, reason = rk.evaluate(req, ref_price=2000.0, spread_bps=25.0)
    assert allowed is False and reason == 'SLIPPAGE_CAP'
    req2 = {'symbol':'ETHUSDT','qty':1.0,'order_type':'LIMIT','price': 2300.0,'meta':{}}
    allowed2, reason2 = rk.evaluate(req2, ref_price=2000.0, spread_bps=5.0)
    assert allowed2 is False and reason2 == 'PRICE_GUARD'

