import asyncio
import pytest

from backend.exchanges.binance import BinanceAdapter
from backend.core.contracts import OrderRequest


class FakeResponse:
    def __init__(self, status: int, data: dict):
        self.status = status
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._data


class FakeSession:
    def __init__(self, routes: dict):
        self.routes = routes

    def get(self, url, params=None, timeout=None):
        key = ("GET", url)
        status, data = self.routes.get(key, (200, {}))
        return FakeResponse(status, data)

    def post(self, url, data=None, timeout=None):
        key = ("POST", url)
        status, resp = self.routes.get(key, (200, {}))
        return FakeResponse(status, resp)

    def delete(self, url, params=None, timeout=None):
        key = ("DELETE", url)
        status, resp = self.routes.get(key, (200, {}))
        return FakeResponse(status, resp)


@pytest.mark.asyncio
async def test_binance_healthcheck_spot(monkeypatch):
    tos = {"exchanges": {"binance": {"rate_limit_rps": 10}}}
    adp = BinanceAdapter(tos, mode="spot", api_key="k", api_secret="s")

    async def fake_get_session():
        routes = {("GET", "https://api.binance.com/api/v3/time"): (200, {"serverTime": 123})}
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    ok = await adp.healthcheck()
    assert ok is True


@pytest.mark.asyncio
async def test_binance_place_limit_success(monkeypatch):
    tos = {"exchanges": {"binance": {"rate_limit_rps": 10}}}
    adp = BinanceAdapter(tos, mode="spot", api_key="k", api_secret="s")
    routes = {
        ("GET", "https://api.binance.com/api/v3/time"): (200, {"serverTime": 123}),
        ("POST", "https://api.binance.com/api/v3/order"): (200, {"orderId": 98765}),
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    req = OrderRequest(symbol="BTCUSDT", side="BUY", order_type="LIMIT", qty=0.01, price=30000.0, client_id="t", idempotency_key="idem")
    ack = await adp.place_order(req, timeout_ms=500)
    assert ack.accepted is True and ack.broker_order_id == "98765"


@pytest.mark.asyncio
async def test_binance_place_error(monkeypatch):
    tos = {"exchanges": {"binance": {}}}
    adp = BinanceAdapter(tos, mode="spot", api_key="k", api_secret="s")
    routes = {
        ("GET", "https://api.binance.com/api/v3/time"): (200, {"serverTime": 123}),
        ("POST", "https://api.binance.com/api/v3/order"): (400, {"code": -1102, "msg": "Bad"}),
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    req = OrderRequest(symbol="BTCUSDT", side="BUY", order_type="MARKET", qty=0.01, client_id="t", idempotency_key="idem2")
    ack = await adp.place_order(req, timeout_ms=500)
    assert ack.accepted is False and "Bad" in (ack.message or "")


@pytest.mark.asyncio
async def test_binance_cancel_ok(monkeypatch):
    tos = {"exchanges": {"binance": {}}}
    adp = BinanceAdapter(tos, mode="spot", api_key="k", api_secret="s")
    routes = {
        ("GET", "https://api.binance.com/api/v3/time"): (200, {"serverTime": 123}),
        ("DELETE", "https://api.binance.com/api/v3/order"): (200, {"status": "CANCELED"}),
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    ok = await adp.cancel("idem", timeout_ms=500)
    assert ok is True


@pytest.mark.asyncio
async def test_binance_balances_parse_spot(monkeypatch):
    tos = {"exchanges": {"binance": {}}}
    adp = BinanceAdapter(tos, mode="spot", api_key="k", api_secret="s")
    routes = {
        ("GET", "https://api.binance.com/api/v3/time"): (200, {"serverTime": 123}),
        ("GET", "https://api.binance.com/api/v3/account"): (200, {"balances": [{"asset": "USDT", "free": "10", "locked": "1"}]})
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    bals = await adp.balances()
    assert "USDT" in bals and float(bals["USDT"].free) == 10.0

