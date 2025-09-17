import asyncio
import json
import pytest

from backend.exchanges.bybit import BybitAdapter
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

    def get(self, url, params=None, headers=None, timeout=None):
        key = ("GET", url)
        status, data = self.routes.get(key, (200, {}))
        return FakeResponse(status, data)

    def post(self, url, data=None, headers=None, timeout=None):
        key = ("POST", url)
        status, resp = self.routes.get(key, (200, {}))
        return FakeResponse(status, resp)


@pytest.mark.asyncio
async def test_bybit_healthcheck(monkeypatch):
    tos = {"exchanges": {"bybit": {"rate_limit_rps": 10}}}
    adp = BybitAdapter(tos, category="spot", api_key="k", api_secret="s")

    async def fake_get_session():
        routes = {("GET", "https://api.bybit.com/v5/market/time"): (200, {"result": {"timeSecond": "1700000000"}})}
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    ok = await adp.healthcheck()
    assert ok is True


@pytest.mark.asyncio
async def test_bybit_place_success(monkeypatch):
    tos = {"exchanges": {"bybit": {}}}
    adp = BybitAdapter(tos, category="spot", api_key="k", api_secret="s")
    routes = {("POST", "https://api.bybit.com/v5/order/create"): (200, {"retCode": 0, "result": {"orderId": "O-1"}})}

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    req = OrderRequest(symbol="BTCUSDT", side="BUY", order_type="MARKET", qty=0.01, client_id="t", idempotency_key="x")
    ack = await adp.place_order(req, timeout_ms=500)
    assert ack.accepted and ack.broker_order_id == "O-1"


@pytest.mark.asyncio
async def test_bybit_place_error(monkeypatch):
    tos = {"exchanges": {"bybit": {}}}
    adp = BybitAdapter(tos, category="spot", api_key="k", api_secret="s")
    routes = {("POST", "https://api.bybit.com/v5/order/create"): (200, {"retCode": 130001, "retMsg": "bad req"})}

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    req = OrderRequest(symbol="BTCUSDT", side="SELL", order_type="LIMIT", qty=0.02, price=30000.0, client_id="t", idempotency_key="y")
    ack = await adp.place_order(req, timeout_ms=500)
    assert not ack.accepted and "retCode" in (ack.message or "")


@pytest.mark.asyncio
async def test_bybit_cancel_and_balances(monkeypatch):
    tos = {"exchanges": {"bybit": {}}}
    adp = BybitAdapter(tos, category="spot", api_key="k", api_secret="s")
    # Cancel route
    routes = {
        ("POST", "https://api.bybit.com/v5/order/cancel"): (200, {"retCode": 0, "result": {"success": True}}),
        ("GET", "https://api.bybit.com/v5/account/wallet-balance?accountType=UNIFIED"): (200, {"retCode": 0, "result": {"list": [{"coin": [{"coin":"USDT","availableToWithdraw":"12.3","walletBalance":"12.3"}]}]}}),
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    ok = await adp.cancel("idem", timeout_ms=500)
    assert ok is True
    bals = await adp.balances()
    assert "USDT" in bals and float(bals["USDT"].total) == 12.3
