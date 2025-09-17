import base64
import asyncio
import types
import pytest

from backend.exchanges.kraken import KrakenAdapter
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
        # routes: (method, url) -> (status, data)
        self.routes = routes

    def get(self, url, timeout=None):
        key = ("GET", url)
        status, data = self.routes.get(key, (200, {}))
        return FakeResponse(status, data)

    def post(self, url, data=None, headers=None, timeout=None):
        key = ("POST", url)
        status, resp = self.routes.get(key, (200, {}))
        return FakeResponse(status, resp)


@pytest.mark.asyncio
async def test_kraken_healthcheck_ok(monkeypatch):
    tos = {"exchanges": {"kraken": {"rate_limit_rps": 5}}}
    adp = KrakenAdapter(tos, api_key="k", api_secret_b64=base64.b64encode(b"s").decode())

    async def fake_get_session():
        routes = {("GET", "https://api.kraken.com/0/public/Time"): (200, {"result": {"unixtime": 123}})}
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    ok = await adp.healthcheck()
    assert ok is True


def test_kraken_sign_matches_reference():
    tos = {"exchanges": {"kraken": {}}}
    secret = base64.b64encode(b"mysecret").decode()
    adp = KrakenAdapter(tos, api_key="key", api_secret_b64=secret)
    path = "/0/private/AddOrder"
    data = {"nonce": "1700000000000", "ordertype": "limit", "type": "buy", "volume": "0.01", "pair": "BTCUSDT", "price": "30000"}
    sig = adp._sign(path, data)
    # recompute reference signature
    from hashlib import sha256
    import hmac
    from urllib.parse import urlencode
    postdata = urlencode(data)
    sha = sha256((data["nonce"] + postdata).encode()).digest()
    message = path.encode() + sha
    ref = hmac.new(base64.b64decode(secret), message, digestmod="sha512").digest()
    ref_b64 = base64.b64encode(ref).decode()
    assert sig == ref_b64


@pytest.mark.asyncio
async def test_place_order_success(monkeypatch):
    tos = {"exchanges": {"kraken": {"rate_limit_rps": 5}}}
    adp = KrakenAdapter(tos, api_key="key", api_secret_b64=base64.b64encode(b"s").decode())

    routes = {
        ("POST", "https://api.kraken.com/0/private/AddOrder"): (200, {"error": [], "result": {"txid": ["O-ABC"]}})
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    req = OrderRequest(symbol="BTCUSDT", side="BUY", order_type="LIMIT", qty=0.01, price=30000.0, client_id="t", idempotency_key="idem-1")
    ack = await adp.place_order(req, timeout_ms=500)
    assert ack.accepted is True
    assert ack.broker_order_id == "O-ABC"


@pytest.mark.asyncio
async def test_place_order_error(monkeypatch):
    tos = {"exchanges": {"kraken": {}}}
    adp = KrakenAdapter(tos, api_key="key", api_secret_b64=base64.b64encode(b"s").decode())
    routes = {
        ("POST", "https://api.kraken.com/0/private/AddOrder"): (200, {"error": ["EGeneral:Invalid arguments"]})
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    req = OrderRequest(symbol="BTCUSDT", side="BUY", order_type="MARKET", qty=0.01, client_id="t", idempotency_key="idem-2")
    ack = await adp.place_order(req, timeout_ms=500)
    assert ack.accepted is False
    assert "Invalid" in (ack.message or "")


@pytest.mark.asyncio
async def test_cancel_ok(monkeypatch):
    tos = {"exchanges": {"kraken": {}}}
    adp = KrakenAdapter(tos, api_key="key", api_secret_b64=base64.b64encode(b"s").decode())
    routes = {
        ("POST", "https://api.kraken.com/0/private/CancelOrder"): (200, {"error": [], "result": {"count": 1}})
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    ok = await adp.cancel("O-ABC", timeout_ms=500)
    assert ok is True


@pytest.mark.asyncio
async def test_balances_parse(monkeypatch):
    tos = {"exchanges": {"kraken": {}}}
    adp = KrakenAdapter(tos, api_key="key", api_secret_b64=base64.b64encode(b"s").decode())
    routes = {
        ("POST", "https://api.kraken.com/0/private/Balance"): (200, {"error": [], "result": {"ZUSD": "100.0", "XXBT": "0.01"}})
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    bals = await adp.balances()
    assert "ZUSD" in bals and float(bals["ZUSD"].total) == 100.0
    assert "XXBT" in bals and float(bals["XXBT"].total) == 0.01
