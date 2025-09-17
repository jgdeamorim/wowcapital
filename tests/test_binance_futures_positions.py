import pytest

from backend.exchanges.binance import BinanceAdapter


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


@pytest.mark.asyncio
async def test_binance_positions_futures(monkeypatch):
    tos = {"exchanges": {"binance": {"rate_limit_rps": 10}}}
    adp = BinanceAdapter(tos, mode="futures", api_key="k", api_secret="s")
    routes = {
        ("GET", "https://fapi.binance.com/fapi/v1/time"): (200, {"serverTime": 123}),
        ("GET", "https://fapi.binance.com/fapi/v2/account"): (200, {"positions": [{"symbol":"BTCUSDT","positionAmt":"0.5"}]})
    }

    async def fake_get_session():
        return FakeSession(routes)

    monkeypatch.setattr(adp, "_get_session", fake_get_session)
    pos = await adp.positions()
    assert "BTCUSDT" in pos and float(pos["BTCUSDT"].qty) == 0.5

