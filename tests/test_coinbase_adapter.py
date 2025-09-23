import asyncio
from typing import Dict, Tuple
import base64

import pytest
import os
import sys
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

sys.path.insert(0, os.path.abspath("."))

try:
    from backend.core.contracts import OrderRequest
except ImportError:  # pragma: no cover
    from core.contracts import OrderRequest  # type: ignore

try:
    from backend.exchanges.coinbase import CoinbaseAdapter
except ImportError:  # pragma: no cover
    from exchanges.coinbase import CoinbaseAdapter  # type: ignore


_TEST_KEY = ec.generate_private_key(ec.SECP256K1())
TEST_PEM = _TEST_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()


class FakeResponse:
    def __init__(self, status: int, payload: Dict):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"status {self.status}")

    async def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes: Dict[Tuple[str, str], Tuple[int, Dict]]):
        self.routes = routes
        self.closed = False
        self.last_request = None

    def _match(self, method: str, url: str):
        return self.routes.get((method.upper(), url))

    def get(self, url, headers=None, timeout=None):
        self.last_request = ("GET", url, headers, None)
        status, payload = self._match("GET", url)
        return FakeResponse(status, payload)

    def post(self, url, headers=None, data=None, timeout=None):
        self.last_request = ("POST", url, headers, data)
        status, payload = self._match("POST", url)
        return FakeResponse(status, payload)


@pytest.mark.asyncio
async def test_product_id_mapping():
    adapter = CoinbaseAdapter({"exchanges": {"coinbase": {}}}, api_key="k", private_key_pem=TEST_PEM)
    assert adapter._product_id("BTCUSDT") == "BTC-USDT"
    assert adapter._product_id("ETHUSD") == "ETH-USD"
    assert adapter._product_id("SOL/USDC") == "SOL-USDC"


@pytest.mark.asyncio
async def test_place_market_order_success(monkeypatch):
    adapter = CoinbaseAdapter({"exchanges": {"coinbase": {}}}, api_key="k", private_key_pem=TEST_PEM)
    routes = {
        ("POST", "https://api.coinbase.com/api/v3/brokerage/orders"): (200, {"success": True, "order_id": "abc"})
    }
    session = FakeSession(routes)

    async def fake_get_session():
        return session

    monkeypatch.setattr(adapter, "_get_session", fake_get_session)

    req = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.01, order_type="MARKET", client_id="c1", idempotency_key="idem1")
    ack = await adapter.place_order(req, timeout_ms=1000)
    assert ack.accepted is True
    assert ack.broker_order_id == "abc"
    method, url, headers, data = session.last_request
    assert method == "POST"
    assert "brokerage/orders" in url
    assert headers["CB-ACCESS-KEY"] == "k"
    sig_bytes = base64.b64decode(headers["CB-ACCESS-SIGNATURE"])
    assert len(sig_bytes) == 64


@pytest.mark.asyncio
async def test_balances_parses(monkeypatch):
    adapter = CoinbaseAdapter({"exchanges": {"coinbase": {}}}, api_key="k", private_key_pem=TEST_PEM)
    payload = {
        "accounts": [
            {
                "currency": "USDC",
                "available_balance": {"value": "120.5"},
                "hold": {"value": "5.0"},
                "ready": {"value": "125.5"}
            }
        ]
    }
    routes = {
        ("GET", "https://api.coinbase.com/api/v3/brokerage/accounts"): (200, payload)
    }
    session = FakeSession(routes)

    async def fake_get_session():
        return session

    monkeypatch.setattr(adapter, "_get_session", fake_get_session)

    balances = await adapter.balances()
    assert "USDC" in balances
    bal = balances["USDC"]
    assert bal.free == 120.5
    assert bal.used == 5.0
    assert bal.total == 125.5
