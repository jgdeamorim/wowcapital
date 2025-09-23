#!/usr/bin/env python3
"""Quick Bybit live API check (public + account info)."""
import os
import time
import hmac
import hashlib
import requests
from typing import Dict

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise SystemExit("Set BYBIT_API_KEY/SECRET env vars")


def _sign(params: Dict[str, str], timestamp: str) -> str:
    sorted_params = "".join(f"{k}{v}" for k, v in sorted(params.items()))
    param_str = timestamp + BYBIT_API_KEY + sorted_params
    return hmac.new(BYBIT_API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()


def get_public_time():
    resp = requests.get(f"{BASE}/v5/market/time", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_account_info():
    endpoint = f"{BASE}/v5/account/info"
    params = {}
    timestamp = str(int(time.time() * 1000))
    signature = _sign(params, timestamp)
    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(endpoint, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("Public time:", get_public_time())
    try:
        info = get_account_info()
        print("Account info:", info)
    except Exception as exc:
        print("Account info error:", exc)
        if hasattr(exc, "response") and exc.response is not None:
            print("Response:", exc.response.text)
