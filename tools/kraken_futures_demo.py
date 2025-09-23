#!/usr/bin/env python3
"""Quick validation for Kraken Futures Demo endpoints (/time, /accounts, /sendorder, /cancelorder).

Reads credentials from config/local/kraken_bybit.env (same format already in repo).
Usage:
  python tools/kraken_futures_demo.py [--env-file path] [--symbol SYMBOL] [--size SIZE]
"""

from __future__ import annotations
import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple
import hmac
import hashlib
import requests

DEFAULT_ENV = Path(__file__).resolve().parents[1] / "config" / "local" / "kraken_bybit.env"
DEFAULT_BASE = "https://demo-futures.kraken.com/derivatives/api/v3"


def load_env(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def build_headers(api_key: str, api_secret: str, body: str = "") -> Tuple[Dict[str, str], str]:
    nonce = str(int(time.time() * 1000))
    message = nonce + api_key + body
    signature = hmac.new(
        base64.b64decode(api_secret),
        msg=message.encode(),
        digestmod=hashlib.sha512,
    ).digest()
    authent = base64.b64encode(signature).decode()
    headers = {
        "APIKey": api_key,
        "Authent": authent,
        "Nonce": nonce,
        "Content-Type": "application/json",
    }
    return headers, nonce


def perform_requests(base_url: str, api_key: str, api_secret: str, symbol: str, size: float) -> None:
    print("=== GET /time ===")
    resp_time = requests.get(f"{base_url}/time", timeout=10)
    print(resp_time.status_code, resp_time.text)

    print("\n=== GET /accounts ===")
    headers, _ = build_headers(api_key, api_secret)
    resp_accounts = requests.get(f"{base_url}/accounts", headers=headers, timeout=10)
    print(resp_accounts.status_code, resp_accounts.text)

    if resp_accounts.status_code != 200 or resp_accounts.json().get("result") != "success":
        print("⚠️  Skipping order tests because /accounts did not return success.")
        return

    order_body = json.dumps({"orderType": "mkt", "symbol": symbol, "side": "buy", "size": size})
    headers, _ = build_headers(api_key, api_secret, order_body)
    print("\n=== POST /sendorder ===")
    resp_send = requests.post(f"{base_url}/sendorder", headers=headers, data=order_body, timeout=10)
    print(resp_send.status_code, resp_send.text)

    data = resp_send.json()
    order_id = data.get("sendStatus", {}).get("order_id")
    if not order_id:
        print("⚠️  sendorder failed or did not return order_id; skipping cancel.")
        return

    cancel_body = json.dumps({"order_id": order_id})
    headers, _ = build_headers(api_key, api_secret, cancel_body)
    print("\n=== POST /cancelorder ===")
    resp_cancel = requests.post(f"{base_url}/cancelorder", headers=headers, data=cancel_body, timeout=10)
    print(resp_cancel.status_code, resp_cancel.text)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Kraken Futures Demo endpoint validator")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV, help="Env file with KRAKEN_FUTURES_API_KEY/SECRET")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="Base URL (default demo)")
    parser.add_argument("--symbol", default="pi_xbtusd", help="Contract symbol")
    parser.add_argument("--size", type=float, default=1.0, help="Order size in contracts")
    args = parser.parse_args(argv)

    env = load_env(args.env_file)
    api_key = env.get("KRAKEN_FUTURES_API_KEY") or env.get("KRAKEN_API_KEY")
    api_secret = env.get("KRAKEN_FUTURES_API_SECRET") or env.get("KRAKEN_PRIVATE_KEY")
    if not api_key or not api_secret:
        print("❌ Missing KRAKEN_FUTURES_API_KEY or SECRET in env file", file=sys.stderr)
        return 1

    try:
        perform_requests(args.base_url.rstrip("/"), api_key, api_secret, args.symbol, args.size)
    except Exception as exc:
        print(f"❌ Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
