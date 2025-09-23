#!/usr/bin/env python3
"""Coinbase Advanced Trade SDK diagnostics.

Features
========
- Loads ECDSA credentials from JSON export, PEM file or environment variables.
- Exercises REST endpoints (accounts + BTC-USD price) and WebSocket ticker stream.
- Supports production and sandbox environments via CLI flags.

Examples
--------
$ python tools/coinbase_sdk_test.py --prod --key-file config/local/coinbase/cdp_api_key_ECDSA.json
$ python tools/coinbase_sdk_test.py --sandbox --ws-duration 5
$ COINBASE_API_KEY=... COINBASE_API_SECRET=... python tools/coinbase_sdk_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from coinbase.rest import RESTClient
from coinbase.websocket import WSClient, WebsocketResponse


ENV_HOST = {
    "prod": "api.coinbase.com",
    "sandbox": "api-sandbox.coinbase.com",
}

DISPLAY_URL = {
    "prod": "https://api.coinbase.com/api/v3",
    "sandbox": "https://api-sandbox.coinbase.com/api/v3",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json_credentials(path: Path) -> Tuple[str, str]:
    data = json.loads(_read(path))
    api_key = data.get("name") or data.get("apiKey")
    private_key = data.get("privateKey") or data.get("private_key")
    if not api_key or not private_key:
        raise ValueError("JSON não possui campos 'name'/'apiKey' e 'privateKey'")
    return api_key, private_key


def _resolve_env_credentials() -> Tuple[Optional[str], Optional[str]]:
    return os.getenv("COINBASE_API_KEY"), os.getenv("COINBASE_API_SECRET")


def rest_tests(client: RESTClient, env: str) -> None:
    print("=== REST API TESTS ===")
    try:
        accounts = client.get_accounts()
        sample = accounts.accounts[0] if getattr(accounts, "accounts", []) else None
        print("✅ get_accounts OK")
        if sample:
            value = sample.available_balance.get("value") if isinstance(sample.available_balance, dict) else sample.available_balance
            print(f"   → Primeira carteira: {sample.name} ({sample.currency}) saldo={value}")
    except Exception as exc:
        print("❌ REST get_accounts error:", exc)

    try:
        product = client.get_product("BTC-USD")
        print(f"✅ BTC-USD Price: {product.price}")
    except Exception as exc:
        message = str(exc)
        if "404" in message and env != "prod":
            print("ℹ️  BTC-USD indisponível no ambiente sandbox (404)")
        else:
            print("❌ REST get_product error:", exc)


def ws_tests(client: WSClient, duration: int) -> None:
    print("\n=== WEBSOCKET TESTS ===")

    def on_message(msg: str) -> None:
        try:
            ws_obj = WebsocketResponse(json.loads(msg))
            if ws_obj.channel == "ticker":
                for event in ws_obj.events:
                    for ticker in event.tickers:
                        print(f"📈 {ticker.product_id}: {ticker.price}")
        except Exception:
            print("WS RAW:", msg)

    client.on_message = on_message
    client.open()
    client.ticker(product_ids=["BTC-USD", "ETH-USD"])
    time.sleep(duration)
    client.ticker_unsubscribe(product_ids=["BTC-USD", "ETH-USD"])
    client.close()
    print("✅ WS stream finalizado")


def create_clients(env: str, key_file: Optional[Path], verbose: bool) -> Tuple[RESTClient, WSClient]:
    host = ENV_HOST[env]
    if key_file:
        expanded = key_file.expanduser().resolve()
        if not expanded.exists():
            raise FileNotFoundError(f"Arquivo de credenciais não encontrado: {expanded}")
        rest_client = RESTClient(key_file=str(expanded), base_url=host, verbose=verbose)
        ws_client = WSClient(key_file=str(expanded), on_message=lambda msg: None, verbose=verbose)
        return rest_client, ws_client

    api_key, api_secret = _resolve_env_credentials()
    if not api_key or not api_secret:
        raise SystemExit("⚠️ Defina COINBASE_API_KEY/COINBASE_API_SECRET ou informe --key-file")

    rest_client = RESTClient(api_key=api_key, api_secret=api_secret, base_url=host, verbose=verbose)
    ws_client = WSClient(api_key=api_key, api_secret=api_secret, on_message=lambda msg: None, verbose=verbose)
    return rest_client, ws_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coinbase Advanced Trade SDK diagnostics")
    parser.add_argument("--key-file", help="Path to JSON key export", default=os.getenv("COINBASE_KEY_FILE"))
    parser.add_argument("--sandbox", action="store_true", help="Executa apenas sandbox")
    parser.add_argument("--prod", action="store_true", help="Executa apenas produção")
    parser.add_argument("--all", action="store_true", help="Executa sandbox e produção")
    parser.add_argument("--ws-duration", type=int, default=10, help="Segundos de stream WS")
    parser.add_argument("--verbose", action="store_true", help="Habilita logs verbosos do SDK")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.all:
        envs = ["sandbox", "prod"]
    elif args.sandbox:
        envs = ["sandbox"]
    elif args.prod:
        envs = ["prod"]
    else:
        envs = ["prod"]

    key_file_path = Path(args.key_file).expanduser() if args.key_file else None

    for env in envs:
        print(f"\n🌍 Testing environment: {env.upper()} ({DISPLAY_URL[env]})")
        if key_file_path:
            print(f"🔑 Using key file: {key_file_path.resolve()}")
        rest_client, ws_client = create_clients(env, key_file_path, args.verbose)
        rest_tests(rest_client, env)
        ws_tests(ws_client, args.ws_duration)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
