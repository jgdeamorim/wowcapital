#!/usr/bin/env python3
"""Diagnostic script for Coinbase Advanced Trade API authentication."""

import os
import time
import base64
import json
import requests
from typing import Dict, Any

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec


CLIENT_KEY = os.getenv("COINBASE_API_KEY")
KEY_ID = os.getenv("COINBASE_KEY_ID")
PEM_PATH = os.getenv("COINBASE_PRIVATE_KEY_PATH")
BASE_URL = os.getenv("COINBASE_BASE_URL", "https://api.coinbase.com/api/v3")

print("Loaded from env:")
print(f"  CLIENT_KEY: {CLIENT_KEY!r}")
print(f"  KEY_ID: {KEY_ID!r}")
print(f"  PEM_PATH: {PEM_PATH!r}")
print(f"  BASE_URL: {BASE_URL!r}")

if not CLIENT_KEY or not PEM_PATH:
    raise SystemExit("Missing COINBASE_API_KEY or COINBASE_PRIVATE_KEY_PATH")

# Load PEM
with open(PEM_PATH, "rb") as f:
    pem_data = f.read()
    print(f"PEM size: {len(pem_data)} bytes")

private_key = serialization.load_pem_private_key(pem_data, password=None)
print(f"Private key type: {type(private_key)}")

def sign(timestamp: str, method: str, path: str, body: str) -> str:
    message = f"{timestamp}{method}{path}{body}".encode()
    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    print(f"Raw signature len: {len(signature)}")
    # try base64 raw
    return base64.b64encode(signature).decode()

def call(method: str, path: str, body: Dict[str, Any] | None = None) -> requests.Response:
    body_str = json.dumps(body, separators=(",", ":")) if body else ""
    timestamp = str(int(time.time()))
    sig = sign(timestamp, method.upper(), path, body_str)
    headers = {
        "CB-ACCESS-KEY": CLIENT_KEY,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-SIGNATURE": sig,
    }
    print("Headers:", headers)
    url = BASE_URL.rstrip("/") + path
    resp = requests.request(method.upper(), url, headers=headers, data=body_str or None, timeout=10)
    print("Response status:", resp.status_code)
    print("Response headers:", resp.headers)
    print("Response text:", resp.text[:500])
    return resp

print("\n=== GET /brokerage/products ===")
call("GET", "/brokerage/products")

print("\n=== GET /brokerage/accounts ===")
call("GET", "/brokerage/accounts")
