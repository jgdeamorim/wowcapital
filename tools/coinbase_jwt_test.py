#!/usr/bin/env python3
import os, time, requests, jwt

# --- ENV (ajuste conforme seu .env) ---
API_KEY_ID = os.getenv("COINBASE_KEY_ID")   # ex: a4f087f7-d90a-...
PEM_PATH   = os.getenv("COINBASE_PRIVATE_KEY_PATH", "config/local/coinbase/private_key_coinbase.pem")
BASE_URL   = os.getenv("COINBASE_BASE_URL", "https://api.cdp.coinbase.com")

if not API_KEY_ID or not PEM_PATH:
    raise SystemExit("⚠️ Configure COINBASE_KEY_ID e COINBASE_PRIVATE_KEY_PATH no .env")

with open(PEM_PATH, "r") as f:
    PRIVATE_KEY = f.read()

# --- JWT claims ---
now = int(time.time())
payload = {
    "sub": API_KEY_ID,   # API Key ID
    "iss": "cdp",
    "nbf": now,
    "iat": now,
    "exp": now + 120,    # expira em 2 min
}

# --- gerar JWT ---
token = jwt.encode(
    payload,
    PRIVATE_KEY,
    algorithm="ES256"   # ECDSA P-256 + SHA-256
)

print("🔑 Generated JWT:", token)

# --- chamada teste ---
url = BASE_URL.rstrip("/") + "/v1/brokerage/accounts"  # <- ✅ endpoint correto
headers = {"Authorization": f"Bearer {token}"}

resp = requests.get(url, headers=headers, timeout=10)

print("\n=== GET /v1/brokerage/accounts ===")
print("Status:", resp.status_code)
print("Body:", resp.text[:500])
