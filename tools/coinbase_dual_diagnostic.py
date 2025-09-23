#!/usr/bin/env python3
import os, time, base64, json, requests, jwt
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

# === ENV ===
COINBASE_CLIENT_KEY   = os.getenv("COINBASE_CLIENT_KEY")
COINBASE_API_KEY      = os.getenv("COINBASE_API_KEY")
COINBASE_KEY_ID       = os.getenv("COINBASE_KEY_ID")
PEM_PATH              = os.getenv("COINBASE_PRIVATE_KEY_PATH")

CDP_API_KEY_ID        = os.getenv("CDP_API_KEY_ID")
CDP_SECRET_API_KEY    = os.getenv("CDP_SECRET_API_KEY")

TRADING_MODE          = os.getenv("TRADING_MODE", "real").lower()
ENABLE_REAL_TRADING   = os.getenv("ENABLE_REAL_TRADING", "false").lower() in ["1","true","yes"]

# === BASE URL candidates ===
ADVANCED_BASES = [
    "https://api.coinbase.com/api/v3",        # real
    "https://api-sandbox.coinbase.com/api/v3" # sandbox
]
CDP_BASES = [
    "https://api.cdp.coinbase.com/v1"
]

# === Load ECDSA private key ===
PRIVATE_KEY = None
if PEM_PATH and os.path.exists(PEM_PATH):
    with open(PEM_PATH, "rb") as f:
        PRIVATE_KEY = serialization.load_pem_private_key(f.read(), password=None)

def sign_ecdsa(ts: str, method: str, path: str, body: str) -> str:
    """ECDSA signature for Advanced Trading"""
    msg = f"{ts}{method.upper()}{path}{body}".encode()
    sig = PRIVATE_KEY.sign(msg, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(sig)
    return base64.b64encode(r.to_bytes(32,"big") + s.to_bytes(32,"big")).decode()

def request_advanced(base, key, path, method="GET", body=None):
    """Signed request for Advanced Trading"""
    body_str = json.dumps(body,separators=(",",":")) if body else ""
    ts = str(int(time.time()))
    headers = {
        "CB-ACCESS-KEY": key,
        "CB-ACCESS-TIMESTAMP": ts,
        "CB-ACCESS-SIGNATURE": sign_ecdsa(ts, method, path, body_str),
        "Content-Type": "application/json"
    }
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, headers=headers, data=body_str or None, timeout=10)
        return r.status_code, r.text[:300]
    except Exception as e:
        return "ERR", str(e)

def request_cdp(base, path, method="GET", body=None):
    """JWT Ed25519 for CDP"""
    now = int(time.time())
    payload = {
        "sub": CDP_API_KEY_ID,
        "iss": "cdp",
        "nbf": now,
        "iat": now,
        "exp": now + 120
    }

    # Decodifica secret base64 → chave privada Ed25519
    try:
        key_bytes = base64.b64decode(CDP_SECRET_API_KEY)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
    except Exception as e:
        return "ERR", f"Invalid Ed25519 key: {e}"

    try:
        token = jwt.encode(payload, private_key, algorithm="EdDSA")
    except Exception as e:
        return "ERR", f"JWT encode error: {e}"

    headers = {"Authorization": f"Bearer {token}"}
    url = base.rstrip("/") + path
    try:
        r = requests.request(method, url, headers=headers, json=body or None, timeout=10)
        return r.status_code, r.text[:300]
    except Exception as e:
        return "ERR", str(e)

def run_matrix():
    scenarios = [
        ("real", True),
        ("demo", True),
        ("real", False),
        ("demo", False),
    ]
    for mode, trading_flag in scenarios:
        print(f"\n=== 🔎 Scenario: TRADING_MODE={mode.upper()}, ENABLE_REAL_TRADING={trading_flag} ===")
        
        # Advanced Trading tests
        print("\n--- Advanced Trading (ECDSA) ---")
        if not PRIVATE_KEY or not COINBASE_CLIENT_KEY:
            print("⚠️ Missing ECDSA config, skipping…")
        else:
            for base in ADVANCED_BASES:
                print(f"\n[BASE={base}]")
                for path in ["/brokerage/accounts", "/brokerage/products"]:
                    status, body = request_advanced(base, COINBASE_CLIENT_KEY, path)
                    print(f"{path} → {status} | {body}")

        # CDP tests
        print("\n--- CDP (Ed25519 JWT) ---")
        if not CDP_API_KEY_ID or not CDP_SECRET_API_KEY:
            print("⚠️ Missing CDP config, skipping…")
        else:
            for base in CDP_BASES:
                print(f"\n[BASE={base}]")
                for path in ["/wallets", "/accounts"]:
                    status, body = request_cdp(base, path)
                    print(f"{path} → {status} | {body}")

if __name__ == "__main__":
    run_matrix()
