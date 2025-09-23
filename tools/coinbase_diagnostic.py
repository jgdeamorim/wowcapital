#!/usr/bin/env python3
import os, time, json, traceback
from coinbase.rest import RESTClient
from coinbase.websocket import WSClient, WebsocketResponse

# ============= REST TESTS =============
def test_rest(api_key=None, api_secret=None):
    print("\n=== [RESTClient Test] ===")
    try:
        if api_key and api_secret:
            client = RESTClient(api_key=api_key, api_secret=api_secret, verbose=True)
        else:
            client = RESTClient()  # só público

        # Testa contas
        try:
            accounts = client.get_accounts()
            print("Accounts →", json.dumps(accounts.to_dict(), indent=2)[:300])
        except Exception as e:
            print("Accounts ERR:", e)

        # Testa produtos
        try:
            products = client.get_products()
            print("Products →", json.dumps(products.to_dict(), indent=2)[:300])
        except Exception as e:
            print("Products ERR:", e)

    except Exception:
        traceback.print_exc()

# ============= WS TESTS =============
def test_ws(api_key=None, api_secret=None):
    print("\n=== [WSClient Test] ===")

    def on_message(msg):
        try:
            ws_obj = WebsocketResponse(json.loads(msg))
            if ws_obj.channel == "ticker":
                for event in ws_obj.events:
                    for ticker in event.tickers:
                        print(f"TICKER {ticker.product_id}: {ticker.price}")
        except Exception:
            print("WS RAW:", msg)

    try:
        if api_key and api_secret:
            client = WSClient(api_key=api_key, api_secret=api_secret, on_message=on_message, verbose=True)
        else:
            client = WSClient(on_message=on_message)

        client.open()
        client.ticker(product_ids=["BTC-USD", "ETH-USD"])
        time.sleep(5)
        client.ticker_unsubscribe(product_ids=["BTC-USD", "ETH-USD"])
        client.close()

    except Exception:
        traceback.print_exc()

# ============= MAIN RUN =============
if __name__ == "__main__":
    API_KEY = os.getenv("COINBASE_API_KEY")   # organizations/.../apiKeys/...
    API_SECRET = os.getenv("COINBASE_API_SECRET")  # -----BEGIN EC PRIVATE KEY----- ...

    print(">> Running Coinbase SDK diagnostics...")

    # Testes REST
    test_rest(API_KEY, API_SECRET)
    test_rest()  # só público

    # Testes WS
    test_ws(API_KEY, API_SECRET)
    test_ws()  # só público
