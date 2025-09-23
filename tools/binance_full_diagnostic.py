#!/usr/bin/env python3
"""
Binance Full Diagnostic Tool

Testa:
 - REST API (account, exchange info, ticker, order test)
 - WebSocket streams (aggTrade, depth, ticker)

Suporta:
 - Credenciais via argumentos CLI (--api-key, --api-secret)
 - Arquivo .env (via --env-file)
 - Variáveis de ambiente já exportadas

Ambientes:
 - Produção (https://api.binance.com)
 - Sandbox/Testnet (https://testnet.binance.vision)

Exemplos:
  python tools/binance_full_diagnostic.py --env-file config/local/binance_demo.env --sandbox --symbols BTCUSDT,ETHUSDT
  python tools/binance_full_diagnostic.py --env-file config/local/binance_real.env --prod --symbols BTCUSDT --ws-secs 15
"""

import argparse
import os
import sys
import time
from dotenv import load_dotenv
from binance.spot import Spot as SpotClient

# WebSocket client compatível com múltiplas versões (ou ausência) do binance-connector
WSClient = None
WS_CLIENT_KIND = None
try:
    from binance.websocket.spot.websocket_client import SpotWebsocketClient as WSClient
    WS_CLIENT_KIND = "client"
except ImportError:
    try:
        from binance.websocket.spot import SpotWebsocketClient as WSClient  # type: ignore
        WS_CLIENT_KIND = "client"
    except ImportError:
        try:
            from binance.websocket.spot import SpotWebsocketStreamClient as WSClient  # type: ignore
            WS_CLIENT_KIND = "stream"
        except ImportError:
            try:
                from binance.websocket.spot.websocket_api import SpotWebsocket as WSClient  # type: ignore
                WS_CLIENT_KIND = "api"
            except ImportError:
                WSClient = None
                WS_CLIENT_KIND = None


# ----------------- Helpers -----------------

def load_credentials(args):
    """Carrega credenciais da linha de comando, .env ou env vars."""
    if args.env_file:
        if not os.path.exists(args.env_file):
            raise SystemExit(f"❌ Arquivo .env não encontrado: {args.env_file}")
        load_dotenv(args.env_file)

    api_key = args.api_key or os.getenv("BINANCE_API_KEY")
    api_secret = args.api_secret or os.getenv("BINANCE_API_SECRET")
    base_url = os.getenv("BINANCE_BASE_URL")

    if args.sandbox:
        base_url = "https://testnet.binance.vision"
    elif args.prod:
        base_url = "https://api.binance.com"

    if not api_key or not api_secret:
        raise SystemExit("❌ BINANCE_API_KEY / BINANCE_API_SECRET não definidos (use --api-key/--api-secret ou --env-file)")

    return api_key, api_secret, base_url


def build_spot_client(api_key, api_secret, base_url, verbose=False):
    return SpotClient(api_key=api_key, api_secret=api_secret, base_url=base_url, show_limit_usage=verbose)


# ----------------- REST -----------------

def rest_tests(client, symbols):
    print("=== REST API TESTS ===")

    try:
        acc = client.account()
        balances = [b for b in acc["balances"] if float(b["free"]) > 0 or float(b["locked"]) > 0]
        print("✅ account() OK")
        if balances:
            for b in balances[:5]:
                print(f"   → {b['asset']}: free={b['free']} locked={b['locked']}")
        else:
            print("   → Nenhum saldo disponível")
    except Exception as e:
        print("❌ account() error:", e)

    try:
        ex_info = client.exchange_info()
        print(f"✅ exchange_info() OK, {len(ex_info['symbols'])} pares disponíveis")
    except Exception as e:
        print("❌ exchange_info() error:", e)

    for s in symbols:
        try:
            t = client.ticker_price(s)
            print(f"✅ ticker_price({s}) = {t['price']}")
        except Exception as e:
            print(f"❌ ticker_price({s}) error:", e)

    # order/test (não cria ordem real)
    for s in symbols[:1]:  # testar só o 1º símbolo
        try:
            resp = client.new_order_test(symbol=s, side="BUY", type="MARKET", quantity=0.001)
            print(f"✅ new_order_test({s}) OK (simulação de ordem)")
        except Exception as e:
            print(f"❌ new_order_test({s}) error:", e)


# ----------------- WEBSOCKET -----------------

def ws_tests(symbols, duration, stream_url):
    print("\n=== WEBSOCKET TESTS ===")

    if WSClient is None:
        print("⚠️ WebSocket Spot não disponível nesta instalação do binance-connector. Pulando testes WS.")
        return

    if WS_CLIENT_KIND == "api":
        print("⚠️ Cliente websocket_api detectado; testes WS ainda não suportados neste diagnóstico. Pulando.")
        return

    ws_client = WSClient(stream_url=stream_url) if WS_CLIENT_KIND in {"client", "stream"} else WSClient()

    def handle(msg):
        if isinstance(msg, dict) and "s" in msg and "p" in msg:
            print(f"📈 {msg['s']}: {msg['p']}")
        else:
            print("WS Raw:", msg)

    if hasattr(ws_client, "start"):
        ws_client.start()

    try:
        for s in symbols:
            symbol_lower = s.lower()
            if hasattr(ws_client, "agg_trade"):
                ws_client.agg_trade(symbol=symbol_lower, id=f"aggtrade-{s}", callback=handle)
            elif hasattr(ws_client, "agg_trade_subscribe"):
                ws_client.agg_trade_subscribe(symbol=symbol_lower, id=f"aggtrade-{s}", callback=handle)
            else:
                print("⚠️ Cliente WS sem método agg_trade compatível. Pulando streams.")
                return

        time.sleep(duration)
    finally:
        if hasattr(ws_client, "stop"):
            ws_client.stop()
        elif hasattr(ws_client, "close"):
            ws_client.close()
    print("✅ WS stream finalizado")


# ----------------- Runner -----------------

def main():
    parser = argparse.ArgumentParser(description="Binance Full Diagnostic Tool")
    parser.add_argument("--api-key", help="API Key (override)")
    parser.add_argument("--api-secret", help="API Secret (override)")
    parser.add_argument("--env-file", help="Arquivo .env para carregar credenciais", default=None)
    parser.add_argument("--sandbox", action="store_true", help="Usar Binance Testnet")
    parser.add_argument("--prod", action="store_true", help="Usar Binance Prod")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="Símbolos separados por vírgula")
    parser.add_argument("--ws-secs", type=int, default=10, help="Duração WebSocket em segundos")
    parser.add_argument("--verbose", action="store_true", help="Mostrar cabeçalhos de limite de rate")
    args = parser.parse_args()

    api_key, api_secret, base_url = load_credentials(args)
    env_name = "SANDBOX" if args.sandbox else "PROD"
    print(f"\n🌍 Testing environment: {env_name} ({base_url})")

    client = build_spot_client(api_key, api_secret, base_url, args.verbose)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # REST
    rest_tests(client, symbols)

    # WS
    stream_url = "wss://testnet.binance.vision/ws" if args.sandbox else "wss://stream.binance.com:9443/ws"
    ws_tests(symbols, args.ws_secs, stream_url)


if __name__ == "__main__":
    sys.exit(main())
