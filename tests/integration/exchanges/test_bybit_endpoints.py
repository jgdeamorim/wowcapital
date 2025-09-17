#!/usr/bin/env python3
"""
Teste de Endpoints Bybit
Testa conectividade e operações básicas com a API Bybit (testnet)

IMPORTANTE: APENAS TESTNET!

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import sys
import os
import time
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional
import json
import logging
from urllib.parse import urlencode

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tests.integration.exchanges.credentials_manager import CredentialsManager


class BybitAPITester:
    """Testador da API Bybit"""

    def __init__(self, credentials_manager: CredentialsManager):
        self.cred_manager = credentials_manager
        self.credentials = credentials_manager.load_credentials()

        if not self.credentials.bybit:
            raise ValueError("Credenciais Bybit não disponíveis")

        # Bybit API endpoints
        if self.credentials.bybit.environment == 'testnet':
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        self.api_key = self.credentials.bybit.api_key
        self.api_secret = self.credentials.bybit.api_secret

        self.logger = logging.getLogger(__name__)

    def _generate_signature(self, timestamp: str, params: str) -> str:
        """Gera assinatura para autenticação Bybit"""

        param_str = f"{timestamp}{self.api_key}{params}"
        return hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _make_public_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Faz requisição pública para Bybit"""

        url = f"{self.base_url}{endpoint}"
        params = params or {}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro na requisição pública {endpoint}: {str(e)}")
            raise

    def _make_private_request(self, endpoint: str, params: Optional[Dict] = None, method: str = "GET") -> Dict[str, Any]:
        """Faz requisição privada para Bybit"""

        params = params or {}
        timestamp = str(int(time.time() * 1000))

        # Prepare parameters
        if method == "GET":
            param_str = urlencode(sorted(params.items()))
        else:
            param_str = json.dumps(params) if params else ""

        # Generate signature
        signature = self._generate_signature(timestamp, param_str)

        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': timestamp,
            'Content-Type': 'application/json'
        }

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            else:
                response = requests.post(url, json=params, headers=headers, timeout=10)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro na requisição privada {endpoint}: {str(e)}")
            raise

    def test_server_time(self) -> Dict[str, Any]:
        """Testa endpoint de tempo do servidor"""

        self.logger.info("Testando server time...")

        try:
            result = self._make_public_request("/v5/market/time")

            if result.get('retCode') != 0:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'endpoint': '/v5/market/time'
                }

            server_time = int(result.get('result', {}).get('timeSecond', 0))
            local_time = int(time.time())
            time_diff = abs(server_time - local_time)

            return {
                'success': True,
                'server_time': server_time,
                'local_time': local_time,
                'time_difference_seconds': time_diff,
                'sync_ok': time_diff < 30,
                'endpoint': '/v5/market/time'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': '/v5/market/time'
            }

    def test_instruments_info(self) -> Dict[str, Any]:
        """Testa endpoint de informações de instrumentos"""

        self.logger.info("Testando instruments info...")

        try:
            result = self._make_public_request("/v5/market/instruments-info", {
                "category": "spot",
                "limit": 10
            })

            if result.get('retCode') != 0:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'endpoint': '/v5/market/instruments-info'
                }

            instruments = result.get('result', {}).get('list', [])
            btc_instruments = [instr for instr in instruments if 'BTC' in instr.get('symbol', '')]

            return {
                'success': True,
                'total_instruments': len(instruments),
                'btc_instruments_count': len(btc_instruments),
                'sample_instruments': [instr.get('symbol') for instr in instruments[:5]],
                'btc_instruments': [instr.get('symbol') for instr in btc_instruments[:3]],
                'endpoint': '/v5/market/instruments-info'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': '/v5/market/instruments-info'
            }

    def test_ticker_data(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Testa endpoint de ticker"""

        self.logger.info(f"Testando ticker para {symbol}...")

        try:
            result = self._make_public_request("/v5/market/tickers", {
                "category": "spot",
                "symbol": symbol
            })

            if result.get('retCode') != 0:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'endpoint': '/v5/market/tickers',
                    'symbol': symbol
                }

            tickers = result.get('result', {}).get('list', [])
            if not tickers:
                return {
                    'success': False,
                    'error': 'No ticker data found',
                    'endpoint': '/v5/market/tickers',
                    'symbol': symbol
                }

            ticker = tickers[0]

            return {
                'success': True,
                'symbol': ticker.get('symbol'),
                'bid1_price': ticker.get('bid1Price'),
                'ask1_price': ticker.get('ask1Price'),
                'last_price': ticker.get('lastPrice'),
                'volume24h': ticker.get('volume24h'),
                'high_price24h': ticker.get('highPrice24h'),
                'low_price24h': ticker.get('lowPrice24h'),
                'price_change24h': ticker.get('price24hPcnt'),
                'endpoint': '/v5/market/tickers'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': '/v5/market/tickers',
                'symbol': symbol
            }

    def test_account_wallet_balance(self) -> Dict[str, Any]:
        """Testa endpoint de saldo da carteira"""

        self.logger.info("Testando wallet balance...")

        try:
            result = self._make_private_request("/v5/account/wallet-balance", {
                "accountType": "UNIFIED"
            })

            if result.get('retCode') != 0:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'endpoint': '/v5/account/wallet-balance'
                }

            wallet_list = result.get('result', {}).get('list', [])

            if not wallet_list:
                return {
                    'success': True,
                    'total_wallets': 0,
                    'coins': [],
                    'endpoint': '/v5/account/wallet-balance'
                }

            wallet = wallet_list[0]
            coins = wallet.get('coin', [])

            # Filter coins with balance > 0
            coins_with_balance = [
                coin for coin in coins
                if float(coin.get('equity', '0')) > 0
            ]

            return {
                'success': True,
                'account_type': wallet.get('accountType'),
                'total_equity': wallet.get('totalEquity'),
                'total_wallet_balance': wallet.get('totalWalletBalance'),
                'total_coins': len(coins),
                'coins_with_balance': len(coins_with_balance),
                'coins': [
                    {
                        'coin': coin.get('coin'),
                        'equity': coin.get('equity'),
                        'wallet_balance': coin.get('walletBalance'),
                        'available': coin.get('availableToWithdraw')
                    }
                    for coin in coins_with_balance
                ],
                'endpoint': '/v5/account/wallet-balance'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': '/v5/account/wallet-balance'
            }

    def test_open_orders(self) -> Dict[str, Any]:
        """Testa endpoint de ordens abertas"""

        self.logger.info("Testando open orders...")

        try:
            result = self._make_private_request("/v5/order/realtime", {
                "category": "spot"
            })

            if result.get('retCode') != 0:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'endpoint': '/v5/order/realtime'
                }

            orders = result.get('result', {}).get('list', [])

            return {
                'success': True,
                'open_orders_count': len(orders),
                'orders': [
                    {
                        'order_id': order.get('orderId'),
                        'symbol': order.get('symbol'),
                        'side': order.get('side'),
                        'order_type': order.get('orderType'),
                        'qty': order.get('qty'),
                        'price': order.get('price'),
                        'order_status': order.get('orderStatus')
                    }
                    for order in orders
                ],
                'endpoint': '/v5/order/realtime'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': '/v5/order/realtime'
            }

    def test_account_info(self) -> Dict[str, Any]:
        """Testa endpoint de informações da conta"""

        self.logger.info("Testando account info...")

        try:
            result = self._make_private_request("/v5/account/info")

            if result.get('retCode') != 0:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'endpoint': '/v5/account/info'
                }

            account_info = result.get('result', {})

            return {
                'success': True,
                'unified_margin_status': account_info.get('unifiedMarginStatus'),
                'margin_mode': account_info.get('marginMode'),
                'dcpStatus': account_info.get('dcpStatus'),
                'spot_hedging': account_info.get('spotHedgingStatus'),
                'endpoint': '/v5/account/info'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': '/v5/account/info'
            }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Executa teste abrangente da API Bybit"""

        self.logger.info("Iniciando teste abrangente Bybit...")

        results = {}

        # Test public endpoints
        results['server_time'] = self.test_server_time()
        results['instruments_info'] = self.test_instruments_info()
        results['ticker_btcusdt'] = self.test_ticker_data("BTCUSDT")

        # Test private endpoints
        results['account_info'] = self.test_account_info()
        results['wallet_balance'] = self.test_account_wallet_balance()
        results['open_orders'] = self.test_open_orders()

        # Calculate success rate
        successful_tests = sum(1 for result in results.values() if result.get('success', False))
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100

        return {
            'exchange': 'Bybit',
            'environment': self.credentials.bybit.environment,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'results': results,
            'overall_success': success_rate >= 80
        }


def main():
    """Teste principal Bybit"""

    logging.basicConfig(level=logging.INFO)

    print("🏦 WOW Capital - Bybit API Test")
    print("=" * 50)

    try:
        # Load credentials
        cred_manager = CredentialsManager()
        credentials = cred_manager.load_credentials()

        if not credentials.bybit:
            print("❌ Credenciais Bybit não disponíveis")
            return False

        # Create tester
        bybit_tester = BybitAPITester(cred_manager)

        # Run comprehensive test
        results = bybit_tester.run_comprehensive_test()

        # Display results
        print(f"\n📊 Resultados Bybit ({results['environment']}):")
        print(f"   Testes: {results['successful_tests']}/{results['total_tests']} ({results['success_rate']:.1f}%)")

        print(f"\n📋 Detalhes dos Testes:")
        for test_name, test_result in results['results'].items():
            status = "✅" if test_result.get('success', False) else "❌"
            endpoint = test_result.get('endpoint', test_name)
            error = test_result.get('error', '')

            print(f"   {status} {test_name.replace('_', ' ').title()} ({endpoint})")
            if error and not test_result.get('success', False):
                print(f"      Error: {error}")

            # Show some specific data
            if test_name == 'ticker_btcusdt' and test_result.get('success'):
                print(f"      BTC/USDT: ${test_result.get('last_price', 'N/A')}")

            elif test_name == 'wallet_balance' and test_result.get('success'):
                coins_with_balance = test_result.get('coins_with_balance', 0)
                total_equity = test_result.get('total_equity', 'N/A')
                print(f"      Coins with balance: {coins_with_balance}, Total equity: ${total_equity}")

            elif test_name == 'open_orders' and test_result.get('success'):
                order_count = test_result.get('open_orders_count', 0)
                print(f"      Open orders: {order_count}")

        # Final assessment
        if results['overall_success']:
            print(f"\n🎉 Bybit API funcionando corretamente!")
        else:
            print(f"\n⚠️  Alguns endpoints Bybit precisam de atenção")

        return results['overall_success']

    except Exception as e:
        print(f"❌ Erro geral: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)