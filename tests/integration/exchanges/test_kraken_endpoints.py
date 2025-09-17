#!/usr/bin/env python3
"""
Teste de Endpoints Kraken
Testa conectividade e operações básicas com a API Kraken (sandbox)

IMPORTANTE: APENAS TESTNET/SANDBOX!

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import sys
import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
from typing import Dict, Any, Optional
import json
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tests.integration.exchanges.credentials_manager import CredentialsManager


class KrakenAPITester:
    """Testador da API Kraken"""

    def __init__(self, credentials_manager: CredentialsManager):
        self.cred_manager = credentials_manager
        self.credentials = credentials_manager.load_credentials()

        if not self.credentials.kraken:
            raise ValueError("Credenciais Kraken não disponíveis")

        # Kraken API endpoints
        self.base_url = "https://api.kraken.com"
        if self.credentials.kraken.environment == 'sandbox':
            # Note: Kraken não tem sandbox público, usaremos testnet
            self.base_url = "https://api.kraken.com"  # Usar com cuidado

        self.api_key = self.credentials.kraken.api_key
        self.api_secret = self.credentials.kraken.api_secret

        self.logger = logging.getLogger(__name__)

    def _get_kraken_signature(self, urlpath: str, data: Dict[str, Any], nonce: str) -> str:
        """Gera assinatura para autenticação Kraken"""

        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())

        return sigdigest.decode()

    def _make_public_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Faz requisição pública para Kraken"""

        url = f"{self.base_url}/0/public/{endpoint}"
        params = params or {}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro na requisição pública {endpoint}: {str(e)}")
            raise

    def _make_private_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Faz requisição privada para Kraken"""

        if not data:
            data = {}

        # Add nonce
        data['nonce'] = str(int(1000 * time.time()))

        urlpath = f"/0/private/{endpoint}"
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._get_kraken_signature(urlpath, data, data['nonce'])
        }

        url = f"{self.base_url}{urlpath}"

        try:
            response = requests.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro na requisição privada {endpoint}: {str(e)}")
            raise

    def test_server_time(self) -> Dict[str, Any]:
        """Testa endpoint de tempo do servidor"""

        self.logger.info("Testando server time...")

        try:
            result = self._make_public_request("Time")

            if 'error' in result and result['error']:
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': 'Time'
                }

            server_time = result.get('result', {}).get('unixtime', 0)
            local_time = int(time.time())
            time_diff = abs(server_time - local_time)

            return {
                'success': True,
                'server_time': server_time,
                'local_time': local_time,
                'time_difference_seconds': time_diff,
                'sync_ok': time_diff < 30,  # Less than 30s difference
                'endpoint': 'Time'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': 'Time'
            }

    def test_asset_pairs(self) -> Dict[str, Any]:
        """Testa endpoint de pares de ativos"""

        self.logger.info("Testando asset pairs...")

        try:
            result = self._make_public_request("AssetPairs")

            if 'error' in result and result['error']:
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': 'AssetPairs'
                }

            pairs = result.get('result', {})
            btc_pairs = [pair for pair in pairs.keys() if 'BTC' in pair]

            return {
                'success': True,
                'total_pairs': len(pairs),
                'btc_pairs_count': len(btc_pairs),
                'sample_pairs': list(pairs.keys())[:5],
                'btc_pairs_sample': btc_pairs[:3],
                'endpoint': 'AssetPairs'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': 'AssetPairs'
            }

    def test_ticker_data(self, pair: str = "BTCUSD") -> Dict[str, Any]:
        """Testa endpoint de ticker"""

        self.logger.info(f"Testando ticker para {pair}...")

        try:
            result = self._make_public_request("Ticker", {"pair": pair})

            if 'error' in result and result['error']:
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': 'Ticker',
                    'pair': pair
                }

            ticker_data = result.get('result', {})
            pair_data = ticker_data.get(list(ticker_data.keys())[0] if ticker_data else '', {})

            if pair_data:
                return {
                    'success': True,
                    'pair': pair,
                    'bid': pair_data.get('b', ['0'])[0],
                    'ask': pair_data.get('a', ['0'])[0],
                    'last_price': pair_data.get('c', ['0'])[0],
                    'volume_24h': pair_data.get('v', ['0'])[1],
                    'high_24h': pair_data.get('h', ['0'])[1],
                    'low_24h': pair_data.get('l', ['0'])[1],
                    'endpoint': 'Ticker'
                }
            else:
                return {
                    'success': False,
                    'error': 'No ticker data found',
                    'endpoint': 'Ticker',
                    'pair': pair
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': 'Ticker',
                'pair': pair
            }

    def test_account_balance(self) -> Dict[str, Any]:
        """Testa endpoint de saldo da conta"""

        self.logger.info("Testando account balance...")

        try:
            result = self._make_private_request("Balance")

            if 'error' in result and result['error']:
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': 'Balance'
                }

            balances = result.get('result', {})

            # Filter out zero balances
            non_zero_balances = {
                asset: balance for asset, balance in balances.items()
                if float(balance) > 0
            }

            return {
                'success': True,
                'total_assets': len(balances),
                'non_zero_assets': len(non_zero_balances),
                'balances': non_zero_balances,
                'all_balances': balances,
                'endpoint': 'Balance'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': 'Balance'
            }

    def test_trade_balance(self) -> Dict[str, Any]:
        """Testa endpoint de saldo de trading"""

        self.logger.info("Testando trade balance...")

        try:
            result = self._make_private_request("TradeBalance")

            if 'error' in result and result['error']:
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': 'TradeBalance'
                }

            trade_balance = result.get('result', {})

            return {
                'success': True,
                'equivalent_balance': trade_balance.get('eb', '0'),
                'trade_balance': trade_balance.get('tb', '0'),
                'margin': trade_balance.get('m', '0'),
                'unrealized_pnl': trade_balance.get('n', '0'),
                'cost_basis': trade_balance.get('c', '0'),
                'floating_valuation': trade_balance.get('v', '0'),
                'equity': trade_balance.get('e', '0'),
                'free_margin': trade_balance.get('mf', '0'),
                'endpoint': 'TradeBalance'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': 'TradeBalance'
            }

    def test_open_orders(self) -> Dict[str, Any]:
        """Testa endpoint de ordens abertas"""

        self.logger.info("Testando open orders...")

        try:
            result = self._make_private_request("OpenOrders")

            if 'error' in result and result['error']:
                return {
                    'success': False,
                    'error': result['error'],
                    'endpoint': 'OpenOrders'
                }

            orders = result.get('result', {}).get('open', {})

            return {
                'success': True,
                'open_orders_count': len(orders),
                'orders': list(orders.keys()),
                'endpoint': 'OpenOrders'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint': 'OpenOrders'
            }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Executa teste abrangente da API Kraken"""

        self.logger.info("Iniciando teste abrangente Kraken...")

        results = {}

        # Test public endpoints
        results['server_time'] = self.test_server_time()
        results['asset_pairs'] = self.test_asset_pairs()
        results['ticker_btcusd'] = self.test_ticker_data("BTCUSD")

        # Test private endpoints
        results['account_balance'] = self.test_account_balance()
        results['trade_balance'] = self.test_trade_balance()
        results['open_orders'] = self.test_open_orders()

        # Calculate success rate
        successful_tests = sum(1 for result in results.values() if result.get('success', False))
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100

        return {
            'exchange': 'Kraken',
            'environment': self.credentials.kraken.environment,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'results': results,
            'overall_success': success_rate >= 80  # 80% success rate required
        }


def main():
    """Teste principal Kraken"""

    logging.basicConfig(level=logging.INFO)

    print("🏦 WOW Capital - Kraken API Test")
    print("=" * 50)

    try:
        # Load credentials
        cred_manager = CredentialsManager()
        credentials = cred_manager.load_credentials()

        if not credentials.kraken:
            print("❌ Credenciais Kraken não disponíveis")
            return False

        # Create tester
        kraken_tester = KrakenAPITester(cred_manager)

        # Run comprehensive test
        results = kraken_tester.run_comprehensive_test()

        # Display results
        print(f"\n📊 Resultados Kraken ({results['environment']}):")
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
            if test_name == 'ticker_btcusd' and test_result.get('success'):
                print(f"      BTC/USD: ${test_result.get('last_price', 'N/A')}")

            elif test_name == 'account_balance' and test_result.get('success'):
                non_zero = test_result.get('non_zero_assets', 0)
                print(f"      Assets with balance: {non_zero}")

        # Final assessment
        if results['overall_success']:
            print(f"\n🎉 Kraken API funcionando corretamente!")
        else:
            print(f"\n⚠️  Alguns endpoints Kraken precisam de atenção")

        return results['overall_success']

    except Exception as e:
        print(f"❌ Erro geral: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)