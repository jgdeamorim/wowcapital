#!/usr/bin/env python3
"""
Teste de Endpoints Kraken (Spot API)
Verifica conectividade e operações básicas com a API Kraken Spot.

Autor: WOW Capital Trading System
Data: 2024-09-22 (atualizado para Spot API)
"""

import sys
import os
import time
import hmac
import hashlib
import base64
import requests
from typing import Dict, Any, Optional
import logging
import argparse
from urllib.parse import urlencode

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tests.integration.exchanges.credentials_manager import CredentialsManager

SPOT_LIVE_BASE = "https://api.kraken.com"
# A API de Spot da Kraken não tem um ambiente de sandbox público e persistente.
# Os testes devem ser feitos com cuidado contra a produção.

class KrakenAPITester:
    """Testador da API Spot da Kraken"""

    def __init__(self, credentials_manager: CredentialsManager):
        self.cred_manager = credentials_manager
        self.credentials = self.cred_manager.load_credentials()

        if not self.credentials.kraken:
            raise ValueError("Credenciais Kraken não disponíveis")

        self.base_url = SPOT_LIVE_BASE
        self.api_key = self.credentials.kraken.api_key
        self.api_secret = self.credentials.kraken.api_secret
        try:
            self._api_secret_bytes = base64.b64decode(self.api_secret)
        except Exception as exc:
            raise ValueError("KRAKEN_PRIVATE_KEY deve estar em Base64") from exc

        self.logger = logging.getLogger(__name__)

    def _make_public_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get('error'):
                raise RuntimeError(f"API Error: {result['error']}")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro na requisição pública {endpoint}: {str(e)}")
            raise

    def get_kraken_signature(self, urlpath, data):
        """Gera a assinatura API-Sign para a API Spot da Kraken."""
        postdata = urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(self._api_secret_bytes, message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _make_private_request(self, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Faz requisição privada para a API Spot da Kraken."""
        payload = payload or {}
        payload['nonce'] = str(int(time.time() * 1000))
        url = f"{self.base_url}{endpoint}"

        headers = {
            'API-Key': self.api_key,
            'API-Sign': self.get_kraken_signature(endpoint, payload)
        }

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get('error') and result['error']:
                raise RuntimeError(f"API Error: {result['error']}")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro na requisição privada {endpoint}: {str(e)}")
            if e.response:
                self.logger.error(f"Response: {e.response.text}")
            raise

    def test_server_time(self) -> Dict[str, Any]:
        self.logger.info("Testando server time...")
        try:
            result = self._make_public_request("/0/public/Time")
            return {'success': True, 'server_time': result.get('result', {}).get('rfc1123'), 'endpoint': '/0/public/Time'}
        except Exception as e:
            return {'success': False, 'error': str(e), 'endpoint': '/0/public/Time'}

    def test_assets(self) -> Dict[str, Any]:
        self.logger.info("Testando assets...")
        try:
            result = self._make_public_request("/0/public/Assets")
            return {'success': True, 'asset_count': len(result.get('result', {})), 'endpoint': '/0/public/Assets'}
        except Exception as e:
            return {'success': False, 'error': str(e), 'endpoint': '/0/public/Assets'}

    def test_ticker_data(self, pair: str = "XBTUSD") -> Dict[str, Any]:
        self.logger.info(f"Testando ticker para {pair}...")
        try:
            result = self._make_public_request("/0/public/Ticker", {"pair": pair})
            return {'success': True, 'ticker': result.get('result', {}).get(pair), 'endpoint': '/0/public/Ticker'}
        except Exception as e:
            return {'success': False, 'error': str(e), 'endpoint': '/0/public/Ticker'}

    def test_account_balance(self) -> Dict[str, Any]:
        self.logger.info("Testando account balance...")
        try:
            result = self._make_private_request("/0/private/Balance")
            return {'success': True, 'balance_count': len(result.get('result', {})), 'endpoint': '/0/private/Balance'}
        except Exception as e:
            return {'success': False, 'error': str(e), 'endpoint': '/0/private/Balance'}

    def test_open_orders(self) -> Dict[str, Any]:
        self.logger.info("Testando open orders...")
        try:
            result = self._make_private_request("/0/private/OpenOrders", {"trades": True})
            return {'success': True, 'open_orders_count': len(result.get('result', {}).get('open', {})), 'endpoint': '/0/private/OpenOrders'}
        except Exception as e:
            return {'success': False, 'error': str(e), 'endpoint': '/0/private/OpenOrders'}

    def run_comprehensive_test(self) -> Dict[str, Any]:
        self.logger.info("Iniciando teste abrangente Kraken Spot API...")
        results = {}
        results['server_time'] = self.test_server_time()
        results['assets'] = self.test_assets()
        results['ticker_xbtusd'] = self.test_ticker_data()
        results['account_balance'] = self.test_account_balance()
        results['open_orders'] = self.test_open_orders()
        
        successful_tests = sum(1 for r in results.values() if r.get('success'))
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests else 0

        return {
            'exchange': 'Kraken Spot',
            'environment': self.credentials.kraken.environment,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'results': results,
            'overall_success': success_rate > 99
        }

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Kraken Spot API integration test")
    parser.add_argument("--env-file", dest="env_file", help="Arquivo .env com credenciais", required=True)
    args = parser.parse_args()

    print("🏦 WOW Capital - Kraken Spot API Test")
    print("=" * 50)

    try:
        cred_manager = CredentialsManager(env_file=args.env_file)
        tester = KrakenAPITester(cred_manager)
        results = tester.run_comprehensive_test()

        print(f"\n📊 Resultados Kraken Spot ({results['environment']}):")
        print(f"   Testes: {results['successful_tests']}/{results['total_tests']} ({results['success_rate']:.1f}%)")

        print("\n📋 Detalhes dos Testes:")
        for test_name, test_result in results['results'].items():
            status = "✅" if test_result.get('success') else "❌"
            endpoint = test_result.get('endpoint', 'N/A')
            error = test_result.get('error', '')

            print(f"   {status} {test_name.replace('_', ' ').title()} ({endpoint})")
            if not test_result.get('success'):
                print(f"      Error: {error}")

        if results['overall_success']:
            print("\n🎉 Kraken Spot API funcionando corretamente!")
        else:
            print("\n⚠️  Alguns endpoints Kraken Spot precisam de atenção")

        return results['overall_success']

    except Exception as e:
        print(f"❌ Erro geral: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
