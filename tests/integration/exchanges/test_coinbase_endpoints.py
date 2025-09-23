#!/usr/bin/env python3
"""
Teste de Endpoints Coinbase
Verifica conectividade e operações básicas com a API Coinbase Advanced Trade (sandbox/demo)

IMPORTANTE: Apenas uso seguro (sandbox/demo). Credenciais reais devem possuir limites rígidos.

Autor: WOW Capital Trading System
Data: 2024-09-17
"""

import sys
import os
import json
import time
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
import base64

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tests.integration.exchanges.credentials_manager import CredentialsManager


class CoinbaseAPITester:
    """Testador da API Coinbase Advanced Trade."""

    def __init__(self, credentials_manager: CredentialsManager):
        self.logger = logging.getLogger(__name__)
        self.cred_manager = credentials_manager
        creds = credentials_manager.load_credentials()

        if not creds.coinbase:
            raise ValueError("Credenciais Coinbase não disponíveis")

        self.credentials = creds.coinbase
        self.base_url = (self.credentials.base_url or "https://api.coinbase.com/api/v3").rstrip('/')
        parsed = urlparse(self.base_url)
        self.api_prefix = parsed.path.rstrip('/') or ""

        self.api_key = self.credentials.api_key
        self.private_key = None
        if self.credentials.private_key:
            try:
                self.private_key = serialization.load_pem_private_key(self.credentials.private_key.encode(), password=None)
            except Exception as exc:
                self.logger.error(f"Erro carregando chave privada Coinbase: {exc}")
                self.private_key = None

        if not self.private_key:
            self.logger.warning("Chave privada Coinbase ausente ou inválida - endpoints assinados serão marcados como falha")

    # -------- Assinatura / Requests -------- #

    def _path_for_signature(self, path: str) -> str:
        if self.api_prefix and not path.startswith(self.api_prefix):
            return f"{self.api_prefix}{path}" if path.startswith('/') else f"{self.api_prefix}/{path}"
        return path

    def _sign(self, method: str, path: str, body: str, timestamp: str) -> str:
        if not self.private_key:
            raise RuntimeError("Chave privada Coinbase não carregada")
        message = timestamp + method.upper() + self._path_for_signature(path) + body
        signature = self.private_key.sign(message.encode(), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(signature)
        size = 32
        return base64.b64encode(r.to_bytes(size, 'big') + s.to_bytes(size, 'big')).decode()

    def _request(self, method: str, path: str, *, body: Optional[Dict[str, Any]] = None, signed: bool = False) -> requests.Response:
        url = self.base_url.rstrip('/') + path
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        headers = {"Content-Type": "application/json"}
        if signed:
            if not self.private_key:
                raise RuntimeError("Credenciais Coinbase incompletas - faltando chave privada")
            timestamp = str(int(time.time()))
            headers.update({
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-SIGNATURE": self._sign(method, path, body_str, timestamp),
            })
        response = requests.request(method.upper(), url, headers=headers, data=body_str or None, timeout=10)
        response.raise_for_status()
        return response

    # -------- Testes públicos -------- #

    def test_products(self) -> Dict[str, Any]:
        self.logger.info("Testando lista de produtos (requere assinatura)...")
        if not self.private_key:
            return {
                'success': False,
                'endpoint': '/brokerage/products',
                'error': 'missing_private_key'
            }
        try:
            resp = self._request("GET", "/brokerage/products", signed=True)
            data = resp.json()
            count = len(data.get('products', []))
            sample = [prod.get('product_id') for prod in data.get('products', [])[:3]]
            return {
                'success': True,
                'endpoint': '/brokerage/products',
                'products_found': count,
                'sample': sample
            }
        except Exception as exc:
            return {
                'success': False,
                'endpoint': '/brokerage/products',
                'error': str(exc)
            }

    def test_product_ticker(self, product_id: str = "BTC-USD") -> Dict[str, Any]:
        path = f"/brokerage/products/{product_id}"
        self.logger.info("Testando ticker %s (requere assinatura)", product_id)
        if not self.private_key:
            return {
                'success': False,
                'endpoint': path,
                'error': 'missing_private_key'
            }
        try:
            resp = self._request("GET", path, signed=True)
            data = resp.json()
            price = data.get('price') or data.get('market_data', {}).get('price')
            return {
                'success': True,
                'endpoint': path,
                'price': price,
                'product': product_id
            }
        except Exception as exc:
            return {
                'success': False,
                'endpoint': path,
                'error': str(exc)
            }

    def test_public_exchange_rates(self) -> Dict[str, Any]:
        url = "https://api.coinbase.com/v2/exchange-rates?currency=BTC"
        self.logger.info("Testando endpoint público de taxas de câmbio...")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rates = data.get('data', {}).get('rates', {})
            sample = {k: rates[k] for k in list(rates.keys())[:3]} if rates else {}
            return {
                'success': True,
                'endpoint': url,
                'sample_rates': sample
            }
        except Exception as exc:
            return {
                'success': False,
                'endpoint': url,
                'error': str(exc)
            }

    # -------- Testes privados -------- #

    def test_account_balances(self) -> Dict[str, Any]:
        path = "/brokerage/accounts"
        self.logger.info("Testando contas Coinbase...")
        if not self.private_key:
            return {
                'success': False,
                'endpoint': path,
                'error': 'missing_private_key'
            }
        try:
            resp = self._request("GET", path, signed=True)
            data = resp.json()
            accounts = data.get('accounts', [])
            currencies = [acct.get('currency') for acct in accounts[:5]]
            return {
                'success': True,
                'endpoint': path,
                'accounts_found': len(accounts),
                'currencies': currencies
            }
        except Exception as exc:
            return {
                'success': False,
                'endpoint': path,
                'error': str(exc)
            }

    def test_orders_history(self) -> Dict[str, Any]:
        path = "/brokerage/orders/historical/batch"
        self.logger.info("Testando histórico de ordens Coinbase...")
        if not self.private_key:
            return {
                'success': False,
                'endpoint': path,
                'error': 'missing_private_key'
            }
        try:
            resp = self._request("GET", path, signed=True)
            data = resp.json()
            orders = data.get('orders', [])
            return {
                'success': True,
                'endpoint': path,
                'orders_found': len(orders)
            }
        except Exception as exc:
            return {
                'success': False,
                'endpoint': path,
                'error': str(exc)
            }

    # -------- Execução -------- #

    def run(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        results['exchange_rates'] = self.test_public_exchange_rates()
        results['products'] = self.test_products()
        results['product_ticker'] = self.test_product_ticker()
        results['account_balances'] = self.test_account_balances()
        results['orders_history'] = self.test_orders_history()
        return results


def _print_result(name: str, result: Dict[str, Any]):
    status = "✅" if result.get('success') else "❌"
    details = result.get('error')
    if result.get('success'):
        summary_parts = []
        for key in ('products_found', 'price', 'accounts_found', 'orders_found'):
            if key in result and result[key] is not None:
                summary_parts.append(f"{key}: {result[key]}")
        if 'sample_rates' in result and result['sample_rates']:
            summary_parts.append(f"sample_rates: {list(result['sample_rates'].items())[:3]}")
        details = ", ".join(summary_parts) if summary_parts else "OK"
    print(f"   {status} {name}: {details}")


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    print("🏦 WOW Capital - Coinbase Advanced Trade API Test")
    print("=" * 60)

    cred_manager = CredentialsManager()

    try:
        tester = CoinbaseAPITester(cred_manager)
    except Exception as exc:
        print(f"❌ Não foi possível inicializar tester Coinbase: {exc}")
        return 2

    results = tester.run()

    print("\n📊 Resultados Coinbase:")
    for name, result in results.items():
        _print_result(name, result)

    success_count = sum(1 for data in results.values() if data.get('success'))
    total_tests = len(results)

    print("\n============================================================")
    print(f"✅ Testes Aprovados: {success_count}/{total_tests} ({(success_count/total_tests)*100:.1f}%)")
    if success_count == total_tests:
        print("🎉 Coinbase API totalmente funcional!")
    elif success_count >= total_tests / 2:
        print("⚠️ Coinbase API parcialmente funcional")
    else:
        print("🚨 Problemas detectados na API Coinbase")

    return 0 if success_count == total_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())
