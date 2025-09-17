#!/usr/bin/env python3
"""
Bybit Focus Test - Teste específico para resolver problemas Bybit
Testa conectividade focada em endpoints públicos para operações demo

Autor: WOW Capital AI System
Data: 2024-09-16
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tests.integration.exchanges.credentials_manager import CredentialsManager
from tests.integration.exchanges.test_bybit_endpoints import BybitAPITester

async def test_bybit_public_only():
    """Testa apenas endpoints públicos do Bybit para operações demo"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("🏦 BYBIT DEMO TEST - Endpoints Públicos")
    print("=" * 60)

    try:
        # Load credentials
        cred_manager = CredentialsManager()
        credentials = cred_manager.load_credentials()

        if not credentials.bybit:
            print("❌ Credenciais Bybit não disponíveis")
            return False

        # Create tester
        bybit_tester = BybitAPITester(cred_manager)

        # Test public endpoints only (sufficient for demo)
        public_tests = {
            "server_time": bybit_tester.test_server_time,
            "instruments_info": bybit_tester.test_instruments_info,
            "ticker_btcusdt": lambda: bybit_tester.test_ticker_data("BTCUSDT")
        }

        results = {}
        for test_name, test_func in public_tests.items():
            try:
                result = test_func()
                results[test_name] = result
                status = "✅" if result.get('success', False) else "❌"
                print(f"{status} {test_name}: {result.get('success', False)}")

                if test_name == "ticker_btcusdt" and result.get('success'):
                    print(f"   💰 BTC/USDT: ${result.get('last_price', 'N/A')}")

            except Exception as e:
                print(f"❌ {test_name}: Erro - {str(e)}")
                results[test_name] = {"success": False, "error": str(e)}

        # Calculate success rate (public endpoints only)
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        success_rate = (successful / total) * 100

        print(f"\n📊 RESULTADOS:")
        print(f"   ✅ Testes públicos: {successful}/{total} ({success_rate:.1f}%)")
        print(f"   🎯 Status para demo: {'✅ OPERACIONAL' if success_rate >= 100 else '⚠️ LIMITADO'}")

        # For demo operations, public endpoints are sufficient
        demo_ready = success_rate >= 100  # Need all public endpoints working

        if demo_ready:
            print(f"\n🎉 BYBIT PRONTO PARA OPERAÇÕES DEMO!")
            print(f"   📊 Dados de mercado: ✅")
            print(f"   💰 Preços em tempo real: ✅")
            print(f"   📈 Informações de instrumentos: ✅")
            print(f"   ⚠️ Nota: Endpoints privados não necessários para demo")
        else:
            print(f"\n⚠️ BYBIT LIMITADO - Alguns endpoints falharam")

        return demo_ready

    except Exception as e:
        print(f"❌ Erro geral: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bybit_public_only())
    sys.exit(0 if success else 1)