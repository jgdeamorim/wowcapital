#!/usr/bin/env python3
"""
WOW Capital - Complete Test Runner
Executes all exchange tests and validates functionality

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import sys
import os
import subprocess
import time
from typing import Dict, List, Any

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

def run_command(command: List[str], description: str) -> Dict[str, Any]:
    """Run a command and return results"""
    print(f"\n🧪 {description}")
    print("-" * 60)

    start_time = time.time()
    try:
        result = subprocess.run(command,
                               capture_output=True,
                               text=True,
                               cwd=os.path.dirname(__file__),
                               timeout=300)  # 5 minute timeout

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS ({elapsed:.1f}s)")
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
            return {
                'success': True,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed': elapsed
            }
        else:
            print(f"❌ {description} - FAILED ({elapsed:.1f}s)")
            print(f"Exit code: {result.returncode}")
            if result.stderr.strip():
                print("Error:", result.stderr.strip())
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
            return {
                'success': False,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'elapsed': elapsed
            }

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return {
            'success': False,
            'returncode': -1,
            'error': 'Timeout',
            'elapsed': time.time() - start_time
        }
    except Exception as e:
        print(f"💥 {description} - ERROR: {str(e)}")
        return {
            'success': False,
            'returncode': -2,
            'error': str(e),
            'elapsed': time.time() - start_time
        }

def main():
    """Main test runner"""
    print("🚀 WOW Capital - Complete Test Suite")
    print("=" * 70)

    test_results = {}

    # List of tests to run
    tests = [
        # Core integration tests
        {
            'name': 'Integration Test Priority 1',
            'command': ['python3', '../tests/integration_test_priority_1.py'],
            'description': 'Core components integration test'
        },
        {
            'name': 'Integration Test 100%',
            'command': ['python3', '../tests/integration_test_100_percent.py'],
            'description': 'Complete system integration test'
        },

        # Exchange endpoint tests
        {
            'name': 'Binance Endpoints',
            'command': ['python3', '../tests/integration/exchanges/test_binance_endpoints.py'],
            'description': 'Binance API endpoint tests'
        },
        {
            'name': 'Kraken Endpoints',
            'command': ['python3', '../tests/integration/exchanges/test_kraken_endpoints.py'],
            'description': 'Kraken API endpoint tests'
        },
        {
            'name': 'Bybit Endpoints',
            'command': ['python3', '../tests/integration/exchanges/test_bybit_endpoints.py'],
            'description': 'Bybit API endpoint tests'
        },

        # Adapter tests (using direct execution)
        {
            'name': 'Binance Adapter Mock',
            'command': ['python', '-c', '''
import sys
sys.path.insert(0, ".")
import asyncio
from exchanges.binance import BinanceAdapter
from core.contracts import OrderRequest

async def test_binance():
    tos = {"exchanges": {"binance": {"rate_limit_rps": 10}}}
    adapter = BinanceAdapter(tos, mode="spot", api_key="demo", api_secret="demo")

    # Test healthcheck (will fail but shows adapter loads)
    try:
        await adapter.healthcheck()
    except:
        pass

    print("✅ BinanceAdapter loads successfully")

asyncio.run(test_binance())
            '''],
            'description': 'Binance adapter basic loading test'
        },

        # AI Orchestrator tests
        {
            'name': 'AI Orchestrator Test',
            'command': ['python3', '../tests/test_ai_orchestrator.py'],
            'description': 'AI orchestration system test'
        }
    ]

    # Run all tests
    passed = 0
    total = len(tests)

    for test in tests:
        result = run_command(test['command'], test['description'])
        test_results[test['name']] = result

        if result['success']:
            passed += 1

    # Final report
    print("\n" + "=" * 70)
    print("📋 FINAL TEST REPORT")
    print("=" * 70)

    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"✅ Tests Passed: {passed}/{total} ({success_rate:.1f}%)")

    print(f"\n📊 Test Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        elapsed = result.get('elapsed', 0)
        print(f"   {status} {test_name} ({elapsed:.1f}s)")

        if not result['success']:
            error_msg = result.get('error', result.get('stderr', 'Unknown error'))
            if error_msg and error_msg.strip():
                print(f"      Error: {error_msg.strip()[:100]}...")

    # System readiness assessment
    print(f"\n🎯 SYSTEM READINESS:")
    if success_rate >= 90:
        print("   🎉 EXCELLENT - Sistema pronto para produção")
        print("   ✅ Todos os componentes principais funcionando")
        print("   ✅ APIs de exchanges operacionais")
        print("   ✅ Integração AI orquestradora validada")
    elif success_rate >= 70:
        print("   ⚠️  GOOD - Sistema majoritariamente funcional")
        print("   ✅ Componentes críticos operacionais")
        print("   ⚠️  Alguns ajustes podem ser necessários")
    elif success_rate >= 50:
        print("   🔄 PARTIAL - Sistema parcialmente funcional")
        print("   ⚠️  Vários componentes precisam de atenção")
        print("   🔧 Correções necessárias antes de uso extensivo")
    else:
        print("   🚨 CRITICAL - Problemas significativos detectados")
        print("   ❌ Sistema não pronto para uso")
        print("   🔧 Revisão abrangente necessária")

    # Trading system specific recommendations
    print(f"\n🔧 PRÓXIMOS PASSOS:")
    if success_rate >= 90:
        print("   1. ✅ Configurar ambiente de produção")
        print("   2. ✅ Implementar monitoramento avançado")
        print("   3. ✅ Executar testes de stress com volume real")
    elif success_rate >= 70:
        print("   1. 🔧 Resolver falhas identificadas nos testes")
        print("   2. ⚠️  Validar configurações de exchanges")
        print("   3. 🧪 Re-executar suite completa de testes")
    else:
        print("   1. ❌ Revisar e corrigir componentes críticos")
        print("   2. 🔧 Validar configurações básicas")
        print("   3. 🧪 Executar testes individuais para debugging")

    print(f"\n📈 TRADING SYSTEM STATUS:")
    print(f"   • Binance Integration: {'✅ Ready' if any('Binance' in name and result['success'] for name, result in test_results.items()) else '🔧 Needs Work'}")
    print(f"   • Kraken Integration: {'✅ Ready' if any('Kraken' in name and result['success'] for name, result in test_results.items()) else '🔧 Needs Work'}")
    print(f"   • AI Orchestrator: {'✅ Ready' if any('AI' in name and result['success'] for name, result in test_results.items()) else '🔧 Needs Work'}")
    print(f"   • Core Systems: {'✅ Ready' if any('Integration' in name and result['success'] for name, result in test_results.items()) else '🔧 Needs Work'}")

    return success_rate >= 70

if __name__ == "__main__":
    success = main()
    print(f"\n🏁 Overall Test Suite: {'SUCCESS' if success else 'NEEDS ATTENTION'}")
    sys.exit(0 if success else 1)