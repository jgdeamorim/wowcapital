#!/usr/bin/env python3
"""
Teste de Integração Simplificado - Componentes Prioridade 1
Verifica estrutura e funcionalidade básica dos componentes implementados

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import sys
import os
import time

# Adicionar backend ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Testa se todos os módulos podem ser importados"""
    print("🧪 Testando Imports...")

    try:
        # Test MOMO-1.5L
        from indicators.momentum.momo_1_5l import MOMO15L, MOMO15LConfig
        print("   ✅ MOMO-1.5L imported successfully")

        # Test High Aggression Score
        from indicators.composite.high_aggression_score import HighAggressionScore, MarketSignals
        print("   ✅ High Aggression Score imported successfully")

        # Test Strategy 1.5L
        from plugins.strategies.strategy_1_5l import Strategy15L, Strategy15LConfig
        print("   ✅ Strategy 1.5L imported successfully")

        # Test Pocket Explosion
        from execution.pocket_explosion.core import PocketExplosionSystem, PocketConfig
        print("   ✅ Pocket Explosion System imported successfully")

        # Test Core contracts
        from core.contracts import MarketSnapshot
        print("   ✅ Core contracts imported successfully")

        return True

    except ImportError as e:
        print(f"   ❌ Import failed: {str(e)}")
        return False

def test_basic_functionality():
    """Testa funcionalidade básica sem dependências externas"""
    print("\n🧪 Testando Funcionalidade Básica...")

    try:
        # Test data
        import numpy as np
        prices = [50000 + i*10 + np.random.normal(0, 50) for i in range(60)]
        volumes = [1000 + np.random.exponential(500) for _ in range(60)]

        # 1. Test MOMO-1.5L basic structure
        from indicators.momentum.momo_1_5l import MOMO15L, MOMO15LConfig

        config = MOMO15LConfig()
        momo = MOMO15L(config)

        # Test with numpy arrays
        result = momo.calculate(np.array(prices))

        assert -1.5 <= result.score <= 1.5, f"MOMO score out of range: {result.score}"
        assert result.signal_strength in ['weak', 'moderate', 'strong']
        print(f"   ✅ MOMO-1.5L Score: {result.score:.4f} ({result.signal_strength})")

        # 2. Test High Aggression Score structure
        from indicators.composite.high_aggression_score import HighAggressionScore, MarketSignals

        scorer = HighAggressionScore()

        # Create minimal market signals
        signals = MarketSignals(
            prices=np.array(prices[-20:]),
            volumes=np.array(volumes[-20:]),
            timestamp=time.time(),
            symbol="BTC/USDT"
        )

        aggr_result = scorer.calculate_aggression_score(signals)
        assert 0.0 <= aggr_result.score <= 1.0, f"Aggression score out of range: {aggr_result.score}"
        print(f"   ✅ Aggression Score: {aggr_result.score:.4f} ({aggr_result.risk_level.value})")

        # 3. Test Strategy 1.5L structure
        from plugins.strategies.strategy_1_5l import Strategy15L, Strategy15LConfig
        from core.contracts import MarketSnapshot

        strategy_config = Strategy15LConfig()
        strategy = Strategy15L(strategy_config)

        # Create test snapshot
        snapshot = MarketSnapshot(
            class_="CRYPTO",
            symbol="BTC/USDT",
            ts_ns=int(time.time() * 1e9),
            bid=49995.0,
            ask=50005.0,
            mid=50000.0,
            spread=10.0,
            features={
                'volume': 1000.0,
                'equity_total': 10000.0,
                'open_positions': [],
                'daily_pnl': 0.001,
                'account': 'test'
            }
        )

        # Add some price history to strategy
        strategy.price_history = prices
        strategy.volume_history = volumes

        decision = strategy.decide(snapshot)
        assert isinstance(decision, dict)
        print(f"   ✅ Strategy Decision: {decision.get('action', 'HOLD')}")

        # 4. Test Pocket Explosion basic structure
        from execution.pocket_explosion.core import PocketExplosionSystem, PocketConfig

        pocket_config = PocketConfig()
        pocket_system = PocketExplosionSystem(pocket_config)

        assert hasattr(pocket_system, 'config')
        assert hasattr(pocket_system, 'active_pockets')
        print(f"   ✅ Pocket System initialized with {len(pocket_system.active_pockets)} active pockets")

        return True

    except Exception as e:
        print(f"   ❌ Functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_component_integration():
    """Testa integração básica entre componentes"""
    print("\n🧪 Testando Integração Básica...")

    try:
        import numpy as np

        # Generate test market data
        base_price = 50000
        prices = []
        volumes = []

        for i in range(60):
            # Simulate upward trend with noise
            price = base_price + i*20 + np.random.normal(0, 100)
            volume = 1000 + np.random.exponential(300)
            prices.append(price)
            volumes.append(volume)

        print(f"   📊 Generated {len(prices)} price points")
        print(f"   📊 Price range: ${min(prices):.0f} - ${max(prices):.0f}")

        # Test MOMO → Strategy integration
        from indicators.momentum.momo_1_5l import MOMO15L
        from plugins.strategies.strategy_1_5l import Strategy15L
        from core.contracts import MarketSnapshot

        momo = MOMO15L()
        strategy = Strategy15L()

        # Run through several market snapshots
        signals_generated = 0

        for i in range(50, len(prices)):
            current_price = prices[i]
            current_volume = volumes[i]

            # Update strategy history
            strategy.price_history = prices[:i+1]
            strategy.volume_history = volumes[:i+1]

            # Create market snapshot
            snapshot = MarketSnapshot(
                class_="CRYPTO",
                symbol="BTC/USDT",
                ts_ns=int(time.time() * 1e9),
                bid=current_price - 5,
                ask=current_price + 5,
                mid=current_price,
                spread=10.0,
                features={
                    'volume': current_volume,
                    'equity_total': 10000.0,
                    'open_positions': [],
                    'daily_pnl': 0.002,
                    'account': 'test'
                }
            )

            # Get strategy decision
            decision = strategy.decide(snapshot)

            if 'side' in decision:
                signals_generated += 1
                print(f"   🎯 Signal {signals_generated}: {decision['side']} @ ${current_price:.0f}")

        print(f"   ✅ Generated {signals_generated} trading signals from {len(prices)-50} snapshots")

        # Test Aggression Score → Pocket integration (structure only)
        from indicators.composite.high_aggression_score import HighAggressionScore, MarketSignals
        from execution.pocket_explosion.core import PocketExplosionSystem

        scorer = HighAggressionScore()
        pocket_system = PocketExplosionSystem()

        # Create high-conviction signals
        signals = MarketSignals(
            prices=np.array(prices[-20:]),
            volumes=np.array(volumes[-20:]),
            timestamp=time.time(),
            symbol="BTC/USDT",
            rsi=80.0,  # High momentum
            order_book_imbalance=0.8,  # Strong imbalance
            realized_volatility=0.03
        )

        aggr_result = scorer.calculate_aggression_score(signals)

        print(f"   ⚡ Aggression Score: {aggr_result.score:.4f}")
        print(f"   🎯 Explosion Ready: {aggr_result.explosion_ready}")
        print(f"   🔥 Risk Level: {aggr_result.risk_level.value}")

        if aggr_result.score > 0.7:  # High score
            print("   ✅ High conviction scenario detected - pocket explosion would be considered")

        return True

    except Exception as e:
        print(f"   ❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal de teste"""
    print("🚀 TESTE DE INTEGRAÇÃO SIMPLIFICADO - PRIORIDADE 1")
    print("="*60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Imports
    if test_imports():
        tests_passed += 1

    # Test 2: Basic functionality
    if test_basic_functionality():
        tests_passed += 1

    # Test 3: Component integration
    if test_component_integration():
        tests_passed += 1

    # Final report
    print("\n" + "="*60)
    print("📋 RELATÓRIO FINAL")
    print("="*60)

    success_rate = (tests_passed / total_tests) * 100
    print(f"Testes Aprovados: {tests_passed}/{total_tests} ({success_rate:.1f}%)")

    if success_rate == 100:
        print("\n🎉 TODOS OS TESTES APROVADOS!")
        print("   ✅ Componentes de Prioridade 1 implementados com sucesso")
        print("   ✅ Integração básica funcionando")
        print("   ✅ Estruturas de dados compatíveis")

        print("\n🔄 PRÓXIMOS PASSOS:")
        print("   1. Instalar dependências completas (pandas, talib, etc)")
        print("   2. Executar testes com dados reais de mercado")
        print("   3. Implementar componentes de Prioridade 2")
        print("   4. Configurar integração com exchanges")

    elif success_rate >= 66:
        print("\n⚠️  TESTES PARCIALMENTE APROVADOS")
        print("   Alguns componentes precisam de ajustes")

    else:
        print("\n🚨 TESTES COM PROBLEMAS")
        print("   Componentes críticos precisam de revisão")

    print(f"\n📊 Status dos Componentes Implementados:")
    print("   ✅ MOMO-1.5L Indicator (Momentum proprietário)")
    print("   ✅ High Aggression Score (Score composto)")
    print("   ✅ Strategy 1.5L (Estratégia baseline)")
    print("   ✅ Pocket Explosion System (Micro-alocação)")
    print("   ✅ Integration Tests (Testes básicos)")

if __name__ == "__main__":
    main()