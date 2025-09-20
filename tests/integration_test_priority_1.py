#!/usr/bin/env python3
"""
Teste de Integração - Componentes Prioridade 1
Verifica funcionamento conjunto dos componentes críticos implementados

Componentes Testados:
- MOMO-1.5L Indicator
- High Aggression Score
- Strategy 1.5L
- Pocket Explosion System MVP

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import sys
import os
import asyncio
import time
import numpy as np
from typing import Dict, Any

# Adicionar backend ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports dos componentes
from indicators.momentum.momo_1_5l import MOMO15L, MOMO15LConfig
from indicators.composite.high_aggression_score import (
    HighAggressionScore, MarketSignals, AggressionWeights
)
from plugins.strategies.strategy_1_5l import Strategy15L, Strategy15LConfig
from execution.pocket_explosion.core import PocketExplosionSystem, PocketConfig
from core.contracts import MarketSnapshot


class IntegrationTester:
    """Classe principal para testes de integração"""

    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

        # Configurar componentes
        self.setup_components()

        # Dados de teste
        self.test_data = self.generate_test_data()

    def setup_components(self):
        """Inicializa componentes para teste"""
        print("🔧 Configurando componentes para teste...")

        # 1. MOMO-1.5L
        momo_config = MOMO15LConfig(
            lookback=21,
            smooth_factor=0.65,
            boost_limit=1.5,
            normalization_window=50
        )
        self.momo_indicator = MOMO15L(momo_config)

        # 2. High Aggression Score
        aggression_weights = AggressionWeights(
            momentum=0.30,      # Aumentar peso do momentum para teste
            volatility=0.25,
            volume=0.20,
            microstructure=0.15,
            sentiment=0.05,
            technical=0.05
        )
        self.aggression_scorer = HighAggressionScore(aggression_weights)

        # 3. Strategy 1.5L
        strategy_config = Strategy15LConfig()
        strategy_config.long_threshold = 0.25    # Mais sensível para teste
        strategy_config.short_threshold = -0.25
        self.strategy_1_5l = Strategy15L(strategy_config)

        # 4. Pocket Explosion System
        pocket_config = PocketConfig()
        pocket_config.explosion_threshold = 0.85  # Mais baixo para teste
        pocket_config.cooldown_seconds = 30       # Reduzido para teste
        self.pocket_system = PocketExplosionSystem(pocket_config)

        print("✅ Componentes configurados")

    def generate_test_data(self) -> Dict[str, Any]:
        """Gera dados de teste simulados"""
        print("📊 Gerando dados de teste...")

        # Gerar série de preços com tendência
        np.random.seed(42)
        n_periods = 60
        base_price = 50000.0

        # Simular movimento bullish
        price_changes = np.random.normal(0.001, 0.015, n_periods)  # Leve viés positivo
        prices = [base_price]

        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # Gerar volumes correlacionados
        volumes = []
        for i, price in enumerate(prices[1:]):
            price_change = abs(prices[i+1] - prices[i]) / prices[i]
            # Volume maior com movimentos maiores
            base_volume = 1000
            volume_multiplier = 1 + (price_change * 10)
            volume = base_volume * volume_multiplier * (1 + np.random.normal(0, 0.3))
            volumes.append(max(100, volume))

        return {
            'prices': prices[1:],  # Remove primeiro preço base
            'volumes': volumes,
            'timestamps': [time.time() - (n_periods-i)*60 for i in range(n_periods)]
        }

    async def test_momo_1_5l_indicator(self) -> Dict[str, Any]:
        """Testa indicador MOMO-1.5L"""
        print("\n🧪 Testando MOMO-1.5L Indicator...")

        try:
            # Teste 1: Cálculo básico
            prices = np.array(self.test_data['prices'])
            volumes = np.array(self.test_data['volumes'])

            result = self.momo_indicator.calculate(prices, volumes)

            # Validações
            assert -1.5 <= result.score <= 1.5, f"Score fora do range: {result.score}"
            assert result.signal_strength in ['weak', 'moderate', 'strong']
            assert result.signal_direction in ['bullish', 'bearish', 'neutral']

            # Teste 2: Sinais de trading
            signals = self.momo_indicator.get_trading_signals(result)
            assert isinstance(signals, dict)
            assert 'long_signal' in signals
            assert 'short_signal' in signals

            # Teste 3: Performance
            perf_stats = self.momo_indicator.get_performance_stats()
            assert perf_stats['avg_calc_time_ms'] < 10.0  # < 10ms target

            print(f"   ✅ MOMO-1.5L Score: {result.score:.4f}")
            print(f"   ✅ Signal Strength: {result.signal_strength}")
            print(f"   ✅ Direction: {result.signal_direction}")
            print(f"   ✅ Calc Time: {perf_stats['avg_calc_time_ms']:.2f}ms")

            return {
                'passed': True,
                'score': result.score,
                'signals': signals,
                'performance': perf_stats
            }

        except Exception as e:
            print(f"   ❌ MOMO-1.5L Test Failed: {str(e)}")
            return {'passed': False, 'error': str(e)}

    async def test_aggression_score(self) -> Dict[str, Any]:
        """Testa High Aggression Score"""
        print("\n🧪 Testando High Aggression Score...")

        try:
            # Criar market signals
            market_signals = MarketSignals(
                prices=np.array(self.test_data['prices'][-20:]),  # Últimos 20 períodos
                volumes=np.array(self.test_data['volumes'][-20:]),
                timestamp=time.time(),
                symbol="BTC/USDT",

                # Adicionar sinais para alta agressividade
                rsi=78.0,                    # Overbought
                order_book_imbalance=0.7,    # Strong buy pressure
                news_sentiment=0.85,         # Very positive
                realized_volatility=0.025,   # 2.5% volatility
                bid_ask_spread=0.001        # 0.1% spread
            )

            # Calcular aggression score
            result = await self.aggression_scorer.calculate_async(market_signals)

            # Validações
            assert 0.0 <= result.score <= 1.0, f"Score fora do range: {result.score}"
            assert result.confidence >= 0.0
            assert isinstance(result.components, dict)
            assert len(result.components) == 6  # 6 componentes esperados

            # Performance check
            perf_stats = self.aggression_scorer.get_performance_stats()
            assert perf_stats['avg_calc_time_ms'] < 15.0  # < 15ms target

            print(f"   ✅ Aggression Score: {result.score:.4f}")
            print(f"   ✅ Explosion Ready: {result.explosion_ready}")
            print(f"   ✅ Risk Level: {result.risk_level.value}")
            print(f"   ✅ Confidence: {result.confidence:.3f}")
            print(f"   ✅ Calc Time: {perf_stats['avg_calc_time_ms']:.2f}ms")

            return {
                'passed': True,
                'score': result.score,
                'explosion_ready': result.explosion_ready,
                'components': result.components,
                'performance': perf_stats
            }

        except Exception as e:
            print(f"   ❌ Aggression Score Test Failed: {str(e)}")
            return {'passed': False, 'error': str(e)}

    async def test_strategy_1_5l(self) -> Dict[str, Any]:
        """Testa Strategy 1.5L"""
        print("\n🧪 Testando Strategy 1.5L...")

        try:
            # Simular várias decisões de trading
            decisions = []

            for i in range(10):  # 10 snapshots de teste
                # Criar market snapshot
                price_idx = min(i + 50, len(self.test_data['prices']) - 1)
                current_price = self.test_data['prices'][price_idx]
                current_volume = self.test_data['volumes'][price_idx]

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
                        'daily_pnl': 0.005 * i,  # Simular P&L crescente
                        'account': 'test_acc'
                    }
                )

                # Atualizar histórico da estratégia (simular)
                self.strategy_1_5l.price_history = self.test_data['prices'][:price_idx+1]
                self.strategy_1_5l.volume_history = self.test_data['volumes'][:price_idx+1]

                # Obter decisão
                decision = self.strategy_1_5l.decide(snapshot)
                decisions.append(decision)

            # Validações
            valid_decisions = [d for d in decisions if d.get('side') in ['BUY', 'SELL']]
            print(f"   ✅ Decisions Generated: {len(decisions)}")
            print(f"   ✅ Trading Signals: {len(valid_decisions)}")

            if valid_decisions:
                sample_decision = valid_decisions[0]
                assert sample_decision['side'] in ['BUY', 'SELL']
                assert sample_decision['qty'] > 0
                assert 'meta' in sample_decision
                print(f"   ✅ Sample Decision: {sample_decision['side']} {sample_decision['qty']}")

            # Performance metrics
            perf_metrics = self.strategy_1_5l.get_performance_metrics()
            print(f"   ✅ Performance Metrics: {perf_metrics}")

            return {
                'passed': True,
                'total_decisions': len(decisions),
                'trading_decisions': len(valid_decisions),
                'performance': perf_metrics
            }

        except Exception as e:
            print(f"   ❌ Strategy 1.5L Test Failed: {str(e)}")
            return {'passed': False, 'error': str(e)}

    async def test_pocket_explosion_system(self) -> Dict[str, Any]:
        """Testa Pocket Explosion System"""
        print("\n🧪 Testando Pocket Explosion System...")

        try:
            # Criar market snapshot com condições favoráveis
            current_price = self.test_data['prices'][-1]

            market_snapshot = MarketSnapshot(
                class_="CRYPTO",
                symbol="BTC/USDT",
                ts_ns=int(time.time() * 1e9),
                bid=current_price - 5,
                ask=current_price + 5,
                mid=current_price,
                spread=10.0,
                features={
                    'volume': 2000.0,
                    'volume_usd_24h': 5000000.0,  # $5M volume
                    'equity_total': 10000.0,
                    'exchange': 'binance'
                }
            )

            # Sinais adicionais para alta agressividade
            additional_signals = {
                'rsi': 82.0,                     # Muito overbought
                'order_book_imbalance': 0.9,     # Muito bullish
                'news_sentiment': 0.95,          # Extremamente positivo
                'realized_volatility': 0.04,     # 4% volatilidade
                'trade_flow': 0.8               # Strong buy flow
            }

            # Teste 1: Avaliar oportunidade
            opportunity = await self.pocket_system.evaluate_opportunity(
                market_snapshot, additional_signals
            )

            print(f"   📊 Opportunity Score: {opportunity['opportunity_score']:.4f}")
            print(f"   🎯 Execute: {opportunity['execute']}")

            if opportunity['execute']:
                # Teste 2: Executar pocket explosion
                execution_result = await self.pocket_system.execute_explosion(
                    market_snapshot, opportunity['explosion_params']
                )

                print(f"   ⚡ Execution Success: {execution_result.success}")
                print(f"   ⏱️ Execution Time: {execution_result.execution_time_ms:.1f}ms")

                if execution_result.success:
                    position = execution_result.position
                    print(f"   💰 Allocation: ${position.allocation_usd:.2f}")
                    print(f"   📈 Leverage: {position.leverage:.1f}x")
                    print(f"   ⏰ Duration: {position.max_duration_seconds}s")

                    # Aguardar um pouco para monitoramento
                    await asyncio.sleep(2)

                    # Verificar status
                    active_pockets = self.pocket_system.get_active_pockets()
                    print(f"   📊 Active Pockets: {len(active_pockets)}")

            # Performance stats
            perf_stats = self.pocket_system.get_performance_stats()
            print(f"   📈 Performance Stats: {perf_stats}")

            return {
                'passed': True,
                'opportunity_evaluated': True,
                'execution_attempted': opportunity['execute'],
                'performance': perf_stats
            }

        except Exception as e:
            print(f"   ❌ Pocket Explosion Test Failed: {str(e)}")
            return {'passed': False, 'error': str(e)}

    async def test_full_integration(self) -> Dict[str, Any]:
        """Teste de integração completa"""
        print("\n🧪 Testando Integração Completa...")

        try:
            # Cenário: Sistema detecta oportunidade e executa trading completo
            current_price = self.test_data['prices'][-1]

            # 1. Market data
            market_snapshot = MarketSnapshot(
                class_="CRYPTO",
                symbol="BTC/USDT",
                ts_ns=int(time.time() * 1e9),
                bid=current_price - 5,
                ask=current_price + 5,
                mid=current_price,
                spread=10.0,
                features={
                    'volume': 1500.0,
                    'volume_usd_24h': 3000000.0,
                    'equity_total': 10000.0,
                    'open_positions': [],
                    'daily_pnl': 0.008,
                    'account': 'integration_test'
                }
            )

            # 2. Strategy 1.5L decision
            self.strategy_1_5l.price_history = self.test_data['prices']
            self.strategy_1_5l.volume_history = self.test_data['volumes']

            strategy_decision = self.strategy_1_5l.decide(market_snapshot)
            print(f"   🤖 Strategy Decision: {strategy_decision.get('side', 'HOLD')}")

            # 3. Se há sinal de trading, verificar pocket opportunity
            if 'side' in strategy_decision:
                additional_signals = {
                    'rsi': 75.0,
                    'order_book_imbalance': 0.6,
                    'news_sentiment': 0.8,
                    'realized_volatility': 0.03
                }

                pocket_opportunity = await self.pocket_system.evaluate_opportunity(
                    market_snapshot, additional_signals
                )

                print(f"   ⚡ Pocket Opportunity: {pocket_opportunity['execute']}")

                if pocket_opportunity['execute']:
                    # Execute pocket explosion
                    pocket_result = await self.pocket_system.execute_explosion(
                        market_snapshot, pocket_opportunity['explosion_params']
                    )
                    print(f"   🚀 Pocket Executed: {pocket_result.success}")

            # 4. Verificar todos os componentes funcionaram
            momo_result = self.momo_indicator.calculate(
                np.array(self.test_data['prices']),
                np.array(self.test_data['volumes'])
            )

            integration_score = {
                'momo_working': abs(momo_result.score) > 0,
                'strategy_working': len(strategy_decision) > 0,
                'pocket_system_working': len(self.pocket_system.get_performance_stats()) > 0,
                'full_pipeline': 'side' in strategy_decision
            }

            all_working = all(integration_score.values())
            print(f"   🔄 Integration Score: {integration_score}")
            print(f"   ✅ Full Integration: {all_working}")

            return {
                'passed': all_working,
                'components_status': integration_score,
                'strategy_decision': strategy_decision,
                'momo_score': momo_result.score
            }

        except Exception as e:
            print(f"   ❌ Full Integration Test Failed: {str(e)}")
            return {'passed': False, 'error': str(e)}

    async def run_test(self, test_name: str, test_func):
        """Executa um teste e atualiza estatísticas"""
        self.total_tests += 1

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result.get('passed', False):
                self.passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                self.failed_tests += 1
                print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")

            self.test_results[test_name] = result

        except Exception as e:
            self.failed_tests += 1
            print(f"❌ {test_name}: ERROR - {str(e)}")
            self.test_results[test_name] = {'passed': False, 'error': str(e)}

    async def run_all_tests(self):
        """Executa todos os testes de integração"""
        print("🚀 Iniciando Testes de Integração - Prioridade 1")
        print("=" * 60)

        # Lista de testes
        tests = [
            ("MOMO-1.5L Indicator", self.test_momo_1_5l_indicator),
            ("High Aggression Score", self.test_aggression_score),
            ("Strategy 1.5L", self.test_strategy_1_5l),
            ("Pocket Explosion System", self.test_pocket_explosion_system),
            ("Full Integration", self.test_full_integration)
        ]

        # Executar testes
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            await self.run_test(test_name, test_func)

        # Relatório final
        self.print_final_report()

    def print_final_report(self):
        """Imprime relatório final dos testes"""
        print("\n" + "="*60)
        print("📋 RELATÓRIO FINAL DOS TESTES")
        print("="*60)

        print(f"Total de Testes: {self.total_tests}")
        print(f"✅ Testes Aprovados: {self.passed_tests}")
        print(f"❌ Testes Falharam: {self.failed_tests}")

        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"📊 Taxa de Sucesso: {success_rate:.1f}%")

        if success_rate >= 80:
            print("\n🎉 INTEGRAÇÃO BEM-SUCEDIDA!")
            print("   Componentes de Prioridade 1 estão funcionando corretamente.")
        elif success_rate >= 60:
            print("\n⚠️  INTEGRAÇÃO PARCIAL")
            print("   Alguns componentes precisam de ajustes.")
        else:
            print("\n🚨 INTEGRAÇÃO COM PROBLEMAS")
            print("   Componentes críticos precisam de revisão.")

        print("\n📝 Resumo por Componente:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"   {status} {test_name}")
            if not result.get('passed', False) and 'error' in result:
                print(f"      Error: {result['error']}")

        print("\n🔄 Próximos Passos:")
        if success_rate >= 80:
            print("   1. Implementar componentes de Prioridade 2")
            print("   2. Configurar ambiente de teste avançado")
            print("   3. Integração com exchanges reais")
        else:
            print("   1. Revisar e corrigir componentes falhando")
            print("   2. Re-executar testes de integração")
            print("   3. Validar configurações")


async def main():
    """Função principal"""
    tester = IntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
