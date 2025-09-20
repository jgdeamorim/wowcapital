"""
Pocket Explosion System - Core Module
Sistema de micro-alocações com alta alavancagem para oportunidades de alta convicção

Características:
- Alocação: 0.5-2.5% equity
- Alavancagem: até 25x
- Duração: <60s
- Detonação automática: ImpulseScore ≥ 0.92

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import asyncio
import time
import random
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging

# Core imports (allow running inside repo or packaged as backend)
try:
    from backend.core.contracts import MarketSnapshot, OrderRequest  # type: ignore
    from backend.indicators.composite.high_aggression_score import (
        HighAggressionScore, MarketSignals, AggressionResult,
    )  # type: ignore
except ImportError:  # pragma: no cover - local fallback
    from core.contracts import MarketSnapshot, OrderRequest
    from indicators.composite.high_aggression_score import (  # type: ignore
        HighAggressionScore,
        MarketSignals,
        AggressionResult,
    )


class PocketStatus(Enum):
    """Status de uma pocket explosion"""
    ARMED = "armed"
    TRIGGERED = "triggered"
    EXECUTING = "executing"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class PocketType(Enum):
    """Tipo de pocket explosion"""
    MOMENTUM = "momentum"
    MEAN_REVERT = "mean_revert"
    BREAKOUT = "breakout"
    SCALP = "scalp"


@dataclass
class PocketConfig:
    """Configuração do sistema Pocket Explosion"""
    # Alocação
    min_allocation_pct: float = 0.005      # 0.5% mínimo
    max_allocation_pct: float = 0.025      # 2.5% máximo
    base_allocation_pct: float = 0.015     # 1.5% base

    # Alavancagem
    min_leverage: float = 5.0              # 5x mínimo
    max_leverage: float = 25.0             # 25x máximo
    base_leverage: float = 15.0            # 15x base

    # Timing
    max_duration_seconds: int = 60         # 60s máximo
    min_duration_seconds: int = 30         # 30s mínimo
    explosion_threshold: float = 0.92      # ImpulseScore threshold

    # Risk controls
    hard_stop_loss_pct: float = 0.004      # 0.4% não-negociável
    max_stop_loss_pct: float = 0.006       # 0.6% máximo absoluto
    kill_switch_dd_pct: float = 0.15       # 15% portfolio DD = kill all

    # Operational
    max_concurrent_pockets: int = 3         # Máximo simultâneo
    cooldown_seconds: int = 300            # 5min cooldown
    min_liquidity_usd: float = 100000      # $100k volume mínimo


@dataclass
class PocketPosition:
    """Representa uma pocket explosion ativa"""
    id: str
    symbol: str
    side: str                              # BUY/SELL
    entry_price: float
    quantity: float
    leverage: float
    allocation_pct: float
    allocation_usd: float

    # Timing
    start_time: float
    max_duration_seconds: int
    time_stop_at: float

    # Risk controls
    stop_loss_price: float
    take_profit_price: Optional[float] = None

    # Status
    status: PocketStatus = PocketStatus.ARMED
    pocket_type: PocketType = PocketType.MOMENTUM

    # Performance tracking
    current_pnl_usd: float = 0.0
    current_pnl_pct: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0

    # Meta data
    impulse_score: float = 0.0
    conviction_level: str = "medium"
    exchange: str = ""
    order_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class PocketExecutionResult:
    """Resultado de execução de pocket explosion"""
    success: bool
    position: Optional[PocketPosition]
    execution_time_ms: float
    error_message: str = ""
    exchange_response: Optional[Dict] = None


class PocketExplosionSystem:
    """
    Sistema central de Pocket Explosions

    Responsabilidades:
    - Detecção de oportunidades de alta convicção
    - Execução ultra-rápida (<50ms)
    - Risk management rigoroso
    - Monitoramento ativo de posições
    """

    def __init__(self, config: Optional[PocketConfig] = None):
        self.config = config or PocketConfig()
        self.aggression_scorer = HighAggressionScore()

        # Estado do sistema
        self.active_pockets: Dict[str, PocketPosition] = {}
        self.pocket_history: List[PocketPosition] = []
        self.last_explosion_time = 0.0
        self.total_explosions = 0

        # Performance metrics
        self.performance_stats = {
            'total_explosions': 0,
            'successful_explosions': 0,
            'total_pnl_usd': 0.0,
            'avg_duration_seconds': 0.0,
            'avg_return_pct': 0.0,
            'max_concurrent': 0,
            'avg_execution_time_ms': 0.0
        }

        # Risk tracking
        self.daily_pocket_pnl = 0.0
        self.max_daily_drawdown = 0.0

        # Logger
        self.logger = logging.getLogger('PocketExplosion')

    async def evaluate_opportunity(self,
                                 market_data: MarketSnapshot,
                                 additional_signals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Avalia se há oportunidade para pocket explosion

        Args:
            market_data: Dados de mercado atuais
            additional_signals: Sinais adicionais opcionais

        Returns:
            Dict: Resultado da avaliação
        """
        try:
            # 1. Verificar pré-condições básicas
            pre_check = await self._check_preconditions(market_data)
            if not pre_check['allowed']:
                return {
                    'execute': False,
                    'reason': pre_check['reason'],
                    'opportunity_score': 0.0
                }

            # 2. Calcular aggression score
            market_signals = self._build_market_signals(market_data, additional_signals)
            aggression_result = await self.aggression_scorer.calculate_async(market_signals)

            # 3. Verificar threshold para explosão
            if aggression_result.score < self.config.explosion_threshold:
                return {
                    'execute': False,
                    'reason': f'Score below threshold: {aggression_result.score:.4f} < {self.config.explosion_threshold}',
                    'opportunity_score': aggression_result.score,
                    'aggression_result': aggression_result
                }

            # 4. Validações adicionais para alta convicção
            conviction_check = self._validate_high_conviction(aggression_result, market_data)
            if not conviction_check['valid']:
                return {
                    'execute': False,
                    'reason': conviction_check['reason'],
                    'opportunity_score': aggression_result.score,
                    'aggression_result': aggression_result
                }

            # 5. Calcular parâmetros da explosão
            explosion_params = await self._calculate_explosion_params(
                aggression_result, market_data
            )

            return {
                'execute': True,
                'opportunity_score': aggression_result.score,
                'aggression_result': aggression_result,
                'explosion_params': explosion_params,
                'estimated_duration': f"{explosion_params['duration_seconds']}s",
                'risk_level': 'EXTREME'
            }

        except Exception as e:
            self.logger.error(f"Error evaluating pocket opportunity: {str(e)}")
            return {
                'execute': False,
                'reason': f'Evaluation error: {str(e)}',
                'opportunity_score': 0.0
            }

    async def execute_explosion(self,
                              market_data: MarketSnapshot,
                              explosion_params: Dict) -> PocketExecutionResult:
        """
        Executa pocket explosion com ultra-baixa latência

        Args:
            market_data: Dados de mercado
            explosion_params: Parâmetros da explosão

        Returns:
            PocketExecutionResult: Resultado da execução
        """
        start_time = time.perf_counter()

        try:
            # 1. Criar posição pocket
            position = await self._create_pocket_position(market_data, explosion_params)

            # 2. Executar ordem com multiple attempts
            execution_result = await self._execute_order_with_retries(position, market_data)

            if not execution_result['success']:
                execution_time = (time.perf_counter() - start_time) * 1000
                return PocketExecutionResult(
                    success=False,
                    position=None,
                    execution_time_ms=execution_time,
                    error_message=execution_result['error']
                )

            # 3. Configurar risk controls imediatamente
            await self._setup_risk_controls(position)

            # 4. Adicionar à lista de posições ativas
            self.active_pockets[position.id] = position
            position.status = PocketStatus.ACTIVE

            # 5. Iniciar monitoramento ativo
            asyncio.create_task(self._monitor_pocket_position(position))

            # 6. Update metrics
            self.total_explosions += 1
            self.performance_stats['total_explosions'] += 1
            execution_time = (time.perf_counter() - start_time) * 1000

            self.logger.info(
                f"Pocket explosion executed: {position.symbol} {position.side} "
                f"${position.allocation_usd:.2f} @{position.leverage}x in {execution_time:.1f}ms"
            )

            return PocketExecutionResult(
                success=True,
                position=position,
                execution_time_ms=execution_time,
                exchange_response=execution_result.get('exchange_response')
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Pocket explosion failed: {str(e)}")

            return PocketExecutionResult(
                success=False,
                position=None,
                execution_time_ms=execution_time,
                error_message=str(e)
            )

    async def _check_preconditions(self, market_data: MarketSnapshot) -> Dict[str, Any]:
        """Verifica pré-condições para pocket explosion"""

        # 1. Máximo de posições simultâneas
        if len(self.active_pockets) >= self.config.max_concurrent_pockets:
            return {
                'allowed': False,
                'reason': f'Max concurrent pockets: {len(self.active_pockets)}/{self.config.max_concurrent_pockets}'
            }

        # 2. Cooldown check
        time_since_last = time.time() - self.last_explosion_time
        if time_since_last < self.config.cooldown_seconds:
            return {
                'allowed': False,
                'reason': f'Cooldown active: {self.config.cooldown_seconds - time_since_last:.0f}s remaining'
            }

        # 3. Liquidez mínima
        volume_usd = market_data.features.get('volume_usd_24h', 0)
        if volume_usd < self.config.min_liquidity_usd:
            return {
                'allowed': False,
                'reason': f'Insufficient liquidity: ${volume_usd:.0f} < ${self.config.min_liquidity_usd:.0f}'
            }

        # 4. Spread check
        spread_pct = market_data.spread / market_data.mid
        if spread_pct > 0.005:  # 0.5% máximo
            return {
                'allowed': False,
                'reason': f'Spread too wide: {spread_pct:.3%}'
            }

        # 5. Daily drawdown check
        equity_total = market_data.features.get('equity_total', 10000)
        daily_dd_pct = abs(self.daily_pocket_pnl / equity_total)
        if daily_dd_pct >= self.config.kill_switch_dd_pct:
            return {
                'allowed': False,
                'reason': f'Daily DD limit reached: {daily_dd_pct:.2%}'
            }

        return {'allowed': True, 'reason': 'All preconditions met'}

    def _build_market_signals(self,
                            market_data: MarketSnapshot,
                            additional_signals: Optional[Dict] = None) -> MarketSignals:
        """Constrói MarketSignals para aggression scorer"""

        # Extrair dados básicos
        prices = np.array([market_data.mid])  # Seria histórico completo em produção
        volumes = np.array([market_data.features.get('volume', 0)])

        # Construir sinais de mercado
        signals = MarketSignals(
            prices=prices,
            volumes=volumes,
            timestamp=market_data.ts_ns / 1e9,  # Convert to seconds
            symbol=market_data.symbol,
            bid_ask_spread=market_data.spread / market_data.mid
        )

        # Adicionar sinais adicionais se fornecidos
        if additional_signals:
            for key, value in additional_signals.items():
                if hasattr(signals, key):
                    setattr(signals, key, value)

        return signals

    def _validate_high_conviction(self,
                                aggression_result: AggressionResult,
                                market_data: MarketSnapshot) -> Dict[str, Any]:
        """Valida condições de alta convicção"""

        # 1. Confiança mínima
        if aggression_result.confidence < 0.75:
            return {
                'valid': False,
                'reason': f'Low confidence: {aggression_result.confidence:.3f} < 0.75'
            }

        # 2. Múltiplos componentes devem estar altos
        high_components = sum(
            1 for score in aggression_result.components.values()
            if score > 0.7
        )
        if high_components < 3:
            return {
                'valid': False,
                'reason': f'Insufficient high components: {high_components} < 3'
            }

        # 3. Verificar warnings críticos
        critical_warnings = [w for w in aggression_result.warnings
                           if 'low' in w.lower() or 'insufficient' in w.lower()]
        if critical_warnings:
            return {
                'valid': False,
                'reason': f'Critical warnings: {critical_warnings}'
            }

        return {'valid': True, 'reason': 'High conviction validated'}

    async def _calculate_explosion_params(self,
                                        aggression_result: AggressionResult,
                                        market_data: MarketSnapshot) -> Dict[str, Any]:
        """Calcula parâmetros da explosão baseado no score"""

        equity_total = market_data.features.get('equity_total', 10000)

        # 1. Calcular alocação dinâmica
        score_adjustment = (aggression_result.score - 0.92) / 0.08  # 0.92-1.0 -> 0-1
        allocation_pct = self.config.base_allocation_pct * (1 + score_adjustment)
        allocation_pct = np.clip(
            allocation_pct,
            self.config.min_allocation_pct,
            self.config.max_allocation_pct
        )

        # 2. Calcular alavancagem dinâmica
        leverage = self.config.base_leverage * (1 + score_adjustment * 0.5)
        leverage = np.clip(leverage, self.config.min_leverage, self.config.max_leverage)

        # 3. Duração baseada na convicção
        duration_base = 45  # segundos
        if aggression_result.conviction_level == 'high':
            duration = duration_base + 15
        elif aggression_result.conviction_level == 'medium':
            duration = duration_base
        else:
            duration = duration_base - 10

        duration = np.clip(duration, self.config.min_duration_seconds, self.config.max_duration_seconds)

        # 4. Determinar direção
        momentum_score = aggression_result.components.get('momentum', 0.5)
        side = 'BUY' if momentum_score > 0.5 else 'SELL'

        # 5. Calcular stops
        if side == 'BUY':
            stop_loss_price = market_data.mid * (1 - self.config.hard_stop_loss_pct)
            take_profit_price = market_data.mid * (1 + allocation_pct * leverage * 0.5)
        else:
            stop_loss_price = market_data.mid * (1 + self.config.hard_stop_loss_pct)
            take_profit_price = market_data.mid * (1 - allocation_pct * leverage * 0.5)

        return {
            'allocation_pct': allocation_pct,
            'allocation_usd': equity_total * allocation_pct,
            'leverage': leverage,
            'duration_seconds': int(duration),
            'side': side,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'impulse_score': aggression_result.score,
            'conviction_level': aggression_result.conviction_level
        }

    async def _create_pocket_position(self,
                                    market_data: MarketSnapshot,
                                    params: Dict) -> PocketPosition:
        """Cria objeto PocketPosition"""

        position_id = f"pocket_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        # Calcular quantidade
        quantity = (params['allocation_usd'] * params['leverage']) / market_data.mid

        # Determinar tipo de pocket
        pocket_type = PocketType.MOMENTUM  # Simplificado para MVP

        position = PocketPosition(
            id=position_id,
            symbol=market_data.symbol,
            side=params['side'],
            entry_price=market_data.mid,
            quantity=quantity,
            leverage=params['leverage'],
            allocation_pct=params['allocation_pct'],
            allocation_usd=params['allocation_usd'],
            start_time=time.time(),
            max_duration_seconds=params['duration_seconds'],
            time_stop_at=time.time() + params['duration_seconds'],
            stop_loss_price=params['stop_loss_price'],
            take_profit_price=params['take_profit_price'],
            status=PocketStatus.TRIGGERED,
            pocket_type=pocket_type,
            impulse_score=params['impulse_score'],
            conviction_level=params['conviction_level'],
            exchange=market_data.features.get('exchange', 'binance')
        )

        return position

    async def _execute_order_with_retries(self,
                                        position: PocketPosition,
                                        market_data: MarketSnapshot,
                                        max_retries: int = 3) -> Dict[str, Any]:
        """Executa ordem com múltiplas tentativas"""

        for attempt in range(max_retries):
            try:
                # Simular execução de ordem (em produção integraria com exchange)
                # Por ora, assumir sucesso para MVP

                await asyncio.sleep(0.01)  # Simular latência de rede

                # Simular resposta da exchange
                exchange_response = {
                    'order_id': f"exch_{position.id}",
                    'status': 'FILLED',
                    'filled_qty': position.quantity,
                    'avg_price': position.entry_price,
                    'timestamp': time.time()
                }

                position.order_id = exchange_response['order_id']
                position.status = PocketStatus.EXECUTING

                return {
                    'success': True,
                    'exchange_response': exchange_response
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': f'Max retries exceeded: {str(e)}'
                    }

                # Wait before retry
                await asyncio.sleep(0.005 * (attempt + 1))

        return {'success': False, 'error': 'Unknown execution error'}

    async def _setup_risk_controls(self, position: PocketPosition):
        """Configura controles de risco para a posição"""

        # Em produção, configuraria stops na exchange
        # Por ora, apenas tracking interno

        position.warnings.append(f"Risk controls active: SL@{position.stop_loss_price:.2f}")
        self.logger.info(f"Risk controls setup for {position.id}")

    async def _monitor_pocket_position(self, position: PocketPosition):
        """Monitora posição pocket ativa"""

        try:
            while position.status == PocketStatus.ACTIVE:
                current_time = time.time()

                # 1. Time stop check
                if current_time >= position.time_stop_at:
                    await self._close_pocket_position(position, reason="Time stop")
                    break

                # 2. P&L tracking (simulado)
                # Em produção, obteria preço atual da exchange
                current_price = position.entry_price * (1 + random.uniform(-0.02, 0.02))

                if position.side == 'BUY':
                    pnl_pct = (current_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - current_price) / position.entry_price

                position.current_pnl_pct = pnl_pct * position.leverage
                position.current_pnl_usd = position.allocation_usd * position.current_pnl_pct

                # Update max profit/drawdown
                position.max_profit = max(position.max_profit, position.current_pnl_usd)
                position.max_drawdown = min(position.max_drawdown, position.current_pnl_usd)

                # 3. Stop loss check
                stop_triggered = False
                if position.side == 'BUY' and current_price <= position.stop_loss_price:
                    stop_triggered = True
                elif position.side == 'SELL' and current_price >= position.stop_loss_price:
                    stop_triggered = True

                if stop_triggered:
                    await self._close_pocket_position(position, reason="Stop loss")
                    break

                # Monitor every 100ms
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error monitoring pocket {position.id}: {str(e)}")
            await self._close_pocket_position(position, reason=f"Monitor error: {str(e)}")

    async def _close_pocket_position(self, position: PocketPosition, reason: str):
        """Fecha posição pocket"""

        try:
            position.status = PocketStatus.CLOSING

            # Simular fechamento na exchange
            await asyncio.sleep(0.01)

            # Update performance tracking
            self.daily_pocket_pnl += position.current_pnl_usd
            if position.current_pnl_usd > 0:
                self.performance_stats['successful_explosions'] += 1

            # Move to history
            position.status = PocketStatus.CLOSED
            self.pocket_history.append(position)

            # Remove from active
            if position.id in self.active_pockets:
                del self.active_pockets[position.id]

            duration = time.time() - position.start_time

            self.logger.info(
                f"Pocket closed: {position.id} | PnL: ${position.current_pnl_usd:.2f} "
                f"({position.current_pnl_pct:.2%}) | Duration: {duration:.1f}s | Reason: {reason}"
            )

        except Exception as e:
            self.logger.error(f"Error closing pocket {position.id}: {str(e)}")
            position.status = PocketStatus.FAILED

    def get_active_pockets(self) -> Dict[str, PocketPosition]:
        """Retorna posições ativas"""
        return self.active_pockets.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""

        if self.performance_stats['total_explosions'] == 0:
            return self.performance_stats

        success_rate = (
            self.performance_stats['successful_explosions'] /
            self.performance_stats['total_explosions']
        )

        avg_pnl = (
            self.performance_stats.get('total_pnl_usd', 0) /
            self.performance_stats['total_explosions']
        )

        return {
            **self.performance_stats,
            'success_rate': success_rate,
            'avg_pnl_per_explosion': avg_pnl,
            'active_pockets_count': len(self.active_pockets),
            'daily_pnl': self.daily_pocket_pnl,
            'max_daily_drawdown': self.max_daily_drawdown
        }

    async def emergency_kill_all(self) -> Dict[str, Any]:
        """Kill switch - fecha todas as posições imediatamente"""

        if not self.active_pockets:
            return {'success': True, 'closed_positions': 0}

        positions_to_close = list(self.active_pockets.values())

        # Close all in parallel
        close_tasks = [
            self._close_pocket_position(pos, reason="Emergency kill switch")
            for pos in positions_to_close
        ]

        await asyncio.gather(*close_tasks, return_exceptions=True)

        return {
            'success': True,
            'closed_positions': len(positions_to_close),
            'reason': 'Emergency kill switch activated'
        }


# Exemplo de uso
if __name__ == "__main__":
    async def test_pocket_system():
        # Configuração de teste
        config = PocketConfig()
        pocket_system = PocketExplosionSystem(config)

        # Dados de mercado simulados
        from backend.core.contracts import MarketSnapshot

        market_data = MarketSnapshot(
            class_="CRYPTO",
            symbol="BTC/USDT",
            ts_ns=int(time.time() * 1e9),
            bid=50000.0,
            ask=50010.0,
            mid=50005.0,
            spread=10.0,
            features={
                'volume': 1000.0,
                'volume_usd_24h': 1000000.0,
                'equity_total': 10000.0,
                'exchange': 'binance'
            }
        )

        # Simular sinais de alta agressividade
        additional_signals = {
            'rsi': 75.0,
            'order_book_imbalance': 0.8,
            'news_sentiment': 0.9,
            'realized_volatility': 0.03
        }

        # Avaliar oportunidade
        opportunity = await pocket_system.evaluate_opportunity(market_data, additional_signals)
        print(f"Opportunity evaluation: {opportunity}")

        if opportunity['execute']:
            # Executar explosão
            result = await pocket_system.execute_explosion(
                market_data,
                opportunity['explosion_params']
            )
            print(f"Execution result: {result}")

            if result.success:
                # Aguardar alguns segundos
                await asyncio.sleep(5)

                # Ver status
                stats = pocket_system.get_performance_stats()
                print(f"Performance stats: {stats}")

    # Executar teste
    asyncio.run(test_pocket_system())
