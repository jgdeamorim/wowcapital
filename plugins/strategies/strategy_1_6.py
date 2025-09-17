#!/usr/bin/env python3
"""
Estratégia 1.6 - Regime + Microestrutura + Pocket Explosão
Estratégia agressiva com classificação de regime e análise microestrutura

Performance Target: 1.18% daily return
Risk Profile: Agressivo com controle rígido
Leverage: Até 25x em Pocket Explosão

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import asyncio

from core.contracts import MarketSnapshot, TradingSignal, Position
from core.config import settings
from plugins.strategies.base_strategy import BaseStrategy
from indicators.regime.regime_net import RegimeNet
from indicators.momentum.momo_1_5l import MOMO_1_5L
from indicators.composite.high_aggression_score import HighAggressionScore
from indicators.volatility.vrp_fast import VRPFast
from indicators.trend.dynamic_macd import DynamicMACD
from execution.pocket_explosion.core import PocketExplosion
from risk.position_sizer import PositionSizer
from utils.performance import timeit


@dataclass
class Strategy16Config:
    """Configuração específica da Estratégia 1.6"""

    # Core parameters
    regime_weight: float = 0.35
    momo_weight: float = 0.25
    microstructure_weight: float = 0.20
    macd_weight: float = 0.20

    # Signal thresholds
    long_threshold: float = 0.25
    short_threshold: float = -0.25
    strong_signal_threshold: float = 0.60

    # Pocket Explosion
    pocket_trigger_threshold: float = 0.88  # Lower than 0.92 for more aggressive
    pocket_confidence_min: float = 0.75
    pocket_max_positions: int = 3

    # Risk management
    base_position_pct: float = 0.015  # 1.5% base position
    aggressive_position_pct: float = 0.035  # 3.5% for strong signals
    max_portfolio_heat: float = 0.08  # 8% maximum portfolio exposure

    # Regime-specific parameters
    trending_boost: float = 1.3
    volatile_reduction: float = 0.7
    breakout_boost: float = 1.5
    ranging_reduction: float = 0.6

    # Timeframes
    fast_period: int = 10
    medium_period: int = 21
    slow_period: int = 50

    # Performance tracking
    target_daily_return: float = 0.0118  # 1.18%
    max_drawdown: float = 0.045  # 4.5%

    def __post_init__(self):
        total_weight = (self.regime_weight + self.momo_weight +
                       self.microstructure_weight + self.macd_weight)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")


class Strategy16(BaseStrategy):
    """
    Estratégia 1.6 - Regime + Microestrutura + Pocket Explosão

    Lógica multi-componente:
    1. Classificação de regime de mercado (RegimeNet)
    2. Momentum (MOMO-1.5L)
    3. Microestrutura (VRP-fast + Aggression Score)
    4. Trend confirmation (Dynamic MACD)
    5. Pocket Explosão para sinais extremos
    """

    def __init__(self, config: Optional[Strategy16Config] = None):
        super().__init__()

        self.config = config or Strategy16Config()
        self.name = "Strategy-1.6"
        self.version = "1.0.0"

        # Initialize indicators
        self._init_indicators()

        # Initialize execution engines
        self.pocket_explosion = PocketExplosion()
        self.position_sizer = PositionSizer()

        # State tracking
        self.last_regime: Optional[str] = None
        self.regime_confidence: float = 0.0
        self.signal_history: List[Dict[str, Any]] = []
        self.pocket_positions: List[Dict[str, Any]] = []

        # Performance tracking
        self.trades_today: int = 0
        self.pnl_today: float = 0.0
        self.last_signal_time: Optional[datetime] = None

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Strategy 1.6 initialized with config: {self.config}")

    def _init_indicators(self):
        """Initialize all indicators"""
        self.regime_net = RegimeNet()
        self.momo = MOMO_1_5L()
        self.aggression = HighAggressionScore()
        self.vrp = VRPFast()
        self.macd = DynamicMACD()

        self.logger.info("All indicators initialized for Strategy 1.6")

    @timeit
    def generate_signal(
        self,
        snapshot: MarketSnapshot,
        positions: List[Position],
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Gerar sinal de trading baseado na análise multi-componente

        Args:
            snapshot: Dados de mercado atuais
            positions: Posições atuais
            context: Contexto adicional (histórico de preços, etc.)

        Returns:
            TradingSignal se condições atendidas, None caso contrário
        """
        try:
            start_time = time.time()

            # Get required data from context
            price_history = context.get('price_history', [])
            volume_history = context.get('volume_history', [])
            orderbook_data = context.get('orderbook', {})

            if len(price_history) < self.config.slow_period:
                self.logger.debug("Insufficient price history for analysis")
                return None

            # 1. Regime Classification
            regime_data = self._analyze_regime(price_history, volume_history)

            # 2. Momentum Analysis
            momentum_data = self._analyze_momentum(price_history, volume_history)

            # 3. Microstructure Analysis
            microstructure_data = self._analyze_microstructure(
                snapshot, orderbook_data, price_history, volume_history
            )

            # 4. Trend Confirmation
            trend_data = self._analyze_trend(price_history, volume_history)

            # 5. Composite Signal Calculation
            composite_signal = self._calculate_composite_signal(
                regime_data, momentum_data, microstructure_data, trend_data
            )

            # 6. Risk and Position Management
            risk_adjusted_signal = self._apply_risk_management(
                composite_signal, positions, snapshot
            )

            # 7. Check for Pocket Explosion Conditions
            pocket_signal = self._check_pocket_explosion(
                risk_adjusted_signal, momentum_data, microstructure_data
            )

            # 8. Generate Final Signal
            final_signal = self._generate_final_signal(
                risk_adjusted_signal, pocket_signal, snapshot, context
            )

            # Track performance
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Signal generation took {execution_time:.2f}ms")

            if final_signal:
                self._track_signal_history(final_signal, {
                    'regime': regime_data,
                    'momentum': momentum_data,
                    'microstructure': microstructure_data,
                    'trend': trend_data,
                    'composite': composite_signal,
                    'execution_time': execution_time
                })

            return final_signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None

    def _analyze_regime(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Análise de regime de mercado"""
        try:
            # Get regime classification
            regime_result = self.regime_net.classify_regime(
                np.array(prices), np.array(volumes)
            )

            regime = regime_result.get('regime', 'UNKNOWN')
            confidence = regime_result.get('confidence', 0.0)
            probabilities = regime_result.get('probabilities', {})

            # Calculate regime-specific adjustments
            regime_multiplier = {
                'TRENDING_UP': self.config.trending_boost,
                'TRENDING_DOWN': self.config.trending_boost,
                'BREAKOUT': self.config.breakout_boost,
                'VOLATILE': self.config.volatile_reduction,
                'RANGING': self.config.ranging_reduction
            }.get(regime, 1.0)

            # Track regime changes
            if self.last_regime != regime:
                self.logger.info(f"Regime change: {self.last_regime} → {regime} (confidence: {confidence:.3f})")
                self.last_regime = regime
                self.regime_confidence = confidence

            return {
                'regime': regime,
                'confidence': confidence,
                'probabilities': probabilities,
                'multiplier': regime_multiplier,
                'score': confidence if regime in ['TRENDING_UP', 'BREAKOUT'] else -confidence if regime in ['TRENDING_DOWN'] else 0.0
            }

        except Exception as e:
            self.logger.error(f"Error in regime analysis: {str(e)}")
            return {'regime': 'UNKNOWN', 'confidence': 0.0, 'multiplier': 1.0, 'score': 0.0}

    def _analyze_momentum(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Análise de momentum"""
        try:
            prices_array = np.array(prices)
            volumes_array = np.array(volumes) if volumes else None

            # MOMO-1.5L calculation
            momo_score = self.momo.calculate(prices_array, volumes_array)

            # Additional momentum metrics
            returns = np.diff(prices_array) / prices_array[:-1]
            momentum_strength = np.mean(returns[-self.config.fast_period:])
            momentum_consistency = 1.0 - np.std(returns[-self.config.fast_period:]) / (abs(momentum_strength) + 1e-8)

            return {
                'momo_score': momo_score,
                'strength': momentum_strength,
                'consistency': momentum_consistency,
                'raw_momentum': momentum_strength * 100,
                'score': momo_score
            }

        except Exception as e:
            self.logger.error(f"Error in momentum analysis: {str(e)}")
            return {'momo_score': 0.0, 'strength': 0.0, 'consistency': 0.0, 'score': 0.0}

    def _analyze_microstructure(
        self,
        snapshot: MarketSnapshot,
        orderbook: Dict[str, Any],
        prices: List[float],
        volumes: List[float]
    ) -> Dict[str, Any]:
        """Análise de microestrutura"""
        try:
            prices_array = np.array(prices)
            volumes_array = np.array(volumes) if volumes else None

            # High Aggression Score
            aggr_result = self.aggression.calculate(snapshot, {
                'price_history': prices,
                'volume_history': volumes,
                'orderbook': orderbook
            })

            aggr_score = aggr_result.get('score', 0.0)

            # VRP-fast for volatility risk
            vrp_score = self.vrp.calculate(prices_array, volumes_array)

            # Spread analysis
            spread = snapshot.spread / snapshot.mid if snapshot.mid > 0 else 0.0
            spread_score = 1.0 - min(1.0, spread / 0.001)  # Normalize spread (0.1% = full penalty)

            # Volume profile
            if volumes_array is not None and len(volumes_array) > 10:
                recent_volume = np.mean(volumes_array[-10:])
                avg_volume = np.mean(volumes_array[-50:]) if len(volumes_array) >= 50 else recent_volume
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                volume_score = min(2.0, volume_ratio) / 2.0
            else:
                volume_score = 0.5

            # Composite microstructure score
            microstructure_score = (
                aggr_score * 0.4 +
                vrp_score * 0.3 +
                spread_score * 0.15 +
                volume_score * 0.15
            )

            return {
                'aggression_score': aggr_score,
                'vrp_score': vrp_score,
                'spread_score': spread_score,
                'volume_score': volume_score,
                'microstructure_score': microstructure_score,
                'score': microstructure_score
            }

        except Exception as e:
            self.logger.error(f"Error in microstructure analysis: {str(e)}")
            return {'aggression_score': 0.0, 'vrp_score': 0.0, 'score': 0.0}

    def _analyze_trend(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Análise de tendência"""
        try:
            prices_array = np.array(prices)
            volumes_array = np.array(volumes) if volumes else None

            # Dynamic MACD
            macd_result = self.macd.calculate(prices_array, volumes_array)
            macd_score = macd_result.get('signal', 0.0)

            # Trend strength
            if len(prices_array) >= self.config.medium_period:
                sma_short = np.mean(prices_array[-self.config.fast_period:])
                sma_medium = np.mean(prices_array[-self.config.medium_period:])
                trend_strength = (sma_short - sma_medium) / sma_medium
            else:
                trend_strength = 0.0

            return {
                'macd_score': macd_score,
                'trend_strength': trend_strength,
                'score': macd_score
            }

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {'macd_score': 0.0, 'trend_strength': 0.0, 'score': 0.0}

    def _calculate_composite_signal(
        self,
        regime_data: Dict[str, Any],
        momentum_data: Dict[str, Any],
        microstructure_data: Dict[str, Any],
        trend_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcular sinal composto de todos os componentes"""

        # Extract individual scores
        regime_score = regime_data.get('score', 0.0)
        momentum_score = momentum_data.get('score', 0.0)
        microstructure_score = microstructure_data.get('score', 0.0)
        trend_score = trend_data.get('score', 0.0)

        # Apply regime multiplier to all components
        regime_multiplier = regime_data.get('multiplier', 1.0)

        # Weighted composite score
        composite_score = (
            regime_score * self.config.regime_weight +
            momentum_score * self.config.momo_weight +
            microstructure_score * self.config.microstructure_weight +
            trend_score * self.config.macd_weight
        ) * regime_multiplier

        # Signal classification
        if composite_score > self.config.strong_signal_threshold:
            signal_strength = "STRONG"
            side = "BUY"
        elif composite_score > self.config.long_threshold:
            signal_strength = "MEDIUM"
            side = "BUY"
        elif composite_score < -self.config.strong_signal_threshold:
            signal_strength = "STRONG"
            side = "SELL"
        elif composite_score < self.config.short_threshold:
            signal_strength = "MEDIUM"
            side = "SELL"
        else:
            signal_strength = "WEAK"
            side = "HOLD"

        # Calculate confidence
        confidence = min(0.99, abs(composite_score))

        return {
            'composite_score': composite_score,
            'side': side,
            'strength': signal_strength,
            'confidence': confidence,
            'regime_contribution': regime_score * self.config.regime_weight * regime_multiplier,
            'momentum_contribution': momentum_score * self.config.momo_weight * regime_multiplier,
            'microstructure_contribution': microstructure_score * self.config.microstructure_weight * regime_multiplier,
            'trend_contribution': trend_score * self.config.macd_weight * regime_multiplier
        }

    def _apply_risk_management(
        self,
        composite_signal: Dict[str, Any],
        positions: List[Position],
        snapshot: MarketSnapshot
    ) -> Dict[str, Any]:
        """Aplicar gestão de risco ao sinal"""

        # Calculate current portfolio heat
        total_exposure = sum(abs(pos.notional_value) for pos in positions)
        portfolio_heat = total_exposure / (snapshot.mid * 10000)  # Assuming 10k base equity

        # Risk adjustments
        risk_multiplier = 1.0

        # Reduce size if portfolio is too hot
        if portfolio_heat > self.config.max_portfolio_heat:
            risk_multiplier *= 0.5
            self.logger.warning(f"Portfolio heat too high ({portfolio_heat:.3f}), reducing signal strength")

        # Check for conflicting positions
        conflicting_positions = [p for p in positions if p.symbol == snapshot.symbol]
        if conflicting_positions:
            # Reduce signal if we already have exposure
            risk_multiplier *= 0.7

        # Apply risk adjustments
        adjusted_signal = composite_signal.copy()
        adjusted_signal['composite_score'] *= risk_multiplier
        adjusted_signal['confidence'] *= risk_multiplier
        adjusted_signal['risk_multiplier'] = risk_multiplier
        adjusted_signal['portfolio_heat'] = portfolio_heat

        return adjusted_signal

    def _check_pocket_explosion(
        self,
        signal: Dict[str, Any],
        momentum_data: Dict[str, Any],
        microstructure_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Verificar condições para Pocket Explosão"""

        # Check basic conditions
        if signal.get('side') == 'HOLD':
            return None

        aggression_score = microstructure_data.get('aggression_score', 0.0)
        momo_score = abs(momentum_data.get('momo_score', 0.0))
        confidence = signal.get('confidence', 0.0)

        # Pocket explosion triggers
        if (aggression_score >= self.config.pocket_trigger_threshold and
            confidence >= self.config.pocket_confidence_min and
            momo_score >= 0.5):

            # Check current pocket positions limit
            active_pockets = len([p for p in self.pocket_positions if p.get('active', False)])
            if active_pockets >= self.config.pocket_max_positions:
                self.logger.info(f"Pocket explosion limit reached ({active_pockets}/{self.config.pocket_max_positions})")
                return None

            return {
                'trigger': True,
                'aggression_score': aggression_score,
                'momo_score': momo_score,
                'confidence': confidence,
                'priority': 'HIGH' if aggression_score >= 0.95 else 'MEDIUM'
            }

        return None

    def _generate_final_signal(
        self,
        signal: Dict[str, Any],
        pocket_signal: Optional[Dict[str, Any]],
        snapshot: MarketSnapshot,
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Gerar sinal final de trading"""

        if signal.get('side') == 'HOLD':
            return None

        # Determine position sizing
        base_pct = (self.config.aggressive_position_pct if signal.get('strength') == 'STRONG'
                   else self.config.base_position_pct)

        position_pct = base_pct * signal.get('risk_multiplier', 1.0)

        # Create trading signal
        trading_signal = TradingSignal(
            symbol=snapshot.symbol,
            side=signal['side'],
            signal_strength=signal.get('strength', 'MEDIUM'),
            confidence=signal.get('confidence', 0.5),
            position_pct=position_pct,
            entry_price=snapshot.mid,
            timestamp=datetime.now(),
            strategy_name=self.name,
            metadata={
                'composite_score': signal.get('composite_score', 0.0),
                'regime': signal.get('regime_contribution', 0.0),
                'momentum': signal.get('momentum_contribution', 0.0),
                'microstructure': signal.get('microstructure_contribution', 0.0),
                'trend': signal.get('trend_contribution', 0.0),
                'portfolio_heat': signal.get('portfolio_heat', 0.0),
                'pocket_explosion': pocket_signal is not None,
                'strategy_version': self.version
            }
        )

        # Add pocket explosion details if applicable
        if pocket_signal:
            trading_signal.metadata.update({
                'pocket_priority': pocket_signal.get('priority', 'MEDIUM'),
                'pocket_aggression': pocket_signal.get('aggression_score', 0.0),
                'execution_type': 'POCKET_EXPLOSION'
            })

            self.logger.info(f"🔴 POCKET EXPLOSION triggered: {signal['side']} {snapshot.symbol} "
                           f"(aggr: {pocket_signal.get('aggression_score', 0.0):.3f})")

        self.last_signal_time = datetime.now()
        return trading_signal

    def _track_signal_history(self, signal: TradingSignal, analysis_data: Dict[str, Any]):
        """Track signal history for analysis"""
        self.signal_history.append({
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'side': signal.side,
            'confidence': signal.confidence,
            'analysis': analysis_data
        })

        # Keep only last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        return {
            'name': self.name,
            'version': self.version,
            'last_regime': self.last_regime,
            'regime_confidence': self.regime_confidence,
            'trades_today': self.trades_today,
            'pnl_today': self.pnl_today,
            'active_pocket_positions': len([p for p in self.pocket_positions if p.get('active', False)]),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'config': self.config.__dict__
        }

    def update_performance(self, pnl: float, trade_count: int):
        """Update strategy performance metrics"""
        self.pnl_today += pnl
        self.trades_today += trade_count

    def reset_daily_metrics(self):
        """Reset daily performance metrics"""
        self.trades_today = 0
        self.pnl_today = 0.0
        self.signal_history.clear()

        self.logger.info("Daily metrics reset for Strategy 1.6")