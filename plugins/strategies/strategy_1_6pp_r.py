#!/usr/bin/env python3
"""
Estratégia 1.6pp-R - Refined Risk Management
Versão refinada da 1.6 com controles de risco aprimorados e correlação de portfolio

Performance Target: 0.92% daily return
Risk Profile: Conservador com alta precisão
Max Drawdown: 2.5%

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
from scipy.stats import pearsonr

from core.contracts import MarketSnapshot, TradingSignal, Position
from core.config import settings
from plugins.strategies.strategy_1_6 import Strategy16, Strategy16Config
from indicators.regime.regime_net import RegimeNet
from indicators.momentum.momo_1_5l import MOMO_1_5L
from indicators.composite.high_aggression_score import HighAggressionScore
from indicators.volatility.vrp_fast import VRPFast
from indicators.trend.dynamic_macd import DynamicMACD
from indicators.oscillators.rsi_hybrid import RSIHybrid
from execution.pocket_explosion.core import PocketExplosion
from risk.position_sizer import PositionSizer
from utils.performance import timeit


@dataclass
class Strategy16ppRConfig(Strategy16Config):
    """Configuração refinada da Estratégia 1.6pp-R"""

    # Enhanced risk parameters
    base_position_pct: float = 0.008  # 0.8% base position (more conservative)
    aggressive_position_pct: float = 0.020  # 2.0% for strong signals
    max_portfolio_heat: float = 0.045  # 4.5% maximum portfolio exposure
    max_single_position: float = 0.025  # 2.5% max single position

    # Refined thresholds (higher bars)
    long_threshold: float = 0.35
    short_threshold: float = -0.35
    strong_signal_threshold: float = 0.75

    # Pocket Explosion (more restrictive)
    pocket_trigger_threshold: float = 0.94  # Higher than 1.6
    pocket_confidence_min: float = 0.85
    pocket_max_positions: int = 2

    # Risk-adjusted parameters
    volatility_adjustment: float = 0.85
    correlation_penalty: float = 0.75
    drawdown_protection: float = 0.90

    # Enhanced signal filtering
    min_regime_confidence: float = 0.70
    min_momentum_consistency: float = 0.60
    max_microstructure_noise: float = 0.25

    # Performance targets (more conservative)
    target_daily_return: float = 0.0092  # 0.92%
    max_drawdown: float = 0.025  # 2.5%
    target_sharpe: float = 2.8

    # Multi-timeframe analysis
    use_multi_timeframe: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m'])
    timeframe_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])

    # Correlation and portfolio management
    max_correlation: float = 0.65
    correlation_lookback: int = 50
    rebalance_threshold: float = 0.15

    def __post_init__(self):
        # Validate enhanced parameters
        if len(self.timeframes) != len(self.timeframe_weights):
            raise ValueError("Timeframes and weights must have same length")
        if abs(sum(self.timeframe_weights) - 1.0) > 0.001:
            raise ValueError("Timeframe weights must sum to 1.0")
        super().__post_init__()


class Strategy16ppR(Strategy16):
    """
    Estratégia 1.6pp-R - Refined Risk Management

    Extensão da Strategy16 com:
    1. Controles de risco aprimorados
    2. Análise de correlação de portfolio
    3. Multi-timeframe analysis
    4. Enhanced signal filtering
    5. Dynamic position sizing
    6. Drawdown protection
    """

    def __init__(self, config: Optional[Strategy16ppRConfig] = None):
        # Initialize with refined config
        refined_config = config or Strategy16ppRConfig()
        super().__init__(refined_config)

        self.config = refined_config
        self.name = "Strategy-1.6pp-R"
        self.version = "1.0.0"

        # Enhanced indicators
        self.rsi_hybrid = RSIHybrid()

        # Portfolio correlation tracking
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.price_histories: Dict[str, List[float]] = {}

        # Enhanced risk tracking
        self.current_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.risk_budget_used: float = 0.0

        # Multi-timeframe state
        self.timeframe_signals: Dict[str, Dict[str, Any]] = {}
        self.timeframe_last_update: Dict[str, datetime] = {}

        # Performance attribution
        self.attribution_data: Dict[str, float] = {
            'regime': 0.0,
            'momentum': 0.0,
            'microstructure': 0.0,
            'trend': 0.0,
            'risk_adjustment': 0.0
        }

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Strategy 1.6pp-R initialized with refined config")

    @timeit
    def generate_signal(
        self,
        snapshot: MarketSnapshot,
        positions: List[Position],
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate refined trading signal with enhanced risk management
        """
        try:
            start_time = time.time()

            # Pre-flight risk checks
            if not self._pre_flight_checks(snapshot, positions, context):
                return None

            # Update correlation matrix
            self._update_correlations(snapshot, context)

            # Enhanced signal generation
            if self.config.use_multi_timeframe:
                signal = self._generate_multitimeframe_signal(snapshot, positions, context)
            else:
                signal = super().generate_signal(snapshot, positions, context)

            if not signal:
                return None

            # Enhanced risk management
            refined_signal = self._apply_enhanced_risk_management(
                signal, positions, snapshot, context
            )

            # Portfolio optimization
            optimized_signal = self._optimize_for_portfolio(
                refined_signal, positions, snapshot
            )

            # Final validation
            final_signal = self._final_signal_validation(
                optimized_signal, positions, snapshot, context
            )

            execution_time = (time.time() - start_time) * 1000
            if final_signal:
                self.logger.info(
                    f"1.6pp-R signal: {final_signal.side} {final_signal.symbol} "
                    f"(conf: {final_signal.confidence:.3f}, size: {final_signal.position_pct:.3f}) "
                    f"in {execution_time:.2f}ms"
                )

            return final_signal

        except Exception as e:
            self.logger.error(f"Error in 1.6pp-R signal generation: {str(e)}")
            return None

    def _pre_flight_checks(
        self,
        snapshot: MarketSnapshot,
        positions: List[Position],
        context: Dict[str, Any]
    ) -> bool:
        """Enhanced pre-flight risk checks"""

        # Check drawdown protection
        if self.current_drawdown > self.config.max_drawdown * self.config.drawdown_protection:
            self.logger.warning(f"Drawdown protection activated: {self.current_drawdown:.3f}")
            return False

        # Check regime confidence
        price_history = context.get('price_history', [])
        if len(price_history) >= 20:
            regime_result = self.regime_net.classify_regime(
                np.array(price_history), np.array(context.get('volume_history', []))
            )
            if regime_result.get('confidence', 0.0) < self.config.min_regime_confidence:
                self.logger.debug(f"Regime confidence too low: {regime_result.get('confidence', 0.0):.3f}")
                return False

        # Check portfolio heat
        total_exposure = sum(abs(pos.notional_value) for pos in positions)
        portfolio_heat = total_exposure / (snapshot.mid * 10000)  # Assuming 10k base
        if portfolio_heat > self.config.max_portfolio_heat * 0.9:  # 90% of limit
            self.logger.warning(f"Portfolio heat near limit: {portfolio_heat:.3f}")
            return False

        # Check market hours and volatility
        current_hour = datetime.now().hour
        if current_hour < 8 or current_hour > 20:  # Avoid overnight hours
            return False

        return True

    def _generate_multitimeframe_signal(
        self,
        snapshot: MarketSnapshot,
        positions: List[Position],
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Generate signal using multi-timeframe analysis"""

        timeframe_scores = {}

        for i, timeframe in enumerate(self.config.timeframes):
            # Get timeframe-specific data
            tf_context = self._get_timeframe_context(timeframe, context)
            if not tf_context:
                continue

            # Generate signal for this timeframe
            tf_signal = super().generate_signal(snapshot, positions, tf_context)
            if tf_signal:
                timeframe_scores[timeframe] = {
                    'score': tf_signal.metadata.get('composite_score', 0.0),
                    'confidence': tf_signal.confidence,
                    'side': tf_signal.side,
                    'weight': self.config.timeframe_weights[i]
                }

        if not timeframe_scores:
            return None

        # Combine timeframe signals
        combined_signal = self._combine_timeframe_signals(
            timeframe_scores, snapshot, positions, context
        )

        return combined_signal

    def _get_timeframe_context(self, timeframe: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get context data for specific timeframe"""
        # This would typically resample the data for different timeframes
        # For now, returning the same context (implementation would depend on data source)
        return context

    def _combine_timeframe_signals(
        self,
        timeframe_scores: Dict[str, Dict[str, Any]],
        snapshot: MarketSnapshot,
        positions: List[Position],
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Combine signals from multiple timeframes"""

        # Calculate weighted average
        total_score = 0.0
        total_confidence = 0.0
        total_weight = 0.0
        dominant_side = None

        for tf, data in timeframe_scores.items():
            weight = data['weight']
            score = data['score']
            confidence = data['confidence']

            total_score += score * weight
            total_confidence += confidence * weight
            total_weight += weight

            # Track dominant side
            if abs(score) > abs(total_score * 0.3):  # Significant contributor
                dominant_side = data['side']

        if total_weight == 0:
            return None

        # Normalize
        avg_score = total_score / total_weight
        avg_confidence = total_confidence / total_weight

        # Determine final side and strength
        if abs(avg_score) < self.config.long_threshold:
            return None

        side = "BUY" if avg_score > 0 else "SELL"
        strength = "STRONG" if abs(avg_score) > self.config.strong_signal_threshold else "MEDIUM"

        # Calculate position size
        position_pct = (self.config.aggressive_position_pct if strength == "STRONG"
                       else self.config.base_position_pct)

        return TradingSignal(
            symbol=snapshot.symbol,
            side=side,
            signal_strength=strength,
            confidence=avg_confidence,
            position_pct=position_pct,
            entry_price=snapshot.mid,
            timestamp=datetime.now(),
            strategy_name=self.name,
            metadata={
                'composite_score': avg_score,
                'multitimeframe_analysis': timeframe_scores,
                'combined_confidence': avg_confidence,
                'strategy_version': self.version
            }
        )

    def _apply_enhanced_risk_management(
        self,
        signal: TradingSignal,
        positions: List[Position],
        snapshot: MarketSnapshot,
        context: Dict[str, Any]
    ) -> TradingSignal:
        """Apply enhanced risk management"""

        # Start with current signal
        enhanced_signal = signal
        risk_adjustments = []

        # 1. Volatility adjustment
        price_history = context.get('price_history', [])
        if len(price_history) >= 20:
            returns = np.diff(price_history) / np.array(price_history[:-1])
            volatility = np.std(returns[-20:])
            if volatility > 0.02:  # High volatility threshold
                vol_adjustment = self.config.volatility_adjustment
                enhanced_signal.position_pct *= vol_adjustment
                risk_adjustments.append(f"volatility_adj={vol_adjustment:.3f}")

        # 2. Correlation penalty
        correlation_penalty = self._calculate_correlation_penalty(
            snapshot.symbol, positions
        )
        if correlation_penalty < 1.0:
            enhanced_signal.position_pct *= correlation_penalty
            risk_adjustments.append(f"correlation_penalty={correlation_penalty:.3f}")

        # 3. Drawdown protection
        if self.current_drawdown > 0.01:  # 1% drawdown threshold
            dd_protection = max(0.5, 1.0 - self.current_drawdown * 5)
            enhanced_signal.position_pct *= dd_protection
            risk_adjustments.append(f"drawdown_protection={dd_protection:.3f}")

        # 4. Position size limits
        if enhanced_signal.position_pct > self.config.max_single_position:
            enhanced_signal.position_pct = self.config.max_single_position
            risk_adjustments.append(f"max_position_limit={self.config.max_single_position}")

        # Update metadata
        enhanced_signal.metadata.update({
            'risk_adjustments': risk_adjustments,
            'original_position_pct': signal.position_pct,
            'final_position_pct': enhanced_signal.position_pct
        })

        return enhanced_signal

    def _calculate_correlation_penalty(
        self,
        symbol: str,
        positions: List[Position]
    ) -> float:
        """Calculate penalty based on portfolio correlations"""

        if not positions or symbol not in self.correlation_matrix:
            return 1.0

        penalties = []
        for position in positions:
            pos_symbol = position.symbol
            if pos_symbol in self.correlation_matrix.get(symbol, {}):
                correlation = abs(self.correlation_matrix[symbol][pos_symbol])
                if correlation > self.config.max_correlation:
                    penalty = self.config.correlation_penalty
                    penalties.append(penalty)

        if penalties:
            return min(penalties)
        return 1.0

    def _optimize_for_portfolio(
        self,
        signal: TradingSignal,
        positions: List[Position],
        snapshot: MarketSnapshot
    ) -> TradingSignal:
        """Optimize signal for overall portfolio construction"""

        # Portfolio diversification check
        current_symbols = set(pos.symbol for pos in positions)
        if len(current_symbols) >= 5 and signal.symbol not in current_symbols:
            # Favor diversification
            signal.position_pct *= 1.1

        # Sector/asset class balance (simplified)
        crypto_positions = [p for p in positions if 'USDT' in p.symbol or 'USD' in p.symbol]
        if len(crypto_positions) >= 3 and 'USDT' in signal.symbol:
            # Reduce crypto exposure
            signal.position_pct *= 0.8

        # Risk budget allocation
        used_budget = sum(abs(pos.notional_value) for pos in positions) / 10000  # Assuming 10k base
        remaining_budget = self.config.max_portfolio_heat - used_budget

        if signal.position_pct > remaining_budget:
            signal.position_pct = remaining_budget * 0.9  # Leave some buffer

        return signal

    def _final_signal_validation(
        self,
        signal: TradingSignal,
        positions: List[Position],
        snapshot: MarketSnapshot,
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Final validation before signal emission"""

        # Minimum position size check
        if signal.position_pct < 0.005:  # 0.5% minimum
            self.logger.debug(f"Position too small: {signal.position_pct:.4f}")
            return None

        # Signal quality check
        if signal.confidence < 0.6:
            self.logger.debug(f"Signal confidence too low: {signal.confidence:.3f}")
            return None

        # Market condition check
        price_history = context.get('price_history', [])
        if len(price_history) >= 10:
            recent_volatility = np.std(np.diff(price_history[-10:]) / np.array(price_history[-11:-1]))
            if recent_volatility > 0.05:  # Very high volatility
                self.logger.debug(f"Market too volatile: {recent_volatility:.4f}")
                return None

        # Add final metadata
        signal.metadata.update({
            'final_validation_passed': True,
            'validation_timestamp': datetime.now().isoformat(),
            'refined_strategy': True
        })

        return signal

    def _update_correlations(self, snapshot: MarketSnapshot, context: Dict[str, Any]):
        """Update correlation matrix for portfolio management"""

        symbol = snapshot.symbol
        price_history = context.get('price_history', [])

        if len(price_history) < self.config.correlation_lookback:
            return

        # Store price history
        self.price_histories[symbol] = price_history[-self.config.correlation_lookback:]

        # Update correlations with other symbols
        if symbol not in self.correlation_matrix:
            self.correlation_matrix[symbol] = {}

        for other_symbol, other_prices in self.price_histories.items():
            if other_symbol == symbol or len(other_prices) < self.config.correlation_lookback:
                continue

            # Calculate correlation
            try:
                correlation, _ = pearsonr(
                    self.price_histories[symbol],
                    other_prices[-len(self.price_histories[symbol]):]
                )
                self.correlation_matrix[symbol][other_symbol] = correlation

                # Mirror the correlation
                if other_symbol not in self.correlation_matrix:
                    self.correlation_matrix[other_symbol] = {}
                self.correlation_matrix[other_symbol][symbol] = correlation

            except Exception as e:
                self.logger.debug(f"Error calculating correlation {symbol}-{other_symbol}: {e}")

    def update_drawdown(self, current_equity: float):
        """Update drawdown tracking"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get enhanced strategy state"""
        base_state = super().get_strategy_state()

        base_state.update({
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity,
            'risk_budget_used': self.risk_budget_used,
            'correlation_symbols': list(self.correlation_matrix.keys()),
            'attribution_data': self.attribution_data,
            'enhanced_features': [
                'multi_timeframe_analysis',
                'correlation_management',
                'enhanced_risk_controls',
                'portfolio_optimization'
            ]
        })

        return base_state

    def get_portfolio_analytics(self) -> Dict[str, Any]:
        """Get portfolio analytics specific to 1.6pp-R"""
        return {
            'correlation_matrix': self.correlation_matrix,
            'risk_metrics': {
                'current_drawdown': self.current_drawdown,
                'peak_equity': self.peak_equity,
                'risk_budget_used': self.risk_budget_used
            },
            'performance_attribution': self.attribution_data,
            'timeframe_analysis': self.timeframe_signals
        }