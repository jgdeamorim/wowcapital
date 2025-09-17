#!/usr/bin/env python3
"""
Squeeze-σ - Compressão e Expansão de Volatilidade
Detecta períodos de compressão de volatilidade seguidos de expansão

Identifica:
- Períodos de baixa volatilidade (squeeze)
- Iminência de breakouts
- Direção provável da expansão
- Força da compressão/expansão

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import numpy as np
from datetime import datetime, timedelta

from core.contracts import MarketSnapshot
from utils.performance import timeit


@dataclass
class SqueezeSigmaConfig:
    """Configuração do Squeeze-σ"""

    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # Keltner Channel parameters
    kc_period: int = 20
    kc_atr_multiplier: float = 1.5

    # Squeeze detection
    squeeze_threshold: float = 0.95  # BB must be X% inside KC
    min_squeeze_duration: int = 5    # Minimum bars in squeeze
    max_squeeze_duration: int = 50   # Maximum bars before forced expansion

    # Volatility measurement
    volatility_lookback: int = 14
    volatility_ema_alpha: float = 0.1

    # Expansion detection
    expansion_threshold: float = 1.05  # BB must be X% outside KC
    expansion_momentum_weight: float = 0.7

    # Momentum oscillator (for direction prediction)
    momentum_period: int = 12
    signal_line_period: int = 9

    # Alert system
    pre_breakout_threshold: float = 0.8  # Warning before breakout
    strong_squeeze_threshold: float = 0.9  # Very tight squeeze

    # Performance parameters
    max_history_bars: int = 200
    calculation_timeout_ms: float = 4.0


@dataclass
class SqueezeState:
    """Estado atual da compressão"""
    is_squeezing: bool = False
    squeeze_strength: float = 0.0
    squeeze_duration: int = 0
    expansion_probability: float = 0.0
    breakout_direction: str = "unknown"  # "up", "down", "unknown"
    breakout_strength: float = 0.0


class SqueezeSigma:
    """
    Squeeze-σ - Análise de Compressão e Expansão de Volatilidade

    Funcionalidades:
    1. Detecção de squeeze (BB dentro de KC)
    2. Previsão de breakouts
    3. Direção do breakout
    4. Sistema de alertas
    """

    def __init__(self, config: Optional[SqueezeSigmaConfig] = None):
        self.config = config or SqueezeSigmaConfig()
        self.name = "Squeeze-σ"
        self.version = "1.0.0"

        # Historical data
        self.price_history: deque = deque(maxlen=self.config.max_history_bars)
        self.high_history: deque = deque(maxlen=self.config.max_history_bars)
        self.low_history: deque = deque(maxlen=self.config.max_history_bars)
        self.volume_history: deque = deque(maxlen=self.config.max_history_bars)

        # Calculated indicators
        self.bb_upper: deque = deque(maxlen=100)
        self.bb_middle: deque = deque(maxlen=100)
        self.bb_lower: deque = deque(maxlen=100)
        self.kc_upper: deque = deque(maxlen=100)
        self.kc_middle: deque = deque(maxlen=100)
        self.kc_lower: deque = deque(maxlen=100)

        # Squeeze state tracking
        self.current_state = SqueezeState()
        self.state_history: deque = deque(maxlen=100)

        # Momentum tracking
        self.momentum_values: deque = deque(maxlen=50)
        self.signal_line_values: deque = deque(maxlen=50)

        # Volatility tracking
        self.volatility_ema: float = 0.0
        self.volatility_history: deque = deque(maxlen=50)

        # Performance metrics
        self.calculation_count: int = 0
        self.total_calculation_time: float = 0.0

        # Alert system
        self.last_alert_time: Optional[datetime] = None
        self.alert_cooldown_seconds: int = 60

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Squeeze-σ initialized with {self.config.bb_period}/{self.config.kc_period} periods")

    @timeit
    def calculate(
        self,
        snapshot: MarketSnapshot,
        price_data: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Calcular análise de Squeeze-σ

        Args:
            snapshot: Dados atuais de mercado
            price_data: Histórico de preços (keys: 'prices', 'highs', 'lows', 'volumes')

        Returns:
            Dictionary com análise completa
        """
        start_time = time.time()

        try:
            # Update price history
            self._update_price_history(snapshot, price_data)

            # Check if we have enough data
            if len(self.price_history) < max(self.config.bb_period, self.config.kc_period):
                return self._insufficient_data_result()

            # Calculate Bollinger Bands
            bb_data = self._calculate_bollinger_bands()

            # Calculate Keltner Channels
            kc_data = self._calculate_keltner_channels()

            # Detect squeeze condition
            squeeze_data = self._detect_squeeze(bb_data, kc_data)

            # Calculate momentum for direction prediction
            momentum_data = self._calculate_momentum()

            # Update squeeze state
            self._update_squeeze_state(squeeze_data, momentum_data)

            # Calculate expansion probability
            expansion_prob = self._calculate_expansion_probability()

            # Generate alerts if needed
            alerts = self._generate_alerts()

            # Calculate volatility metrics
            volatility_data = self._calculate_volatility_metrics()

            # Performance tracking
            execution_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time)

            result = {
                # Core squeeze metrics
                'is_squeezing': self.current_state.is_squeezing,
                'squeeze_strength': self.current_state.squeeze_strength,
                'squeeze_duration': self.current_state.squeeze_duration,
                'expansion_probability': expansion_prob,

                # Breakout prediction
                'breakout_direction': self.current_state.breakout_direction,
                'breakout_strength': self.current_state.breakout_strength,
                'pre_breakout_warning': expansion_prob > self.config.pre_breakout_threshold,

                # Technical indicators
                'bb_upper': bb_data['upper'],
                'bb_middle': bb_data['middle'],
                'bb_lower': bb_data['lower'],
                'bb_width': bb_data['width'],
                'kc_upper': kc_data['upper'],
                'kc_middle': kc_data['middle'],
                'kc_lower': kc_data['lower'],
                'kc_width': kc_data['width'],

                # Momentum and direction
                'momentum': momentum_data['momentum'],
                'signal_line': momentum_data['signal_line'],
                'momentum_direction': momentum_data['direction'],

                # Volatility analysis
                'current_volatility': volatility_data['current'],
                'volatility_ema': self.volatility_ema,
                'volatility_percentile': volatility_data['percentile'],

                # Alerts and warnings
                'alerts': alerts,
                'alert_level': self._get_alert_level(expansion_prob),

                # Metadata
                'execution_time_ms': execution_time,
                'data_points': len(self.price_history),
                'calculation_quality': 'high' if execution_time < self.config.calculation_timeout_ms else 'degraded'
            }

            # Store state in history
            self.state_history.append({
                'timestamp': datetime.now(),
                'state': self.current_state,
                'result': result.copy()
            })

            return result

        except Exception as e:
            self.logger.error(f"Error calculating Squeeze-σ: {str(e)}")
            execution_time = (time.time() - start_time) * 1000
            return {
                'is_squeezing': False,
                'squeeze_strength': 0.0,
                'error': str(e),
                'execution_time_ms': execution_time,
                'calculation_quality': 'error'
            }

    def _update_price_history(self, snapshot: MarketSnapshot, price_data: Optional[Dict[str, List[float]]]):
        """Update internal price history"""

        if price_data:
            # Use provided historical data
            prices = price_data.get('prices', [])
            highs = price_data.get('highs', [])
            lows = price_data.get('lows', [])
            volumes = price_data.get('volumes', [])

            # Replace history if we have new data
            if prices:
                self.price_history.clear()
                self.price_history.extend(prices[-self.config.max_history_bars:])

            if highs:
                self.high_history.clear()
                self.high_history.extend(highs[-self.config.max_history_bars:])

            if lows:
                self.low_history.clear()
                self.low_history.extend(lows[-self.config.max_history_bars:])

            if volumes:
                self.volume_history.clear()
                self.volume_history.extend(volumes[-self.config.max_history_bars:])

        # Always append current snapshot
        self.price_history.append(snapshot.mid)

        # Use snapshot values if no separate high/low data
        if not self.high_history or len(self.high_history) < len(self.price_history):
            self.high_history.append(snapshot.mid)

        if not self.low_history or len(self.low_history) < len(self.price_history):
            self.low_history.append(snapshot.mid)

        # Use bid/ask sizes as volume proxy if available
        volume = getattr(snapshot, 'volume', 0) or (getattr(snapshot, 'bid_size', 0) + getattr(snapshot, 'ask_size', 0))
        self.volume_history.append(volume)

    def _calculate_bollinger_bands(self) -> Dict[str, float]:
        """Calculate Bollinger Bands"""

        prices = np.array(list(self.price_history))
        period = self.config.bb_period

        if len(prices) < period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'width': 0.0}

        # Calculate SMA and standard deviation
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:]) * self.config.bb_std_dev

        upper = sma + std
        middle = sma
        lower = sma - std
        width = (upper - lower) / middle if middle > 0 else 0.0

        # Store in history
        self.bb_upper.append(upper)
        self.bb_middle.append(middle)
        self.bb_lower.append(lower)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'std': std / middle if middle > 0 else 0.0
        }

    def _calculate_keltner_channels(self) -> Dict[str, float]:
        """Calculate Keltner Channels"""

        if len(self.price_history) < self.config.kc_period:
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'width': 0.0}

        prices = np.array(list(self.price_history))
        highs = np.array(list(self.high_history))
        lows = np.array(list(self.low_history))

        period = self.config.kc_period

        # Calculate EMA of prices (middle line)
        ema = self._calculate_ema(prices, period)

        # Calculate ATR (Average True Range)
        atr = self._calculate_atr(prices, highs, lows, period)

        # Keltner Channel lines
        upper = ema + (atr * self.config.kc_atr_multiplier)
        middle = ema
        lower = ema - (atr * self.config.kc_atr_multiplier)
        width = (upper - lower) / middle if middle > 0 else 0.0

        # Store in history
        self.kc_upper.append(upper)
        self.kc_middle.append(middle)
        self.kc_lower.append(lower)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'atr': atr
        }

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""

        if len(prices) < period:
            return np.mean(prices)

        # Use standard EMA calculation
        alpha = 2.0 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _calculate_atr(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int) -> float:
        """Calculate Average True Range"""

        if len(prices) < 2:
            return 0.0

        # Ensure all arrays have same length
        min_len = min(len(prices), len(highs), len(lows))
        prices = prices[-min_len:]
        highs = highs[-min_len:]
        lows = lows[-min_len:]

        true_ranges = []

        for i in range(1, len(prices)):
            # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - prices[i-1])
            tr3 = abs(lows[i] - prices[i-1])

            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)

        if not true_ranges:
            return 0.0

        # Return average of last 'period' true ranges
        return np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.mean(true_ranges)

    def _detect_squeeze(self, bb_data: Dict[str, float], kc_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect squeeze condition (BB inside KC)"""

        bb_upper, bb_lower = bb_data['upper'], bb_data['lower']
        kc_upper, kc_lower = kc_data['upper'], kc_data['lower']

        # Check if Bollinger Bands are inside Keltner Channels
        upper_inside = bb_upper <= kc_upper * self.config.squeeze_threshold
        lower_inside = bb_lower >= kc_lower * (2 - self.config.squeeze_threshold)

        is_squeezing = upper_inside and lower_inside

        # Calculate squeeze strength (how tight is the squeeze)
        if kc_upper != kc_lower:
            bb_range = bb_upper - bb_lower
            kc_range = kc_upper - kc_lower
            squeeze_strength = 1.0 - (bb_range / kc_range) if kc_range > 0 else 0.0
        else:
            squeeze_strength = 0.0

        squeeze_strength = np.clip(squeeze_strength, 0.0, 1.0)

        return {
            'is_squeezing': is_squeezing,
            'squeeze_strength': squeeze_strength,
            'bb_inside_kc_ratio': (bb_upper - bb_lower) / (kc_upper - kc_lower) if (kc_upper - kc_lower) > 0 else 1.0,
            'upper_inside': upper_inside,
            'lower_inside': lower_inside
        }

    def _calculate_momentum(self) -> Dict[str, Any]:
        """Calculate momentum oscillator for direction prediction"""

        if len(self.price_history) < self.config.momentum_period:
            return {'momentum': 0.0, 'signal_line': 0.0, 'direction': 'unknown'}

        prices = np.array(list(self.price_history))

        # Calculate momentum as linear regression slope
        period = self.config.momentum_period
        x = np.arange(period)
        recent_prices = prices[-period:]

        # Linear regression
        if len(recent_prices) == period:
            momentum = np.polyfit(x, recent_prices, 1)[0]  # Slope
        else:
            momentum = 0.0

        # Store momentum
        self.momentum_values.append(momentum)

        # Calculate signal line (EMA of momentum)
        if len(self.momentum_values) >= self.config.signal_line_period:
            signal_line = self._calculate_ema(
                np.array(list(self.momentum_values)),
                self.config.signal_line_period
            )
        else:
            signal_line = momentum

        self.signal_line_values.append(signal_line)

        # Determine direction
        if momentum > signal_line * 1.1:
            direction = "up"
        elif momentum < signal_line * 0.9:
            direction = "down"
        else:
            direction = "sideways"

        return {
            'momentum': momentum,
            'signal_line': signal_line,
            'direction': direction,
            'strength': abs(momentum - signal_line) / (abs(signal_line) + 1e-8)
        }

    def _update_squeeze_state(self, squeeze_data: Dict[str, Any], momentum_data: Dict[str, Any]):
        """Update current squeeze state"""

        is_squeezing = squeeze_data['is_squeezing']
        squeeze_strength = squeeze_data['squeeze_strength']

        # Update squeeze duration
        if is_squeezing:
            if self.current_state.is_squeezing:
                self.current_state.squeeze_duration += 1
            else:
                self.current_state.squeeze_duration = 1
                self.logger.info("Squeeze started")
        else:
            if self.current_state.is_squeezing and self.current_state.squeeze_duration > 0:
                self.logger.info(f"Squeeze ended after {self.current_state.squeeze_duration} periods")
            self.current_state.squeeze_duration = 0

        # Update state
        self.current_state.is_squeezing = is_squeezing
        self.current_state.squeeze_strength = squeeze_strength

        # Predict breakout direction based on momentum
        momentum_direction = momentum_data['direction']
        if momentum_direction == "up":
            self.current_state.breakout_direction = "up"
        elif momentum_direction == "down":
            self.current_state.breakout_direction = "down"
        else:
            self.current_state.breakout_direction = "unknown"

        # Calculate breakout strength
        self.current_state.breakout_strength = momentum_data['strength']

    def _calculate_expansion_probability(self) -> float:
        """Calculate probability of imminent expansion/breakout"""

        if not self.current_state.is_squeezing:
            return 0.0

        factors = []

        # Duration factor (longer squeeze = higher probability)
        duration_factor = min(1.0, self.current_state.squeeze_duration / 20)
        factors.append(duration_factor * 0.3)

        # Strength factor (tighter squeeze = higher probability)
        strength_factor = self.current_state.squeeze_strength
        factors.append(strength_factor * 0.3)

        # Momentum factor
        momentum_factor = self.current_state.breakout_strength
        factors.append(momentum_factor * 0.2)

        # Volatility factor (low volatility often precedes high volatility)
        if self.volatility_history:
            current_vol = self.volatility_history[-1] if self.volatility_history else 0.0
            avg_vol = np.mean(list(self.volatility_history)) if self.volatility_history else 0.0
            vol_factor = 1.0 - (current_vol / avg_vol) if avg_vol > 0 else 0.0
            vol_factor = np.clip(vol_factor, 0.0, 1.0)
            factors.append(vol_factor * 0.2)

        return sum(factors)

    def _calculate_volatility_metrics(self) -> Dict[str, float]:
        """Calculate current volatility metrics"""

        if len(self.price_history) < self.config.volatility_lookback:
            return {'current': 0.0, 'percentile': 0.5}

        prices = np.array(list(self.price_history))
        returns = np.diff(prices) / prices[:-1]

        # Current volatility
        current_vol = np.std(returns[-self.config.volatility_lookback:])

        # Update EMA volatility
        if self.volatility_ema == 0.0:
            self.volatility_ema = current_vol
        else:
            self.volatility_ema = (
                self.config.volatility_ema_alpha * current_vol +
                (1 - self.config.volatility_ema_alpha) * self.volatility_ema
            )

        # Store in history
        self.volatility_history.append(current_vol)

        # Calculate percentile
        if len(self.volatility_history) >= 20:
            vol_array = np.array(list(self.volatility_history))
            percentile = np.percentile(vol_array, 50)
        else:
            percentile = 0.5

        return {
            'current': current_vol,
            'percentile': percentile / (percentile + current_vol) if (percentile + current_vol) > 0 else 0.5
        }

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on current conditions"""

        alerts = []
        current_time = datetime.now()

        # Check alert cooldown
        if (self.last_alert_time and
            (current_time - self.last_alert_time).total_seconds() < self.alert_cooldown_seconds):
            return alerts

        # Pre-breakout warning
        expansion_prob = self.current_state.expansion_probability
        if expansion_prob > self.config.pre_breakout_threshold:
            alerts.append({
                'type': 'pre_breakout',
                'message': f"Pre-breakout warning: {expansion_prob:.1%} probability",
                'level': 'warning',
                'direction': self.current_state.breakout_direction,
                'timestamp': current_time
            })

        # Strong squeeze alert
        if (self.current_state.is_squeezing and
            self.current_state.squeeze_strength > self.config.strong_squeeze_threshold):
            alerts.append({
                'type': 'strong_squeeze',
                'message': f"Strong squeeze detected: {self.current_state.squeeze_strength:.1%} strength",
                'level': 'info',
                'duration': self.current_state.squeeze_duration,
                'timestamp': current_time
            })

        # Long squeeze alert
        if (self.current_state.is_squeezing and
            self.current_state.squeeze_duration > 30):
            alerts.append({
                'type': 'long_squeeze',
                'message': f"Extended squeeze: {self.current_state.squeeze_duration} periods",
                'level': 'info',
                'timestamp': current_time
            })

        if alerts:
            self.last_alert_time = current_time

        return alerts

    def _get_alert_level(self, expansion_probability: float) -> str:
        """Get alert level based on expansion probability"""

        if expansion_probability > 0.8:
            return "critical"
        elif expansion_probability > 0.6:
            return "warning"
        elif expansion_probability > 0.4:
            return "info"
        else:
            return "none"

    def _insufficient_data_result(self) -> Dict[str, Any]:
        """Return result when insufficient data is available"""

        return {
            'is_squeezing': False,
            'squeeze_strength': 0.0,
            'squeeze_duration': 0,
            'expansion_probability': 0.0,
            'breakout_direction': 'unknown',
            'breakout_strength': 0.0,
            'calculation_quality': 'insufficient_data',
            'data_points': len(self.price_history),
            'required_points': max(self.config.bb_period, self.config.kc_period)
        }

    def _update_performance_metrics(self, execution_time: float):
        """Update performance tracking"""
        self.calculation_count += 1
        self.total_calculation_time += execution_time

        if self.calculation_count % 50 == 0:
            avg_time = self.total_calculation_time / self.calculation_count
            self.logger.info(f"Squeeze-σ performance: {avg_time:.2f}ms avg over {self.calculation_count} calculations")

    def get_squeeze_interpretation(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Get human-readable interpretation"""

        if result['is_squeezing']:
            strength = "Very tight" if result['squeeze_strength'] > 0.8 else "Tight" if result['squeeze_strength'] > 0.5 else "Moderate"
            duration_desc = f"for {result['squeeze_duration']} periods"
            direction = result['breakout_direction'].title() if result['breakout_direction'] != 'unknown' else 'Unknown direction'

            description = f"{strength} squeeze {duration_desc}. {direction} breakout expected."
        else:
            description = "No squeeze detected. Normal volatility conditions."

        return {
            'status': 'Squeezing' if result['is_squeezing'] else 'Normal',
            'description': description,
            'risk_level': self._get_alert_level(result.get('expansion_probability', 0.0)),
            'trading_advice': self._get_trading_advice(result)
        }

    def _get_trading_advice(self, result: Dict[str, Any]) -> str:
        """Get trading advice based on squeeze analysis"""

        if not result['is_squeezing']:
            return "Monitor for new squeeze formation"

        expansion_prob = result.get('expansion_probability', 0.0)
        direction = result.get('breakout_direction', 'unknown')

        if expansion_prob > 0.8:
            return f"High breakout probability. Prepare for {direction} movement."
        elif expansion_prob > 0.6:
            return f"Moderate breakout probability. Watch for {direction} signals."
        else:
            return "Early stage squeeze. Wait for clearer direction signals."

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""

        if self.calculation_count == 0:
            return {'status': 'no_calculations'}

        avg_time = self.total_calculation_time / self.calculation_count

        return {
            'total_calculations': self.calculation_count,
            'average_execution_time_ms': avg_time,
            'target_time_ms': self.config.calculation_timeout_ms,
            'performance_ratio': self.config.calculation_timeout_ms / avg_time,
            'status': 'optimal' if avg_time < self.config.calculation_timeout_ms else 'degraded',
            'price_history_count': len(self.price_history),
            'current_squeeze_duration': self.current_state.squeeze_duration,
            'volatility_ema': self.volatility_ema
        }