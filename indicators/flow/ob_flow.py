#!/usr/bin/env python3
"""
OB-Flow - Direção de Fluxo Agressor
Análise da direção e intensidade do fluxo de ordens agressoras

Detecta:
- Buy/Sell pressure através de trades
- Volume-weighted flow direction
- Intensidade do fluxo agressor
- Mudanças na microestrutura de trading

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
class TradeData:
    """Dados de uma trade individual"""
    price: float
    size: float
    timestamp: datetime
    side: str  # 'buy', 'sell', or 'unknown'
    is_aggressor: bool = True


@dataclass
class OBFlowConfig:
    """Configuração do OB-Flow"""

    # Time windows for analysis
    fast_window_seconds: int = 30
    medium_window_seconds: int = 120
    slow_window_seconds: int = 300

    # Flow calculation parameters
    volume_threshold: float = 0.1  # Minimum volume percentage to consider
    price_impact_weight: float = 0.4
    volume_weight: float = 0.6

    # Aggressor detection
    tick_size: float = 0.01
    aggressor_threshold: float = 0.6  # Confidence threshold for side determination

    # Flow intensity levels
    weak_flow_threshold: float = 0.2
    medium_flow_threshold: float = 0.5
    strong_flow_threshold: float = 0.8

    # Smoothing parameters
    ema_alpha_fast: float = 0.3
    ema_alpha_slow: float = 0.1

    # Performance parameters
    max_trades_history: int = 1000
    calculation_timeout_ms: float = 2.0

    # Noise filtering
    min_trade_size: float = 0.001
    max_spread_percentage: float = 0.01  # 1% max spread to consider


class OBFlow:
    """
    OB-Flow - Análise de Direção de Fluxo Agressor

    Funcionalidades:
    1. Detecção de lado agressor (buy/sell)
    2. Análise de intensidade de fluxo
    3. Volume-weighted direction
    4. Multi-timeframe flow analysis
    """

    def __init__(self, config: Optional[OBFlowConfig] = None):
        self.config = config or OBFlowConfig()
        self.name = "OB-Flow"
        self.version = "1.0.0"

        # Trade history storage
        self.trade_history: deque = deque(maxlen=self.config.max_trades_history)

        # Flow state tracking
        self.current_flow_direction: float = 0.0
        self.flow_intensity: float = 0.0
        self.last_analysis_time: Optional[datetime] = None

        # EMA smoothed values
        self.ema_fast: float = 0.0
        self.ema_slow: float = 0.0

        # Performance metrics
        self.calculation_count: int = 0
        self.total_calculation_time: float = 0.0

        # Market state
        self.last_snapshot: Optional[MarketSnapshot] = None

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"OB-Flow initialized with {self.config.fast_window_seconds}s/{self.config.medium_window_seconds}s/{self.config.slow_window_seconds}s windows")

    @timeit
    def calculate(
        self,
        snapshot: MarketSnapshot,
        trade_data: Optional[List[Dict[str, Any]]] = None,
        orderbook_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calcular OB-Flow baseado em dados de trades e orderbook

        Args:
            snapshot: Dados básicos de mercado
            trade_data: Lista de trades recentes
            orderbook_data: Dados do orderbook para contexto

        Returns:
            Dictionary com flow metrics
        """
        start_time = time.time()

        try:
            # Update market state
            self.last_snapshot = snapshot
            current_time = datetime.now()

            # Process new trade data
            if trade_data:
                new_trades = self._process_trade_data(trade_data, snapshot, current_time)
                for trade in new_trades:
                    self.trade_history.append(trade)

            # If no recent trades, use snapshot to infer flow
            if not self.trade_history:
                return self._fallback_flow_analysis(snapshot)

            # Clean old trades
            self._clean_old_trades(current_time)

            # Calculate flow for different timeframes
            fast_flow = self._calculate_timeframe_flow(self.config.fast_window_seconds, current_time)
            medium_flow = self._calculate_timeframe_flow(self.config.medium_window_seconds, current_time)
            slow_flow = self._calculate_timeframe_flow(self.config.slow_window_seconds, current_time)

            # Update EMA smoothed values
            self._update_ema_values(fast_flow['flow_direction'])

            # Calculate flow intensity and momentum
            flow_intensity = self._calculate_flow_intensity(fast_flow, medium_flow, slow_flow)
            flow_momentum = self._calculate_flow_momentum(fast_flow, medium_flow, slow_flow)

            # Detect flow regime changes
            regime_change = self._detect_flow_regime_change(fast_flow['flow_direction'])

            # Calculate composite flow score
            composite_flow = self._calculate_composite_flow_score(
                fast_flow, medium_flow, slow_flow, flow_intensity, flow_momentum
            )

            # Performance tracking
            execution_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time)

            result = {
                'flow_direction': composite_flow,
                'flow_intensity': flow_intensity,
                'flow_momentum': flow_momentum,
                'fast_flow': fast_flow['flow_direction'],
                'medium_flow': medium_flow['flow_direction'],
                'slow_flow': slow_flow['flow_direction'],
                'buy_pressure': fast_flow['buy_pressure'],
                'sell_pressure': fast_flow['sell_pressure'],
                'volume_imbalance': fast_flow['volume_imbalance'],
                'trade_count': len([t for t in self.trade_history if (current_time - t.timestamp).total_seconds() <= self.config.fast_window_seconds]),
                'regime_change': regime_change,
                'ema_fast': self.ema_fast,
                'ema_slow': self.ema_slow,
                'execution_time_ms': execution_time,
                'calculation_quality': 'high' if execution_time < self.config.calculation_timeout_ms else 'degraded'
            }

            self.last_analysis_time = current_time
            return result

        except Exception as e:
            self.logger.error(f"Error calculating OB-Flow: {str(e)}")
            execution_time = (time.time() - start_time) * 1000
            return {
                'flow_direction': 0.0,
                'flow_intensity': 0.0,
                'error': str(e),
                'execution_time_ms': execution_time,
                'calculation_quality': 'error'
            }

    def _process_trade_data(
        self,
        trade_data: List[Dict[str, Any]],
        snapshot: MarketSnapshot,
        current_time: datetime
    ) -> List[TradeData]:
        """Process raw trade data into structured format"""

        trades = []

        try:
            for trade_raw in trade_data:
                # Extract trade information
                price = float(trade_raw.get('price', 0))
                size = float(trade_raw.get('size', 0))
                timestamp_raw = trade_raw.get('timestamp')

                # Parse timestamp
                if isinstance(timestamp_raw, str):
                    timestamp = datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
                elif isinstance(timestamp_raw, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp_raw / 1000 if timestamp_raw > 1e10 else timestamp_raw)
                else:
                    timestamp = current_time

                # Filter out noise
                if size < self.config.min_trade_size:
                    continue

                # Determine aggressor side
                side = self._determine_aggressor_side(trade_raw, snapshot, price)

                trade = TradeData(
                    price=price,
                    size=size,
                    timestamp=timestamp,
                    side=side,
                    is_aggressor=True
                )

                trades.append(trade)

        except Exception as e:
            self.logger.error(f"Error processing trade data: {str(e)}")

        return trades

    def _determine_aggressor_side(
        self,
        trade_raw: Dict[str, Any],
        snapshot: MarketSnapshot,
        price: float
    ) -> str:
        """Determine if trade was initiated by buyer or seller"""

        # Method 1: Explicit side from trade data
        if 'side' in trade_raw:
            return trade_raw['side'].lower()

        # Method 2: Price comparison with bid/ask
        if hasattr(snapshot, 'bid') and hasattr(snapshot, 'ask'):
            mid_price = (snapshot.bid + snapshot.ask) / 2
            if price >= mid_price:
                return 'buy'
            else:
                return 'sell'

        # Method 3: Price movement analysis
        if self.last_snapshot:
            if price > self.last_snapshot.mid:
                return 'buy'
            elif price < self.last_snapshot.mid:
                return 'sell'

        # Method 4: Use tick rule (price compared to previous trades)
        recent_trades = list(self.trade_history)[-10:] if self.trade_history else []
        if recent_trades:
            avg_recent_price = np.mean([t.price for t in recent_trades])
            if price > avg_recent_price + self.config.tick_size:
                return 'buy'
            elif price < avg_recent_price - self.config.tick_size:
                return 'sell'

        return 'unknown'

    def _calculate_timeframe_flow(self, window_seconds: int, current_time: datetime) -> Dict[str, float]:
        """Calculate flow metrics for specific timeframe"""

        cutoff_time = current_time - timedelta(seconds=window_seconds)

        # Filter trades in timeframe
        relevant_trades = [
            t for t in self.trade_history
            if t.timestamp >= cutoff_time and t.side != 'unknown'
        ]

        if not relevant_trades:
            return {
                'flow_direction': 0.0,
                'buy_pressure': 0.0,
                'sell_pressure': 0.0,
                'volume_imbalance': 0.0,
                'trade_count': 0
            }

        # Calculate buy/sell volumes and counts
        buy_volume = sum(t.size for t in relevant_trades if t.side == 'buy')
        sell_volume = sum(t.size for t in relevant_trades if t.side == 'sell')
        buy_count = len([t for t in relevant_trades if t.side == 'buy'])
        sell_count = len([t for t in relevant_trades if t.side == 'sell'])

        total_volume = buy_volume + sell_volume
        total_count = buy_count + sell_count

        # Calculate pressures
        buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5
        sell_pressure = sell_volume / total_volume if total_volume > 0 else 0.5

        # Volume-weighted flow direction
        if total_volume > 0:
            volume_flow = (buy_volume - sell_volume) / total_volume
        else:
            volume_flow = 0.0

        # Count-weighted flow (retail vs institutional indication)
        if total_count > 0:
            count_flow = (buy_count - sell_count) / total_count
        else:
            count_flow = 0.0

        # Combine volume and count flows
        flow_direction = (
            volume_flow * self.config.volume_weight +
            count_flow * (1 - self.config.volume_weight)
        )

        # Calculate volume imbalance
        volume_imbalance = abs(buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0

        return {
            'flow_direction': np.clip(flow_direction, -1.0, 1.0),
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'volume_imbalance': volume_imbalance,
            'trade_count': len(relevant_trades),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def _calculate_flow_intensity(
        self,
        fast_flow: Dict[str, float],
        medium_flow: Dict[str, float],
        slow_flow: Dict[str, float]
    ) -> float:
        """Calculate overall flow intensity"""

        # Use volume imbalance and trade count as intensity indicators
        fast_intensity = fast_flow['volume_imbalance']
        medium_intensity = medium_flow['volume_imbalance']

        # Trade count intensity (normalized)
        fast_trade_count = fast_flow['trade_count']
        expected_trades_per_window = self.config.fast_window_seconds / 10  # Assume 1 trade per 10s baseline
        trade_intensity = min(1.0, fast_trade_count / expected_trades_per_window) if expected_trades_per_window > 0 else 0.0

        # Combine intensities
        intensity = (
            fast_intensity * 0.5 +
            medium_intensity * 0.3 +
            trade_intensity * 0.2
        )

        return np.clip(intensity, 0.0, 1.0)

    def _calculate_flow_momentum(
        self,
        fast_flow: Dict[str, float],
        medium_flow: Dict[str, float],
        slow_flow: Dict[str, float]
    ) -> float:
        """Calculate flow momentum (rate of change)"""

        fast_dir = fast_flow['flow_direction']
        medium_dir = medium_flow['flow_direction']
        slow_dir = slow_flow['flow_direction']

        # Calculate momentum as acceleration of flow
        if abs(medium_dir) > 1e-6:
            fast_momentum = (fast_dir - medium_dir) / abs(medium_dir)
        else:
            fast_momentum = fast_dir

        if abs(slow_dir) > 1e-6:
            medium_momentum = (medium_dir - slow_dir) / abs(slow_dir)
        else:
            medium_momentum = medium_dir

        # Combined momentum
        momentum = fast_momentum * 0.7 + medium_momentum * 0.3

        return np.clip(momentum, -2.0, 2.0)

    def _update_ema_values(self, current_flow: float):
        """Update EMA smoothed flow values"""

        if self.ema_fast == 0.0:  # First calculation
            self.ema_fast = current_flow
            self.ema_slow = current_flow
        else:
            self.ema_fast = (
                self.config.ema_alpha_fast * current_flow +
                (1 - self.config.ema_alpha_fast) * self.ema_fast
            )
            self.ema_slow = (
                self.config.ema_alpha_slow * current_flow +
                (1 - self.config.ema_alpha_slow) * self.ema_slow
            )

    def _detect_flow_regime_change(self, current_flow: float) -> bool:
        """Detect significant changes in flow regime"""

        if not self.last_analysis_time:
            return False

        # Compare with EMA values
        ema_diff = abs(self.ema_fast - self.ema_slow)
        current_diff = abs(current_flow - self.ema_slow)

        # Regime change if current flow significantly deviates from slow EMA
        regime_change = current_diff > (ema_diff * 2 + 0.3)

        if regime_change:
            direction = "bullish" if current_flow > self.ema_slow else "bearish"
            self.logger.info(f"Flow regime change detected: {direction} (flow: {current_flow:.3f}, ema_slow: {self.ema_slow:.3f})")

        return regime_change

    def _calculate_composite_flow_score(
        self,
        fast_flow: Dict[str, float],
        medium_flow: Dict[str, float],
        slow_flow: Dict[str, float],
        intensity: float,
        momentum: float
    ) -> float:
        """Calculate final composite flow score"""

        # Weighted combination of timeframes
        timeframe_score = (
            fast_flow['flow_direction'] * 0.5 +
            medium_flow['flow_direction'] * 0.3 +
            slow_flow['flow_direction'] * 0.2
        )

        # Apply intensity and momentum adjustments
        intensity_adjusted = timeframe_score * (0.5 + intensity * 0.5)
        momentum_adjusted = intensity_adjusted + momentum * 0.1

        return np.clip(momentum_adjusted, -1.0, 1.0)

    def _clean_old_trades(self, current_time: datetime):
        """Remove trades older than the longest analysis window"""

        cutoff_time = current_time - timedelta(seconds=self.config.slow_window_seconds * 2)

        # Remove old trades
        trades_to_keep = []
        for trade in self.trade_history:
            if trade.timestamp >= cutoff_time:
                trades_to_keep.append(trade)

        self.trade_history.clear()
        self.trade_history.extend(trades_to_keep)

    def _fallback_flow_analysis(self, snapshot: MarketSnapshot) -> Dict[str, float]:
        """Fallback analysis when no trade data is available"""

        # Use spread and price movement as proxy
        spread_ratio = snapshot.spread / snapshot.mid if snapshot.mid > 0 else 0.0

        # Infer flow from bid/ask sizes if available
        flow_direction = 0.0
        if hasattr(snapshot, 'bid_size') and hasattr(snapshot, 'ask_size'):
            total_size = snapshot.bid_size + snapshot.ask_size
            if total_size > 0:
                flow_direction = (snapshot.bid_size - snapshot.ask_size) / total_size

        # Price movement flow (if we have previous snapshot)
        if self.last_snapshot:
            price_change = (snapshot.mid - self.last_snapshot.mid) / self.last_snapshot.mid
            flow_direction += np.clip(price_change * 10, -0.5, 0.5)  # Scale and clip

        return {
            'flow_direction': np.clip(flow_direction, -1.0, 1.0),
            'flow_intensity': min(1.0, spread_ratio * 5),  # Higher spread = lower intensity
            'flow_momentum': 0.0,
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'volume_imbalance': 0.0,
            'trade_count': 0,
            'regime_change': False,
            'ema_fast': flow_direction,
            'ema_slow': flow_direction,
            'calculation_quality': 'fallback'
        }

    def _update_performance_metrics(self, execution_time: float):
        """Update performance tracking"""
        self.calculation_count += 1
        self.total_calculation_time += execution_time

        if self.calculation_count % 50 == 0:
            avg_time = self.total_calculation_time / self.calculation_count
            self.logger.info(f"OB-Flow performance: {avg_time:.2f}ms avg over {self.calculation_count} calculations")

    def get_flow_interpretation(self, flow_direction: float, intensity: float) -> Dict[str, str]:
        """Get human-readable interpretation of flow"""

        abs_flow = abs(flow_direction)
        direction = "Bullish" if flow_direction > 0 else "Bearish" if flow_direction < 0 else "Neutral"

        # Determine strength based on both direction and intensity
        combined_strength = abs_flow * intensity

        if combined_strength >= self.config.strong_flow_threshold:
            strength = "Strong"
        elif combined_strength >= self.config.medium_flow_threshold:
            strength = "Medium"
        elif combined_strength >= self.config.weak_flow_threshold:
            strength = "Weak"
        else:
            strength = "Minimal"

        return {
            'direction': direction,
            'strength': strength,
            'intensity_level': 'High' if intensity > 0.7 else 'Medium' if intensity > 0.4 else 'Low',
            'description': f"{strength} {direction.lower()} flow with {intensity_level.lower()} intensity",
            'numeric_flow': flow_direction,
            'numeric_intensity': intensity,
            'confidence': min(0.99, combined_strength)
        }

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
            'trade_history_count': len(self.trade_history),
            'current_ema_fast': self.ema_fast,
            'current_ema_slow': self.ema_slow
        }