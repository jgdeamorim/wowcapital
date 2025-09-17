#!/usr/bin/env python3
"""
OB-Score - Orderbook L2 Analysis
Análise de imbalance e microestrutura do orderbook Level 2

Calcula score baseado em:
- Imbalance bid/ask em diferentes níveis
- Densidade de ordens
- Distribuição de tamanhos
- Pressão de compra/venda

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from core.contracts import MarketSnapshot
from utils.performance import timeit


@dataclass
class OrderbookLevel:
    """Representação de um nível do orderbook"""
    price: float
    size: float
    orders_count: int = 1


@dataclass
class OrderbookData:
    """Dados completos do orderbook L2"""
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: datetime
    symbol: str


@dataclass
class OBScoreConfig:
    """Configuração do OB-Score"""

    # Analysis depth
    max_levels: int = 20
    analysis_depth: int = 10

    # Weighting parameters
    level_decay_factor: float = 0.85
    size_weight: float = 0.6
    count_weight: float = 0.4

    # Imbalance thresholds
    strong_imbalance_threshold: float = 0.7
    medium_imbalance_threshold: float = 0.4
    weak_imbalance_threshold: float = 0.15

    # Time-based parameters
    snapshot_window: int = 5  # seconds
    decay_half_life: float = 2.0  # seconds

    # Performance targets
    calculation_timeout_ms: float = 3.0

    def __post_init__(self):
        if self.analysis_depth > self.max_levels:
            self.analysis_depth = self.max_levels


class OBScore:
    """
    Orderbook Score - Análise de microestrutura L2

    Funcionalidades:
    1. Imbalance Analysis por nível
    2. Densidade de liquidez
    3. Pressão de compra/venda
    4. Score composto weighted
    """

    def __init__(self, config: Optional[OBScoreConfig] = None):
        self.config = config or OBScoreConfig()
        self.name = "OB-Score"
        self.version = "1.0.0"

        # Historical data for time-weighted analysis
        self.orderbook_history: List[OrderbookData] = []
        self.score_history: List[Dict[str, float]] = []

        # Performance tracking
        self.calculation_count: int = 0
        self.total_calculation_time: float = 0.0

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"OB-Score initialized with analysis depth: {self.config.analysis_depth}")

    @timeit
    def calculate(
        self,
        snapshot: MarketSnapshot,
        orderbook_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calcular OB-Score baseado no orderbook L2

        Args:
            snapshot: Dados básicos de mercado
            orderbook_data: Dados L2 do orderbook

        Returns:
            Dictionary com scores e métricas
        """
        start_time = time.time()

        try:
            # Parse orderbook data
            if not orderbook_data:
                return self._fallback_score(snapshot)

            orderbook = self._parse_orderbook_data(orderbook_data, snapshot)
            if not orderbook:
                return self._fallback_score(snapshot)

            # Store historical data
            self._update_history(orderbook)

            # Calculate individual components
            imbalance_score = self._calculate_imbalance_score(orderbook)
            density_score = self._calculate_density_score(orderbook)
            pressure_score = self._calculate_pressure_score(orderbook)
            depth_score = self._calculate_depth_score(orderbook)

            # Time-weighted analysis
            time_weighted_score = self._calculate_time_weighted_score()

            # Composite score
            composite_score = self._calculate_composite_score(
                imbalance_score,
                density_score,
                pressure_score,
                depth_score,
                time_weighted_score
            )

            # Performance tracking
            execution_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time)

            result = {
                'ob_score': composite_score,
                'imbalance_score': imbalance_score,
                'density_score': density_score,
                'pressure_score': pressure_score,
                'depth_score': depth_score,
                'time_weighted_score': time_weighted_score,
                'execution_time_ms': execution_time,
                'levels_analyzed': len(orderbook.bids) + len(orderbook.asks),
                'calculation_quality': 'high' if execution_time < self.config.calculation_timeout_ms else 'degraded'
            }

            # Store in history
            self.score_history.append(result)
            if len(self.score_history) > 100:
                self.score_history = self.score_history[-100:]

            return result

        except Exception as e:
            self.logger.error(f"Error calculating OB-Score: {str(e)}")
            execution_time = (time.time() - start_time) * 1000
            return {
                'ob_score': 0.0,
                'error': str(e),
                'execution_time_ms': execution_time,
                'calculation_quality': 'error'
            }

    def _parse_orderbook_data(
        self,
        orderbook_data: Dict[str, Any],
        snapshot: MarketSnapshot
    ) -> Optional[OrderbookData]:
        """Parse raw orderbook data into structured format"""

        try:
            bids_raw = orderbook_data.get('bids', [])
            asks_raw = orderbook_data.get('asks', [])

            if not bids_raw or not asks_raw:
                return None

            # Parse bids (sorted descending by price)
            bids = []
            for bid in bids_raw[:self.config.max_levels]:
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    price, size = float(bid[0]), float(bid[1])
                    orders_count = int(bid[2]) if len(bid) > 2 else 1
                    bids.append(OrderbookLevel(price, size, orders_count))

            # Parse asks (sorted ascending by price)
            asks = []
            for ask in asks_raw[:self.config.max_levels]:
                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                    price, size = float(ask[0]), float(ask[1])
                    orders_count = int(ask[2]) if len(ask) > 2 else 1
                    asks.append(OrderbookLevel(price, size, orders_count))

            # Sort to ensure correct order
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            return OrderbookData(
                bids=bids,
                asks=asks,
                timestamp=datetime.now(),
                symbol=snapshot.symbol
            )

        except Exception as e:
            self.logger.error(f"Error parsing orderbook data: {str(e)}")
            return None

    def _calculate_imbalance_score(self, orderbook: OrderbookData) -> float:
        """Calcular score de imbalance bid/ask"""

        try:
            levels_to_analyze = min(self.config.analysis_depth, len(orderbook.bids), len(orderbook.asks))
            if levels_to_analyze == 0:
                return 0.0

            level_imbalances = []

            for i in range(levels_to_analyze):
                bid_level = orderbook.bids[i]
                ask_level = orderbook.asks[i]

                # Weight by level (closer levels have more weight)
                level_weight = self.config.level_decay_factor ** i

                # Calculate size-weighted imbalance
                bid_strength = bid_level.size * self.config.size_weight + bid_level.orders_count * self.config.count_weight
                ask_strength = ask_level.size * self.config.size_weight + ask_level.orders_count * self.config.count_weight

                if (bid_strength + ask_strength) > 0:
                    level_imbalance = (bid_strength - ask_strength) / (bid_strength + ask_strength)
                    weighted_imbalance = level_imbalance * level_weight
                    level_imbalances.append(weighted_imbalance)

            if not level_imbalances:
                return 0.0

            # Calculate weighted average
            total_weight = sum(self.config.level_decay_factor ** i for i in range(levels_to_analyze))
            imbalance_score = sum(level_imbalances) / total_weight if total_weight > 0 else 0.0

            # Normalize to [-1, 1] range
            return np.clip(imbalance_score, -1.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating imbalance score: {str(e)}")
            return 0.0

    def _calculate_density_score(self, orderbook: OrderbookData) -> float:
        """Calcular score de densidade de liquidez"""

        try:
            if not orderbook.bids or not orderbook.asks:
                return 0.0

            # Calculate spreads between levels
            bid_spreads = []
            ask_spreads = []

            # Bid side analysis
            for i in range(min(len(orderbook.bids) - 1, self.config.analysis_depth - 1)):
                spread = (orderbook.bids[i].price - orderbook.bids[i + 1].price) / orderbook.bids[i].price
                bid_spreads.append(spread)

            # Ask side analysis
            for i in range(min(len(orderbook.asks) - 1, self.config.analysis_depth - 1)):
                spread = (orderbook.asks[i + 1].price - orderbook.asks[i].price) / orderbook.asks[i].price
                ask_spreads.append(spread)

            # Calculate density (inverse of spread consistency)
            all_spreads = bid_spreads + ask_spreads
            if not all_spreads:
                return 0.0

            # Lower spreads = higher density = higher score
            avg_spread = np.mean(all_spreads)
            spread_consistency = 1.0 - np.std(all_spreads) / (avg_spread + 1e-8)

            # Density score: inverse relationship with spreads
            density_score = 1.0 / (1.0 + avg_spread * 1000)  # Normalize spreads

            # Combine with consistency
            final_score = density_score * 0.7 + spread_consistency * 0.3

            return np.clip(final_score, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating density score: {str(e)}")
            return 0.0

    def _calculate_pressure_score(self, orderbook: OrderbookData) -> float:
        """Calcular score de pressão de compra/venda"""

        try:
            levels_to_analyze = min(self.config.analysis_depth, len(orderbook.bids), len(orderbook.asks))
            if levels_to_analyze == 0:
                return 0.0

            # Calculate total volume on each side
            bid_volume = sum(level.size for level in orderbook.bids[:levels_to_analyze])
            ask_volume = sum(level.size for level in orderbook.asks[:levels_to_analyze])

            # Calculate total order count
            bid_orders = sum(level.orders_count for level in orderbook.bids[:levels_to_analyze])
            ask_orders = sum(level.orders_count for level in orderbook.asks[:levels_to_analyze])

            # Volume pressure
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                volume_pressure = (bid_volume - ask_volume) / total_volume
            else:
                volume_pressure = 0.0

            # Order count pressure (indicates retail vs institutional)
            total_orders = bid_orders + ask_orders
            if total_orders > 0:
                order_pressure = (bid_orders - ask_orders) / total_orders
            else:
                order_pressure = 0.0

            # Combine pressures
            pressure_score = volume_pressure * 0.7 + order_pressure * 0.3

            return np.clip(pressure_score, -1.0, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating pressure score: {str(e)}")
            return 0.0

    def _calculate_depth_score(self, orderbook: OrderbookData) -> float:
        """Calcular score de profundidade do mercado"""

        try:
            if not orderbook.bids or not orderbook.asks:
                return 0.0

            # Calculate cumulative volume at different distances from mid
            mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2

            # Define distance thresholds (as percentage of mid price)
            distance_thresholds = [0.001, 0.0025, 0.005, 0.01, 0.025]  # 0.1% to 2.5%

            bid_depths = []
            ask_depths = []

            for threshold in distance_thresholds:
                target_bid_price = mid_price * (1 - threshold)
                target_ask_price = mid_price * (1 + threshold)

                # Calculate cumulative volume to threshold
                bid_volume = sum(level.size for level in orderbook.bids if level.price >= target_bid_price)
                ask_volume = sum(level.size for level in orderbook.asks if level.price <= target_ask_price)

                bid_depths.append(bid_volume)
                ask_depths.append(ask_volume)

            # Calculate depth quality (higher volumes at closer levels = better)
            bid_quality = sum(vol / (i + 1) for i, vol in enumerate(bid_depths))
            ask_quality = sum(vol / (i + 1) for i, vol in enumerate(ask_depths))

            # Normalize by average volume
            avg_bid_vol = np.mean(bid_depths) if bid_depths else 1
            avg_ask_vol = np.mean(ask_depths) if ask_depths else 1

            bid_depth_score = bid_quality / (avg_bid_vol * len(distance_thresholds))
            ask_depth_score = ask_quality / (avg_ask_vol * len(distance_thresholds))

            # Combine and normalize
            depth_score = (bid_depth_score + ask_depth_score) / 2
            return np.clip(depth_score / 10, 0.0, 1.0)  # Scale down

        except Exception as e:
            self.logger.error(f"Error calculating depth score: {str(e)}")
            return 0.0

    def _calculate_time_weighted_score(self) -> float:
        """Calcular score ponderado por tempo"""

        if len(self.score_history) < 2:
            return 0.0

        try:
            current_time = time.time()
            weighted_scores = []

            for i, score_data in enumerate(self.score_history[-10:]):  # Last 10 calculations
                # Time decay weight
                age = (len(self.score_history) - 1 - i) * 0.5  # Assume 0.5s between calculations
                weight = np.exp(-age / self.config.decay_half_life)

                # Weight the composite score
                composite_score = score_data.get('ob_score', 0.0)
                weighted_scores.append(composite_score * weight)

            if weighted_scores:
                return np.mean(weighted_scores)

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating time-weighted score: {str(e)}")
            return 0.0

    def _calculate_composite_score(
        self,
        imbalance_score: float,
        density_score: float,
        pressure_score: float,
        depth_score: float,
        time_weighted_score: float
    ) -> float:
        """Calcular score composto final"""

        # Weights for different components
        weights = {
            'imbalance': 0.35,
            'density': 0.20,
            'pressure': 0.25,
            'depth': 0.15,
            'time_weighted': 0.05
        }

        # Calculate weighted score
        composite = (
            imbalance_score * weights['imbalance'] +
            density_score * weights['density'] +
            pressure_score * weights['pressure'] +
            depth_score * weights['depth'] +
            time_weighted_score * weights['time_weighted']
        )

        return np.clip(composite, -1.0, 1.0)

    def _fallback_score(self, snapshot: MarketSnapshot) -> Dict[str, float]:
        """Fallback score when orderbook data is not available"""

        # Use basic spread analysis
        spread_score = 1.0 - min(1.0, snapshot.spread / snapshot.mid / 0.001) if snapshot.mid > 0 else 0.0

        return {
            'ob_score': spread_score * 0.1,  # Very conservative without L2 data
            'imbalance_score': 0.0,
            'density_score': spread_score,
            'pressure_score': 0.0,
            'depth_score': 0.0,
            'time_weighted_score': 0.0,
            'execution_time_ms': 0.1,
            'levels_analyzed': 0,
            'calculation_quality': 'fallback'
        }

    def _update_history(self, orderbook: OrderbookData):
        """Update historical orderbook data"""
        self.orderbook_history.append(orderbook)

        # Keep only recent history (last 60 seconds assuming 1Hz updates)
        cutoff_time = time.time() - 60
        self.orderbook_history = [
            ob for ob in self.orderbook_history
            if ob.timestamp.timestamp() > cutoff_time
        ]

    def _update_performance_metrics(self, execution_time: float):
        """Update performance tracking metrics"""
        self.calculation_count += 1
        self.total_calculation_time += execution_time

        if self.calculation_count % 100 == 0:
            avg_time = self.total_calculation_time / self.calculation_count
            self.logger.info(f"OB-Score performance: {avg_time:.2f}ms avg over {self.calculation_count} calculations")

    def get_score_interpretation(self, score: float) -> Dict[str, str]:
        """Get human-readable interpretation of score"""

        abs_score = abs(score)
        direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"

        if abs_score >= self.config.strong_imbalance_threshold:
            strength = "Strong"
        elif abs_score >= self.config.medium_imbalance_threshold:
            strength = "Medium"
        elif abs_score >= self.config.weak_imbalance_threshold:
            strength = "Weak"
        else:
            strength = "Minimal"

        return {
            'direction': direction,
            'strength': strength,
            'description': f"{strength} {direction.lower()} pressure in orderbook",
            'numeric_score': score,
            'confidence': min(0.99, abs_score)
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
            'orderbook_history_count': len(self.orderbook_history),
            'score_history_count': len(self.score_history)
        }