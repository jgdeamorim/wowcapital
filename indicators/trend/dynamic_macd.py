"""
Dynamic MACD - Moving Average Convergence Divergence Adaptativo
MACD com parâmetros adaptativos baseados em condições de mercado

Características:
- Parâmetros dinâmicos (fast, slow, signal) baseados em volatilidade
- Detecção de regime de mercado para otimização
- Múltiplos tipos de crossovers e divergências
- Performance target: <3ms por cálculo

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import time
from dataclasses import dataclass
from enum import Enum


class MACDRegime(Enum):
    """Regimes de mercado para MACD dinâmico"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATION = "consolidation"


@dataclass
class DynamicMACDConfig:
    """Configuração do Dynamic MACD"""
    # Parâmetros base
    fast_base: int = 12
    slow_base: int = 26
    signal_base: int = 9

    # Ranges adaptativos
    fast_min: int = 8
    fast_max: int = 16
    slow_min: int = 20
    slow_max: int = 35
    signal_min: int = 6
    signal_max: int = 12

    # Regime detection
    regime_window: int = 50
    trend_threshold: float = 0.02       # 2% para considerar trending
    volatility_threshold: float = 0.04  # 4% para considerar volátil

    # Divergence detection
    divergence_window: int = 25
    min_divergence_periods: int = 5

    # Smoothing
    histogram_smoothing: int = 3


@dataclass
class DynamicMACDResult:
    """Resultado do Dynamic MACD"""
    # Valores principais
    macd_line: float                    # Linha MACD
    signal_line: float                  # Linha de sinal
    histogram: float                    # Histograma (MACD - Signal)
    histogram_smoothed: float           # Histograma suavizado

    # Parâmetros utilizados
    fast_period: int                    # Período rápido utilizado
    slow_period: int                    # Período lento utilizado
    signal_period: int                  # Período do sinal utilizado
    regime: MACDRegime                  # Regime detectado

    # Sinais
    bullish_crossover: bool             # MACD cruza signal para cima
    bearish_crossover: bool             # MACD cruza signal para baixo
    zero_line_cross_up: bool           # MACD cruza zero para cima
    zero_line_cross_down: bool         # MACD cruza zero para baixo

    # Divergências
    bullish_divergence: bool            # Divergência bullish
    bearish_divergence: bool            # Divergência bearish

    # Momentum
    momentum_acceleration: str          # 'increasing', 'decreasing', 'stable'
    trend_strength: float               # Força da tendência (0-1)

    # Meta informação
    confidence: float                   # Confiança no sinal (0-1)
    signal_quality: str                 # 'strong', 'medium', 'weak'
    timestamp: float                    # Timestamp do cálculo


class DynamicMACD:
    """
    Dynamic MACD: MACD adaptativo que ajusta parâmetros baseado
    em condições de mercado para melhor performance

    Algoritmo:
    1. Detecta regime de mercado (trending, ranging, volatile, consolidation)
    2. Ajusta parâmetros MACD baseado no regime
    3. Calcula MACD com parâmetros otimizados
    4. Detecta sinais avançados e divergências
    """

    def __init__(self, config: Optional[DynamicMACDConfig] = None):
        self.config = config or DynamicMACDConfig()

        # Buffers para cálculos eficientes
        self.price_buffer: List[float] = []
        self.macd_buffer: List[float] = []
        self.signal_buffer: List[float] = []
        self.histogram_buffer: List[float] = []

        # EMAs para performance
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.ema_signal: Optional[float] = None

        # Histórico para divergências
        self.price_peaks: List[Tuple[int, float]] = []
        self.price_troughs: List[Tuple[int, float]] = []
        self.macd_peaks: List[Tuple[int, float]] = []
        self.macd_troughs: List[Tuple[int, float]] = []

        # Performance tracking
        self.calculation_times: List[float] = []
        self.total_calculations = 0

    def calculate(self,
                 price_series: Union[pd.Series, np.ndarray],
                 volume_series: Optional[Union[pd.Series, np.ndarray]] = None) -> DynamicMACDResult:
        """
        Calcula Dynamic MACD com parâmetros adaptativos

        Args:
            price_series: Série de preços
            volume_series: Série de volumes (opcional, para regime detection)

        Returns:
            DynamicMACDResult: Resultado completo do cálculo
        """
        start_time = time.perf_counter()

        try:
            # Validação
            if len(price_series) < self.config.slow_max:
                raise ValueError(f"Insufficient data: need at least {self.config.slow_max} periods")

            # Converter para numpy
            prices = np.array(price_series) if not isinstance(price_series, np.ndarray) else price_series

            # 1. Detectar regime de mercado
            regime = self._detect_market_regime(prices, volume_series)

            # 2. Ajustar parâmetros baseado no regime
            fast_period, slow_period, signal_period = self._adjust_parameters(regime, prices)

            # 3. Calcular MACD com parâmetros dinâmicos
            macd_line, signal_line, histogram = self._calculate_macd(
                prices, fast_period, slow_period, signal_period
            )

            # 4. Aplicar suavização ao histograma
            histogram_smoothed = self._smooth_histogram(histogram)

            # 5. Detectar crossovers
            crossover_signals = self._detect_crossovers(macd_line, signal_line)

            # 6. Detectar cruzamentos da linha zero
            zero_cross_signals = self._detect_zero_crossings(macd_line)

            # 7. Detectar divergências
            divergence_signals = self._detect_divergences(prices, macd_line)

            # 8. Analisar momentum
            momentum_analysis = self._analyze_momentum(histogram_smoothed)

            # 9. Calcular força da tendência
            trend_strength = self._calculate_trend_strength(macd_line, signal_line)

            # 10. Calcular confiança e qualidade do sinal
            confidence, signal_quality = self._calculate_signal_metrics(
                regime, crossover_signals, divergence_signals, trend_strength
            )

            # Criar resultado
            result = DynamicMACDResult(
                macd_line=macd_line,
                signal_line=signal_line,
                histogram=histogram,
                histogram_smoothed=histogram_smoothed,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                regime=regime,
                bullish_crossover=crossover_signals['bullish'],
                bearish_crossover=crossover_signals['bearish'],
                zero_line_cross_up=zero_cross_signals['cross_up'],
                zero_line_cross_down=zero_cross_signals['cross_down'],
                bullish_divergence=divergence_signals['bullish'],
                bearish_divergence=divergence_signals['bearish'],
                momentum_acceleration=momentum_analysis,
                trend_strength=trend_strength,
                confidence=confidence,
                signal_quality=signal_quality,
                timestamp=time.time()
            )

            # Performance tracking
            calc_time = (time.perf_counter() - start_time) * 1000
            self.calculation_times.append(calc_time)
            self.total_calculations += 1

            return result

        except Exception as e:
            raise Exception(f"Dynamic MACD calculation error: {str(e)}")

    def _detect_market_regime(self,
                            prices: np.ndarray,
                            volumes: Optional[np.ndarray] = None) -> MACDRegime:
        """
        Detecta regime de mercado para otimização de parâmetros
        """
        if len(prices) < self.config.regime_window:
            return MACDRegime.RANGING  # Default

        # Use últimos períodos para regime
        recent_prices = prices[-self.config.regime_window:]

        # 1. Calcular trend strength
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)
        trend_pct = slope / np.mean(recent_prices)

        # 2. Calcular volatilidade
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Anualizada

        # 3. Calcular range efficiency
        price_range = np.max(recent_prices) - np.min(recent_prices)
        net_move = abs(recent_prices[-1] - recent_prices[0])
        range_efficiency = net_move / price_range if price_range > 0 else 0

        # 4. Determinar regime
        if abs(trend_pct) > self.config.trend_threshold and range_efficiency > 0.6:
            return MACDRegime.TRENDING
        elif volatility > self.config.volatility_threshold:
            return MACDRegime.VOLATILE
        elif range_efficiency < 0.3:
            return MACDRegime.RANGING
        else:
            return MACDRegime.CONSOLIDATION

    def _adjust_parameters(self, regime: MACDRegime, prices: np.ndarray) -> Tuple[int, int, int]:
        """
        Ajusta parâmetros MACD baseado no regime detectado
        """
        # Parâmetros base
        fast = self.config.fast_base
        slow = self.config.slow_base
        signal = self.config.signal_base

        # Ajustes por regime
        if regime == MACDRegime.TRENDING:
            # Em trending: usar parâmetros mais rápidos para capturar momentum
            fast = max(self.config.fast_min, fast - 2)
            slow = max(self.config.slow_min, slow - 4)
            signal = max(self.config.signal_min, signal - 1)

        elif regime == MACDRegime.RANGING:
            # Em ranging: usar parâmetros mais lentos para evitar whipsaws
            fast = min(self.config.fast_max, fast + 2)
            slow = min(self.config.slow_max, slow + 4)
            signal = min(self.config.signal_max, signal + 2)

        elif regime == MACDRegime.VOLATILE:
            # Em volátil: usar parâmetros lentos para filtrar ruído
            fast = min(self.config.fast_max, fast + 3)
            slow = min(self.config.slow_max, slow + 6)
            signal = min(self.config.signal_max, signal + 3)

        elif regime == MACDRegime.CONSOLIDATION:
            # Em consolidação: parâmetros balanced
            # Manter parâmetros base
            pass

        return fast, slow, signal

    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[float, float, float]:
        """
        Calcula MACD com parâmetros especificados usando EMAs eficientes
        """
        if len(prices) < slow:
            # Usar todos os dados disponíveis
            fast = min(fast, len(prices))
            slow = min(slow, len(prices))

        # Calcular EMAs
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Atualizar buffer MACD
        self.macd_buffer.append(macd_line)
        if len(self.macd_buffer) > 100:  # Manter histórico limitado
            self.macd_buffer = self.macd_buffer[-50:]

        # Signal Line (EMA do MACD)
        if len(self.macd_buffer) >= signal:
            signal_line = self._calculate_ema(np.array(self.macd_buffer), signal)
        else:
            signal_line = macd_line  # Fallback

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """
        Calcula EMA eficientemente
        """
        if len(data) == 0:
            return 0.0

        if len(data) < period:
            return np.mean(data)

        # Usar EMA incremental para performance
        alpha = 2.0 / (period + 1)
        ema = data[0]

        for value in data[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return ema

    def _smooth_histogram(self, histogram: float) -> float:
        """
        Suaviza histograma para reduzir ruído
        """
        self.histogram_buffer.append(histogram)

        # Manter buffer limitado
        if len(self.histogram_buffer) > self.config.histogram_smoothing:
            self.histogram_buffer = self.histogram_buffer[-self.config.histogram_smoothing:]

        # Retornar média suavizada
        return np.mean(self.histogram_buffer)

    def _detect_crossovers(self, macd: float, signal: float) -> Dict[str, bool]:
        """
        Detecta crossovers entre MACD e Signal lines
        """
        if len(self.macd_buffer) < 2 or len(self.signal_buffer) < 2:
            self.signal_buffer.append(signal)
            return {'bullish': False, 'bearish': False}

        # Adicionar ao buffer
        self.signal_buffer.append(signal)
        if len(self.signal_buffer) > 100:
            self.signal_buffer = self.signal_buffer[-50:]

        # Verificar crossovers (precisa de pelo menos 2 pontos)
        if len(self.macd_buffer) >= 2 and len(self.signal_buffer) >= 2:
            prev_macd = self.macd_buffer[-2]
            prev_signal = self.signal_buffer[-2]

            # Bullish crossover: MACD cruza signal para cima
            bullish = prev_macd <= prev_signal and macd > signal

            # Bearish crossover: MACD cruza signal para baixo
            bearish = prev_macd >= prev_signal and macd < signal

            return {'bullish': bullish, 'bearish': bearish}

        return {'bullish': False, 'bearish': False}

    def _detect_zero_crossings(self, macd: float) -> Dict[str, bool]:
        """
        Detecta cruzamentos da linha zero
        """
        if len(self.macd_buffer) < 2:
            return {'cross_up': False, 'cross_down': False}

        prev_macd = self.macd_buffer[-2]

        # Cross up: de negativo para positivo
        cross_up = prev_macd <= 0 and macd > 0

        # Cross down: de positivo para negativo
        cross_down = prev_macd >= 0 and macd < 0

        return {'cross_up': cross_up, 'cross_down': cross_down}

    def _detect_divergences(self, prices: np.ndarray, macd: float) -> Dict[str, bool]:
        """
        Detecta divergências entre preço e MACD
        """
        if len(prices) < self.config.divergence_window:
            return {'bullish': False, 'bearish': False}

        # Atualizar extremos
        self._update_extremes(prices, macd)

        # Detectar divergências
        bullish_div = self._check_bullish_divergence()
        bearish_div = self._check_bearish_divergence()

        return {'bullish': bullish_div, 'bearish': bearish_div}

    def _update_extremes(self, prices: np.ndarray, macd: float):
        """
        Atualiza extremos para detecção de divergência
        """
        if len(prices) < 3:
            return

        current_idx = len(prices) - 1
        current_price = prices[-1]

        # Detectar extremos locais simples
        if len(prices) >= 3:
            prev_price = prices[-2]
            prev_prev_price = prices[-3]

            # Peak detection
            if prev_price > current_price and prev_price > prev_prev_price:
                self.price_peaks.append((current_idx - 1, prev_price))
                # Assumir que temos MACD histórico correspondente
                if len(self.macd_buffer) >= 2:
                    self.macd_peaks.append((current_idx - 1, self.macd_buffer[-2]))

            # Trough detection
            if prev_price < current_price and prev_price < prev_prev_price:
                self.price_troughs.append((current_idx - 1, prev_price))
                if len(self.macd_buffer) >= 2:
                    self.macd_troughs.append((current_idx - 1, self.macd_buffer[-2]))

        # Limitar histórico
        max_extremes = 10
        self.price_peaks = self.price_peaks[-max_extremes:]
        self.price_troughs = self.price_troughs[-max_extremes:]
        self.macd_peaks = self.macd_peaks[-max_extremes:]
        self.macd_troughs = self.macd_troughs[-max_extremes:]

    def _check_bullish_divergence(self) -> bool:
        """
        Verifica divergência bullish: preço faz low menor, MACD faz low maior
        """
        if len(self.price_troughs) < 2 or len(self.macd_troughs) < 2:
            return False

        # Comparar últimos troughs
        last_price_trough = self.price_troughs[-1][1]
        prev_price_trough = self.price_troughs[-2][1]
        last_macd_trough = self.macd_troughs[-1][1]
        prev_macd_trough = self.macd_troughs[-2][1]

        # Bullish divergence
        return last_price_trough < prev_price_trough and last_macd_trough > prev_macd_trough

    def _check_bearish_divergence(self) -> bool:
        """
        Verifica divergência bearish: preço faz high maior, MACD faz high menor
        """
        if len(self.price_peaks) < 2 or len(self.macd_peaks) < 2:
            return False

        # Comparar últimos peaks
        last_price_peak = self.price_peaks[-1][1]
        prev_price_peak = self.price_peaks[-2][1]
        last_macd_peak = self.macd_peaks[-1][1]
        prev_macd_peak = self.macd_peaks[-2][1]

        # Bearish divergence
        return last_price_peak > prev_price_peak and last_macd_peak < prev_macd_peak

    def _analyze_momentum(self, histogram: float) -> str:
        """
        Analisa aceleração do momentum baseado no histograma
        """
        if len(self.histogram_buffer) < 3:
            return 'stable'

        recent_hist = self.histogram_buffer[-3:]

        # Tendência do histograma
        if recent_hist[-1] > recent_hist[-2] > recent_hist[-3]:
            return 'increasing'
        elif recent_hist[-1] < recent_hist[-2] < recent_hist[-3]:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_trend_strength(self, macd: float, signal: float) -> float:
        """
        Calcula força da tendência baseada na separação MACD-Signal
        """
        # Força baseada na distância entre linhas
        separation = abs(macd - signal)

        # Normalizar baseado em valores típicos
        # Para cripto, separações de 100-500 são comuns
        normalized_strength = min(separation / 200, 1.0)

        return normalized_strength

    def _calculate_signal_metrics(self,
                                regime: MACDRegime,
                                crossovers: Dict[str, bool],
                                divergences: Dict[str, bool],
                                trend_strength: float) -> Tuple[float, str]:
        """
        Calcula confiança e qualidade do sinal
        """
        confidence_factors = []

        # 1. Regime favorability
        regime_confidence = {
            MACDRegime.TRENDING: 0.8,
            MACDRegime.CONSOLIDATION: 0.6,
            MACDRegime.RANGING: 0.4,
            MACDRegime.VOLATILE: 0.3
        }
        confidence_factors.append(regime_confidence[regime])

        # 2. Signal strength
        confidence_factors.append(trend_strength)

        # 3. Divergence boost
        if divergences['bullish'] or divergences['bearish']:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)

        # 4. Crossover clarity
        if crossovers['bullish'] or crossovers['bearish']:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)

        # Calculate final confidence
        final_confidence = np.mean(confidence_factors)

        # Determine signal quality
        if final_confidence > 0.7:
            quality = 'strong'
        elif final_confidence > 0.5:
            quality = 'medium'
        else:
            quality = 'weak'

        return final_confidence, quality

    def get_trading_signals(self, result: DynamicMACDResult) -> Dict[str, Union[bool, str, float]]:
        """
        Converte resultado MACD em sinais de trading estruturados
        """
        signals = {
            'timestamp': result.timestamp,
            'macd_value': result.macd_line,
            'signal_value': result.signal_line,
            'histogram': result.histogram_smoothed,

            # Basic crossover signals
            'buy_signal': result.bullish_crossover,
            'sell_signal': result.bearish_crossover,

            # Zero line signals
            'trend_bullish': result.zero_line_cross_up,
            'trend_bearish': result.zero_line_cross_down,

            # Strong signals (with divergence)
            'strong_buy': result.bullish_crossover and result.bullish_divergence,
            'strong_sell': result.bearish_crossover and result.bearish_divergence,

            # Momentum signals
            'momentum_increasing': result.momentum_acceleration == 'increasing',
            'momentum_decreasing': result.momentum_acceleration == 'decreasing',

            # Divergence signals
            'bullish_divergence': result.bullish_divergence,
            'bearish_divergence': result.bearish_divergence,

            # Trend signals
            'above_zero': result.macd_line > 0,
            'below_zero': result.macd_line < 0,

            # Meta information
            'trend_strength': result.trend_strength,
            'confidence': result.confidence,
            'signal_quality': result.signal_quality,
            'regime': result.regime.value,
            'parameters': f"{result.fast_period}/{result.slow_period}/{result.signal_period}"
        }

        return signals

    def get_performance_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de performance"""
        if not self.calculation_times:
            return {'avg_calc_time_ms': 0.0}

        return {
            'avg_calc_time_ms': np.mean(self.calculation_times),
            'max_calc_time_ms': np.max(self.calculation_times),
            'min_calc_time_ms': np.min(self.calculation_times),
            'total_calculations': self.total_calculations,
            'target_met_3ms': np.mean(self.calculation_times) < 3.0
        }


# Funções utilitárias

def quick_dynamic_macd(prices: Union[list, np.ndarray]) -> Tuple[float, float, float]:
    """
    Cálculo rápido de Dynamic MACD

    Returns:
        Tuple: (macd_line, signal_line, histogram)
    """
    dynamic_macd = DynamicMACD()
    result = dynamic_macd.calculate(prices)
    return result.macd_line, result.signal_line, result.histogram


# Exemplo de uso
if __name__ == "__main__":
    # Dados de teste
    np.random.seed(42)

    # Simular diferentes regimes
    trending_prices = np.cumsum(np.random.normal(0.003, 0.015, 60)) + 50000  # Trending up
    ranging_prices = 50000 + np.random.normal(0, 50, 30)                    # Ranging
    volatile_prices = np.cumsum(np.random.normal(0, 0.06, 40)) + 50000      # High volatility

    all_prices = np.concatenate([trending_prices, ranging_prices, volatile_prices])

    # Configurar Dynamic MACD
    config = DynamicMACDConfig(
        fast_base=12,
        slow_base=26,
        signal_base=9,
        regime_window=30,
        divergence_window=25
    )

    dynamic_macd = DynamicMACD(config)

    # Calcular para últimos períodos
    result = dynamic_macd.calculate(all_prices)

    print("Dynamic MACD Results:")
    print(f"MACD Line: {result.macd_line:.2f}")
    print(f"Signal Line: {result.signal_line:.2f}")
    print(f"Histogram: {result.histogram:.2f}")
    print(f"Histogram Smoothed: {result.histogram_smoothed:.2f}")

    print(f"\nParameters Used:")
    print(f"Fast: {result.fast_period}, Slow: {result.slow_period}, Signal: {result.signal_period}")
    print(f"Regime: {result.regime.value}")

    print(f"\nSignals:")
    print(f"Bullish Crossover: {result.bullish_crossover}")
    print(f"Bearish Crossover: {result.bearish_crossover}")
    print(f"Zero Line Cross Up: {result.zero_line_cross_up}")
    print(f"Zero Line Cross Down: {result.zero_line_cross_down}")
    print(f"Bullish Divergence: {result.bullish_divergence}")
    print(f"Bearish Divergence: {result.bearish_divergence}")

    print(f"\nMomentum Analysis:")
    print(f"Momentum Acceleration: {result.momentum_acceleration}")
    print(f"Trend Strength: {result.trend_strength:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Signal Quality: {result.signal_quality}")

    # Trading signals
    trading_signals = dynamic_macd.get_trading_signals(result)
    print("\nTrading Signals:")
    for key, value in trading_signals.items():
        if isinstance(value, bool) and value:
            print(f"  ✅ {key}")

    # Performance
    perf = dynamic_macd.get_performance_stats()
    print(f"\nPerformance: {perf['avg_calc_time_ms']:.2f}ms")
    print(f"Target <3ms: {'✅' if perf.get('target_met_3ms', False) else '❌'}")