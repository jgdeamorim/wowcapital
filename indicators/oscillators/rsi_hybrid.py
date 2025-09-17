"""
RSI Híbrido Dinâmico - Relative Strength Index Adaptativo
RSI com janela variável baseada em volatilidade e condições de mercado

Características:
- Período dinâmico que se adapta à volatilidade
- Ajuste de sensibilidade baseado em regime
- Noise filtering para reduzir falsos sinais
- Performance target: <2ms por cálculo

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import time
from dataclasses import dataclass
from enum import Enum


class RSISensitivity(Enum):
    """Níveis de sensibilidade do RSI"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RSIHybridConfig:
    """Configuração do RSI Híbrido"""
    base_period: int = 14               # Período base do RSI
    min_period: int = 7                 # Período mínimo
    max_period: int = 28                # Período máximo

    # Volatility adjustment
    volatility_adjustment: bool = True
    vol_fast_window: int = 10           # Janela rápida para volatilidade
    vol_slow_window: int = 30           # Janela lenta para volatilidade

    # Sensitivity thresholds
    overbought_threshold: float = 70.0
    oversold_threshold: float = 30.0
    extreme_overbought: float = 80.0
    extreme_oversold: float = 20.0

    # Noise filtering
    noise_filter: bool = True
    smoothing_periods: int = 3

    # Divergence detection
    divergence_detection: bool = True
    divergence_window: int = 20


@dataclass
class RSIHybridResult:
    """Resultado do RSI Híbrido"""
    rsi_value: float                    # Valor RSI atual
    rsi_smoothed: float                 # RSI suavizado (noise filtered)
    period_used: int                    # Período dinâmico utilizado
    sensitivity: RSISensitivity         # Nível de sensibilidade aplicado

    # Signals
    overbought: bool                    # RSI em região de sobrecompra
    oversold: bool                      # RSI em região de sobrevenda
    extreme_overbought: bool            # RSI em região extrema alta
    extreme_oversold: bool              # RSI em região extrema baixa

    # Advanced signals
    bullish_divergence: bool            # Divergência bullish detectada
    bearish_divergence: bool            # Divergência bearish detectada
    momentum_shift: str                 # Direção da mudança de momentum

    # Meta information
    volatility_proxy: float             # Proxy de volatilidade usado
    signal_strength: float              # Força do sinal (0-1)
    confidence: float                   # Confiança no sinal (0-1)
    timestamp: float                    # Timestamp do cálculo


class RSIHybrid:
    """
    RSI Híbrido: RSI adaptativo que ajusta período e sensibilidade
    baseado em condições de mercado dinâmicas

    Algoritmo:
    1. Calcula volatilidade de mercado
    2. Ajusta período do RSI baseado na volatilidade
    3. Calcula RSI com período dinâmico
    4. Aplica noise filtering
    5. Detecta divergências e sinais avançados
    """

    def __init__(self, config: Optional[RSIHybridConfig] = None):
        self.config = config or RSIHybridConfig()

        # Buffers para cálculos eficientes
        self.price_buffer: List[float] = []
        self.rsi_buffer: List[float] = []
        self.gains_buffer: List[float] = []
        self.losses_buffer: List[float] = []

        # Estado para EMA
        self.ema_gain: Optional[float] = None
        self.ema_loss: Optional[float] = None

        # Histórico para divergências
        self.price_highs: List[Tuple[int, float]] = []  # (index, price)
        self.price_lows: List[Tuple[int, float]] = []   # (index, price)
        self.rsi_highs: List[Tuple[int, float]] = []    # (index, rsi)
        self.rsi_lows: List[Tuple[int, float]] = []     # (index, rsi)

        # Performance tracking
        self.calculation_times: List[float] = []
        self.total_calculations = 0

    def calculate(self,
                 price_series: Union[pd.Series, np.ndarray],
                 volume_series: Optional[Union[pd.Series, np.ndarray]] = None) -> RSIHybridResult:
        """
        Calcula RSI Híbrido com ajustes dinâmicos

        Args:
            price_series: Série de preços
            volume_series: Série de volumes (opcional, para melhor volatilidade)

        Returns:
            RSIHybridResult: Resultado completo do cálculo
        """
        start_time = time.perf_counter()

        try:
            # Validação
            if len(price_series) < self.config.max_period:
                raise ValueError(f"Insufficient data: need at least {self.config.max_period} periods")

            # Converter para numpy
            prices = np.array(price_series) if not isinstance(price_series, np.ndarray) else price_series

            # 1. Calcular volatilidade proxy
            volatility_proxy = self._calculate_volatility_proxy(prices, volume_series)

            # 2. Determinar período dinâmico
            dynamic_period = self._calculate_dynamic_period(volatility_proxy)

            # 3. Calcular RSI com período dinâmico
            rsi_value = self._calculate_rsi(prices, dynamic_period)

            # 4. Aplicar noise filtering
            rsi_smoothed = self._apply_noise_filter(rsi_value)

            # 5. Determinar sensibilidade
            sensitivity = self._determine_sensitivity(volatility_proxy)

            # 6. Calcular sinais básicos
            signals = self._calculate_basic_signals(rsi_smoothed, sensitivity)

            # 7. Detectar divergências
            divergence_signals = self._detect_divergences(prices, rsi_smoothed)

            # 8. Analisar momentum shift
            momentum_shift = self._analyze_momentum_shift(rsi_smoothed)

            # 9. Calcular força e confiança do sinal
            signal_strength = self._calculate_signal_strength(rsi_smoothed, signals)
            confidence = self._calculate_confidence(rsi_smoothed, volatility_proxy, signals)

            # Criar resultado
            result = RSIHybridResult(
                rsi_value=rsi_value,
                rsi_smoothed=rsi_smoothed,
                period_used=dynamic_period,
                sensitivity=sensitivity,
                overbought=signals['overbought'],
                oversold=signals['oversold'],
                extreme_overbought=signals['extreme_overbought'],
                extreme_oversold=signals['extreme_oversold'],
                bullish_divergence=divergence_signals['bullish'],
                bearish_divergence=divergence_signals['bearish'],
                momentum_shift=momentum_shift,
                volatility_proxy=volatility_proxy,
                signal_strength=signal_strength,
                confidence=confidence,
                timestamp=time.time()
            )

            # Performance tracking
            calc_time = (time.perf_counter() - start_time) * 1000
            self.calculation_times.append(calc_time)
            self.total_calculations += 1

            return result

        except Exception as e:
            raise Exception(f"RSI Hybrid calculation error: {str(e)}")

    def _calculate_volatility_proxy(self,
                                  prices: np.ndarray,
                                  volumes: Optional[np.ndarray] = None) -> float:
        """
        Calcula proxy de volatilidade para ajustes dinâmicos
        """
        if len(prices) < self.config.vol_slow_window:
            # Use all available data
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns) * np.sqrt(252)  # Annualized

        # Volatilidade rápida vs lenta
        fast_window = self.config.vol_fast_window
        slow_window = self.config.vol_slow_window

        fast_returns = np.diff(prices[-fast_window:]) / prices[-fast_window:-1]
        slow_returns = np.diff(prices[-slow_window:]) / prices[-slow_window:-1]

        fast_vol = np.std(fast_returns) * np.sqrt(252)
        slow_vol = np.std(slow_returns) * np.sqrt(252)

        # Ratio como proxy de mudança de volatilidade
        vol_ratio = fast_vol / slow_vol if slow_vol > 0 else 1.0

        # Retornar volatilidade atual
        return fast_vol * vol_ratio

    def _calculate_dynamic_period(self, volatility_proxy: float) -> int:
        """
        Calcula período dinâmico baseado na volatilidade

        Lógica: Alta volatilidade -> período menor (mais sensível)
                Baixa volatilidade -> período maior (menos noise)
        """
        if not self.config.volatility_adjustment:
            return self.config.base_period

        # Normalizar volatilidade (assumindo range típico 0.1 - 0.6 para cripto)
        vol_normalized = np.clip((volatility_proxy - 0.1) / 0.5, 0.0, 1.0)

        # Período inversamente proporcional à volatilidade
        period_range = self.config.max_period - self.config.min_period
        dynamic_period = self.config.max_period - int(vol_normalized * period_range)

        # Ensure within bounds
        return np.clip(dynamic_period, self.config.min_period, self.config.max_period)

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """
        Calcula RSI tradicional com período especificado
        Otimizado para performance usando EMA incremental
        """
        if len(prices) < period + 1:
            period = len(prices) - 1

        # Calcular gains e losses
        price_changes = np.diff(prices)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        if len(gains) == 0:
            return 50.0  # Neutral RSI

        # Use últimos períodos
        recent_gains = gains[-period:]
        recent_losses = losses[-period:]

        # Calcular médias
        if len(self.gains_buffer) == 0:
            # Primeira vez - usar média simples
            avg_gain = np.mean(recent_gains)
            avg_loss = np.mean(recent_losses)
        else:
            # Usar EMA para eficiência
            alpha = 2.0 / (period + 1)

            if self.ema_gain is None:
                self.ema_gain = np.mean(recent_gains)
                self.ema_loss = np.mean(recent_losses)

            # Update EMAs
            current_gain = gains[-1] if len(gains) > 0 else 0
            current_loss = losses[-1] if len(losses) > 0 else 0

            self.ema_gain = alpha * current_gain + (1 - alpha) * self.ema_gain
            self.ema_loss = alpha * current_loss + (1 - alpha) * self.ema_loss

            avg_gain = self.ema_gain
            avg_loss = self.ema_loss

        # Calcular RSI
        if avg_loss == 0:
            return 100.0
        elif avg_gain == 0:
            return 0.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def _apply_noise_filter(self, rsi_value: float) -> float:
        """
        Aplica filtro de ruído usando média móvel simples
        """
        if not self.config.noise_filter:
            return rsi_value

        # Adicionar ao buffer
        self.rsi_buffer.append(rsi_value)

        # Manter tamanho do buffer
        if len(self.rsi_buffer) > self.config.smoothing_periods:
            self.rsi_buffer = self.rsi_buffer[-self.config.smoothing_periods:]

        # Retornar média suavizada
        return np.mean(self.rsi_buffer)

    def _determine_sensitivity(self, volatility_proxy: float) -> RSISensitivity:
        """
        Determina nível de sensibilidade baseado na volatilidade
        """
        if volatility_proxy > 0.4:  # Alta volatilidade
            return RSISensitivity.LOW    # Menos sensível para evitar whipsaws
        elif volatility_proxy > 0.2:   # Volatilidade normal
            return RSISensitivity.MEDIUM
        else:                           # Baixa volatilidade
            return RSISensitivity.HIGH   # Mais sensível para capturar movimentos

    def _calculate_basic_signals(self, rsi: float, sensitivity: RSISensitivity) -> Dict[str, bool]:
        """
        Calcula sinais básicos do RSI baseado na sensibilidade
        """
        # Ajustar thresholds baseado na sensibilidade
        if sensitivity == RSISensitivity.HIGH:
            overbought_thresh = self.config.overbought_threshold - 5  # 65
            oversold_thresh = self.config.oversold_threshold + 5      # 35
        elif sensitivity == RSISensitivity.LOW:
            overbought_thresh = self.config.overbought_threshold + 5  # 75
            oversold_thresh = self.config.oversold_threshold - 5      # 25
        else:  # MEDIUM
            overbought_thresh = self.config.overbought_threshold      # 70
            oversold_thresh = self.config.oversold_threshold          # 30

        return {
            'overbought': rsi >= overbought_thresh,
            'oversold': rsi <= oversold_thresh,
            'extreme_overbought': rsi >= self.config.extreme_overbought,
            'extreme_oversold': rsi <= self.config.extreme_oversold
        }

    def _detect_divergences(self, prices: np.ndarray, rsi: float) -> Dict[str, bool]:
        """
        Detecta divergências bullish e bearish
        """
        if not self.config.divergence_detection or len(prices) < self.config.divergence_window:
            return {'bullish': False, 'bearish': False}

        # Encontrar extremos locais
        self._update_extremes(prices, rsi)

        # Detectar divergências
        bullish_div = self._check_bullish_divergence()
        bearish_div = self._check_bearish_divergence()

        return {
            'bullish': bullish_div,
            'bearish': bearish_div
        }

    def _update_extremes(self, prices: np.ndarray, rsi: float):
        """
        Atualiza extremos locais para detecção de divergência
        """
        if len(prices) < 3:
            return

        current_idx = len(prices) - 1
        current_price = prices[-1]

        # Detectar máximos e mínimos locais (simples)
        if len(prices) >= 3:
            prev_price = prices[-2]
            prev_prev_price = prices[-3]

            # Máximo local
            if prev_price > current_price and prev_price > prev_prev_price:
                self.price_highs.append((current_idx - 1, prev_price))
                self.rsi_highs.append((current_idx - 1, self.rsi_buffer[-2] if len(self.rsi_buffer) >= 2 else rsi))

            # Mínimo local
            if prev_price < current_price and prev_price < prev_prev_price:
                self.price_lows.append((current_idx - 1, prev_price))
                self.rsi_lows.append((current_idx - 1, self.rsi_buffer[-2] if len(self.rsi_buffer) >= 2 else rsi))

        # Limitar histórico
        max_extremes = 10
        if len(self.price_highs) > max_extremes:
            self.price_highs = self.price_highs[-max_extremes:]
            self.rsi_highs = self.rsi_highs[-max_extremes:]
        if len(self.price_lows) > max_extremes:
            self.price_lows = self.price_lows[-max_extremes:]
            self.rsi_lows = self.rsi_lows[-max_extremes:]

    def _check_bullish_divergence(self) -> bool:
        """
        Verifica divergência bullish: preço faz minimum menor, RSI faz minimum maior
        """
        if len(self.price_lows) < 2 or len(self.rsi_lows) < 2:
            return False

        # Comparar últimos dois mínimos
        last_price_low = self.price_lows[-1][1]
        prev_price_low = self.price_lows[-2][1]
        last_rsi_low = self.rsi_lows[-1][1]
        prev_rsi_low = self.rsi_lows[-2][1]

        # Divergência bullish: preço menor, RSI maior
        return last_price_low < prev_price_low and last_rsi_low > prev_rsi_low

    def _check_bearish_divergence(self) -> bool:
        """
        Verifica divergência bearish: preço faz maximum maior, RSI faz maximum menor
        """
        if len(self.price_highs) < 2 or len(self.rsi_highs) < 2:
            return False

        # Comparar últimos dois máximos
        last_price_high = self.price_highs[-1][1]
        prev_price_high = self.price_highs[-2][1]
        last_rsi_high = self.rsi_highs[-1][1]
        prev_rsi_high = self.rsi_highs[-2][1]

        # Divergência bearish: preço maior, RSI menor
        return last_price_high > prev_price_high and last_rsi_high < prev_rsi_high

    def _analyze_momentum_shift(self, rsi: float) -> str:
        """
        Analisa mudança de momentum baseado na tendência do RSI
        """
        if len(self.rsi_buffer) < 3:
            return "neutral"

        recent_rsi = self.rsi_buffer[-3:]

        # Tendência do RSI
        if recent_rsi[-1] > recent_rsi[-2] > recent_rsi[-3]:
            return "strengthening_bullish"
        elif recent_rsi[-1] < recent_rsi[-2] < recent_rsi[-3]:
            return "strengthening_bearish"
        elif recent_rsi[-2] < recent_rsi[-3] and recent_rsi[-1] > recent_rsi[-2]:
            return "bullish_reversal"
        elif recent_rsi[-2] > recent_rsi[-3] and recent_rsi[-1] < recent_rsi[-2]:
            return "bearish_reversal"
        else:
            return "neutral"

    def _calculate_signal_strength(self, rsi: float, signals: Dict[str, bool]) -> float:
        """
        Calcula força do sinal baseado na posição do RSI e sinais ativos
        """
        # Base strength pela distância dos thresholds
        if rsi >= 70:
            base_strength = (rsi - 70) / 30  # 0 to 1 for RSI 70-100
        elif rsi <= 30:
            base_strength = (30 - rsi) / 30  # 0 to 1 for RSI 0-30
        else:
            base_strength = 0.1  # Neutral zone

        # Boost por sinais extremos
        if signals.get('extreme_overbought') or signals.get('extreme_oversold'):
            base_strength *= 1.5

        return min(base_strength, 1.0)

    def _calculate_confidence(self, rsi: float, volatility: float, signals: Dict[str, bool]) -> float:
        """
        Calcula confiança no sinal baseado em múltiplos fatores
        """
        confidence_factors = []

        # 1. Extremidade do RSI (mais extremo = mais confiável)
        if rsi >= 80 or rsi <= 20:
            confidence_factors.append(0.9)
        elif rsi >= 70 or rsi <= 30:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)

        # 2. Volatilidade (menor vol = mais confiável para RSI)
        vol_confidence = max(0.3, 1.0 - volatility)  # Assume volatility 0-1
        confidence_factors.append(vol_confidence)

        # 3. Consistência do buffer RSI
        if len(self.rsi_buffer) >= 3:
            rsi_stability = 1.0 - (np.std(self.rsi_buffer) / 100)
            confidence_factors.append(max(0.2, rsi_stability))
        else:
            confidence_factors.append(0.5)

        # Média ponderada
        return np.mean(confidence_factors)

    def get_trading_signals(self, result: RSIHybridResult) -> Dict[str, Union[bool, str, float]]:
        """
        Converte resultado RSI em sinais de trading estruturados
        """
        signals = {
            'timestamp': result.timestamp,
            'rsi_value': result.rsi_value,
            'rsi_smoothed': result.rsi_smoothed,

            # Basic signals
            'buy_oversold': result.oversold and not result.extreme_oversold,
            'sell_overbought': result.overbought and not result.extreme_overbought,

            # Strong signals
            'strong_buy': result.extreme_oversold or result.bullish_divergence,
            'strong_sell': result.extreme_overbought or result.bearish_divergence,

            # Momentum signals
            'momentum_bullish': result.momentum_shift in ['strengthening_bullish', 'bullish_reversal'],
            'momentum_bearish': result.momentum_shift in ['strengthening_bearish', 'bearish_reversal'],

            # Divergence signals
            'bullish_divergence': result.bullish_divergence,
            'bearish_divergence': result.bearish_divergence,

            # Meta information
            'signal_strength': result.signal_strength,
            'confidence': result.confidence,
            'sensitivity': result.sensitivity.value,
            'period_used': result.period_used
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
            'target_met_2ms': np.mean(self.calculation_times) < 2.0
        }


# Funções utilitárias

def quick_rsi_hybrid(prices: Union[list, np.ndarray], period: Optional[int] = None) -> float:
    """Cálculo rápido de RSI híbrido"""
    config = RSIHybridConfig()
    if period:
        config.base_period = period

    rsi = RSIHybrid(config)
    result = rsi.calculate(prices)
    return result.rsi_smoothed


# Exemplo de uso
if __name__ == "__main__":
    # Dados de teste
    np.random.seed(42)

    # Simular preços com diferentes regimes de volatilidade
    low_vol_prices = np.cumsum(np.random.normal(0.001, 0.01, 50)) + 50000
    high_vol_prices = np.cumsum(np.random.normal(0, 0.05, 30)) + low_vol_prices[-1]
    normal_vol_prices = np.cumsum(np.random.normal(0.002, 0.02, 40)) + high_vol_prices[-1]

    all_prices = np.concatenate([low_vol_prices, high_vol_prices, normal_vol_prices])

    # Configurar RSI Híbrido
    config = RSIHybridConfig(
        base_period=14,
        volatility_adjustment=True,
        noise_filter=True,
        divergence_detection=True
    )

    rsi_hybrid = RSIHybrid(config)

    # Calcular RSI para últimos períodos
    result = rsi_hybrid.calculate(all_prices)

    print("RSI Hybrid Results:")
    print(f"RSI Value: {result.rsi_value:.2f}")
    print(f"RSI Smoothed: {result.rsi_smoothed:.2f}")
    print(f"Period Used: {result.period_used}")
    print(f"Sensitivity: {result.sensitivity.value}")
    print(f"Volatility Proxy: {result.volatility_proxy:.4f}")

    print("\nSignals:")
    print(f"Overbought: {result.overbought}")
    print(f"Oversold: {result.oversold}")
    print(f"Extreme Overbought: {result.extreme_overbought}")
    print(f"Extreme Oversold: {result.extreme_oversold}")
    print(f"Bullish Divergence: {result.bullish_divergence}")
    print(f"Bearish Divergence: {result.bearish_divergence}")
    print(f"Momentum Shift: {result.momentum_shift}")

    print(f"\nSignal Strength: {result.signal_strength:.3f}")
    print(f"Confidence: {result.confidence:.3f}")

    # Trading signals
    trading_signals = rsi_hybrid.get_trading_signals(result)
    print("\nTrading Signals:")
    for key, value in trading_signals.items():
        if isinstance(value, bool) and value:
            print(f"  ✅ {key}")

    # Performance
    perf = rsi_hybrid.get_performance_stats()
    print(f"\nPerformance: {perf['avg_calc_time_ms']:.2f}ms")
    print(f"Target <2ms: {'✅' if perf.get('target_met_2ms', False) else '❌'}")