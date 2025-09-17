"""
MOMO-1.5L - Gradiente Variação Normalizado
Indicador proprietário para estratégia 1.5L baseline de segurança

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import asyncio
from dataclasses import dataclass
import time

@dataclass
class MOMO15LConfig:
    """Configuração do indicador MOMO-1.5L"""
    lookback: int = 21                    # Janela de análise
    smooth_factor: float = 0.65           # Fator de suavização
    boost_limit: float = 1.5              # Limitador low-boost
    normalization_window: int = 50        # Janela para normalização
    min_periods: int = 10                 # Períodos mínimos para cálculo


@dataclass
class MOMO15LResult:
    """Resultado do cálculo MOMO-1.5L"""
    score: float                          # Score final (-1.5 a +1.5)
    raw_gradient: float                   # Gradiente antes de normalização
    normalized_value: float               # Valor após normalização
    smoothed_value: float                 # Valor após suavização
    signal_strength: str                  # 'weak', 'moderate', 'strong'
    signal_direction: str                 # 'bullish', 'bearish', 'neutral'
    timestamp: float                      # Timestamp do cálculo


class MOMO15L:
    """
    MOMO-1.5L: Indicador proprietário para gradiente de variação normalizado
    com boost limitado para controle de risco da estratégia 1.5L

    Características:
    - Detecta momentum de baixa intensidade
    - Controle rigoroso de drawdown
    - Sinais para mean-reversion em extremos
    - Performance target: <5ms por cálculo
    """

    def __init__(self, config: Optional[MOMO15LConfig] = None):
        self.config = config or MOMO15LConfig()
        self.cache = {}
        self.last_calculation_time = 0

        # Buffers para otimização
        self.price_buffer = []
        self.volume_buffer = []
        self.gradient_buffer = []

        # Métricas de performance
        self.calculation_times = []
        self.total_calculations = 0

    def calculate(self,
                 price_series: Union[pd.Series, np.ndarray],
                 volume_series: Optional[Union[pd.Series, np.ndarray]] = None,
                 use_cache: bool = True) -> MOMO15LResult:
        """
        Calcula o indicador MOMO-1.5L

        Args:
            price_series: Série de preços
            volume_series: Série de volumes (opcional)
            use_cache: Usar cache para otimização

        Returns:
            MOMO15LResult: Resultado completo do cálculo

        Performance Target: <5ms
        """
        start_time = time.perf_counter()

        try:
            # Validação de entrada
            if len(price_series) < self.config.min_periods:
                raise ValueError(f"Insufficient data: need at least {self.config.min_periods} periods")

            # Converter para numpy array
            prices = np.array(price_series) if not isinstance(price_series, np.ndarray) else price_series

            # Cache check
            if use_cache:
                cache_key = self._generate_cache_key(prices)
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # 1. Calcular gradiente de variação
            raw_gradient = self._calculate_gradient(prices)

            # 2. Normalização adaptativa
            normalized_value = self._adaptive_normalize(raw_gradient, prices)

            # 3. Suavização exponencial
            smoothed_value = self._exponential_smooth(normalized_value)

            # 4. Aplicação do boost limitado
            boosted_value = smoothed_value * self.config.boost_limit

            # 5. Clipping final
            final_score = np.clip(boosted_value, -1.5, 1.5)

            # 6. Determinar força e direção do sinal
            signal_strength = self._determine_signal_strength(abs(final_score))
            signal_direction = self._determine_signal_direction(final_score)

            # Criar resultado
            result = MOMO15LResult(
                score=float(final_score),
                raw_gradient=float(raw_gradient),
                normalized_value=float(normalized_value),
                smoothed_value=float(smoothed_value),
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                timestamp=time.time()
            )

            # Cache resultado
            if use_cache:
                self.cache[cache_key] = result

            # Tracking de performance
            calculation_time = (time.perf_counter() - start_time) * 1000  # ms
            self.calculation_times.append(calculation_time)
            self.total_calculations += 1

            return result

        except Exception as e:
            raise Exception(f"MOMO-1.5L calculation error: {str(e)}")

    def _calculate_gradient(self, prices: np.ndarray) -> float:
        """
        Calcula gradiente de variação com janela lookback

        Returns:
            float: Gradiente calculado
        """
        if len(prices) < self.config.lookback:
            # Use todos os dados disponíveis se menor que lookback
            lookback_data = prices
        else:
            # Use janela lookback
            lookback_data = prices[-self.config.lookback:]

        # Calcular gradiente usando diferenças finitas
        gradient = np.gradient(lookback_data)

        # Retornar gradiente médio da janela
        return np.mean(gradient)

    def _adaptive_normalize(self, gradient: float, prices: np.ndarray) -> float:
        """
        Normalização adaptativa baseada na volatilidade histórica

        Args:
            gradient: Gradiente calculado
            prices: Série de preços para contexto

        Returns:
            float: Valor normalizado
        """
        # Calcular janela para normalização
        norm_window = min(self.config.normalization_window, len(prices))
        recent_prices = prices[-norm_window:]

        # Calcular gradientes históricos para normalização
        historical_gradients = []
        for i in range(self.config.lookback, len(recent_prices)):
            window = recent_prices[i-self.config.lookback:i]
            hist_gradient = np.mean(np.gradient(window))
            historical_gradients.append(hist_gradient)

        if len(historical_gradients) == 0:
            return 0.0

        historical_gradients = np.array(historical_gradients)

        # Normalização usando desvio padrão rolling
        rolling_std = np.std(historical_gradients)

        if rolling_std == 0:
            return 0.0

        # Normalizar o gradiente atual
        normalized = gradient / (rolling_std + 1e-8)  # Evitar divisão por zero

        return normalized

    def _exponential_smooth(self, value: float) -> float:
        """
        Aplicar suavização exponencial

        Args:
            value: Valor a ser suavizado

        Returns:
            float: Valor suavizado
        """
        if not hasattr(self, '_previous_smoothed'):
            self._previous_smoothed = value
            return value

        # EMA formula: new_value = alpha * current + (1-alpha) * previous
        alpha = 1.0 - self.config.smooth_factor
        smoothed = alpha * value + self.config.smooth_factor * self._previous_smoothed

        self._previous_smoothed = smoothed
        return smoothed

    def _determine_signal_strength(self, abs_score: float) -> str:
        """
        Determina força do sinal baseado no score absoluto

        Args:
            abs_score: Score absoluto

        Returns:
            str: Força do sinal ('weak', 'moderate', 'strong')
        """
        if abs_score < 0.3:
            return 'weak'
        elif abs_score < 0.8:
            return 'moderate'
        else:
            return 'strong'

    def _determine_signal_direction(self, score: float) -> str:
        """
        Determina direção do sinal

        Args:
            score: Score do MOMO-1.5L

        Returns:
            str: Direção ('bullish', 'bearish', 'neutral')
        """
        if score > 0.3:
            return 'bullish'
        elif score < -0.3:
            return 'bearish'
        else:
            return 'neutral'

    def _generate_cache_key(self, prices: np.ndarray) -> str:
        """Gera chave de cache baseada nos dados de entrada"""
        # Usar hash dos últimos valores para cache
        recent_prices = prices[-10:] if len(prices) >= 10 else prices
        return str(hash(tuple(recent_prices.round(4))))

    def get_trading_signals(self, result: MOMO15LResult) -> Dict[str, Union[bool, str]]:
        """
        Gera sinais de trading baseados no resultado MOMO-1.5L

        Args:
            result: Resultado do cálculo MOMO-1.5L

        Returns:
            Dict: Sinais de trading estruturados
        """
        signals = {
            'timestamp': result.timestamp,
            'score': result.score,

            # Sinais de entrada
            'long_signal': result.score > 0.3 and result.signal_strength in ['moderate', 'strong'],
            'short_signal': result.score < -0.3 and result.signal_strength in ['moderate', 'strong'],

            # Sinais de mean-reversion (extremos)
            'mean_revert_long': result.score < -0.8,
            'mean_revert_short': result.score > 0.8,

            # Sinais de saída
            'exit_long': result.score < 0.1 and result.signal_direction != 'bullish',
            'exit_short': result.score > -0.1 and result.signal_direction != 'bearish',

            # Meta informações
            'signal_strength': result.signal_strength,
            'signal_direction': result.signal_direction,
            'conviction_level': 'high' if abs(result.score) > 0.6 else 'medium' if abs(result.score) > 0.3 else 'low'
        }

        return signals

    def get_performance_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de performance do indicador"""
        if not self.calculation_times:
            return {'avg_calc_time_ms': 0.0, 'total_calculations': 0}

        return {
            'avg_calc_time_ms': np.mean(self.calculation_times),
            'max_calc_time_ms': np.max(self.calculation_times),
            'min_calc_time_ms': np.min(self.calculation_times),
            'total_calculations': self.total_calculations,
            'cache_hit_ratio': len(self.cache) / max(self.total_calculations, 1)
        }

    def clear_cache(self):
        """Limpa cache do indicador"""
        self.cache.clear()

    async def calculate_async(self,
                            price_series: Union[pd.Series, np.ndarray],
                            volume_series: Optional[Union[pd.Series, np.ndarray]] = None) -> MOMO15LResult:
        """
        Versão assíncrona do cálculo para integração com sistema de trading
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.calculate, price_series, volume_series)


# Funções utilitárias para integração rápida

def quick_momo_1_5l(prices: Union[list, np.ndarray, pd.Series],
                   config: Optional[MOMO15LConfig] = None) -> float:
    """
    Função utilitária para cálculo rápido do MOMO-1.5L

    Args:
        prices: Lista ou array de preços
        config: Configuração opcional

    Returns:
        float: Score MOMO-1.5L (-1.5 a +1.5)
    """
    indicator = MOMO15L(config)
    result = indicator.calculate(prices)
    return result.score


def momo_1_5l_signals(prices: Union[list, np.ndarray, pd.Series]) -> Dict[str, bool]:
    """
    Função utilitária para sinais rápidos do MOMO-1.5L

    Args:
        prices: Lista ou array de preços

    Returns:
        Dict: Sinais básicos de trading
    """
    indicator = MOMO15L()
    result = indicator.calculate(prices)
    signals = indicator.get_trading_signals(result)

    return {
        'buy': signals['long_signal'],
        'sell': signals['short_signal'],
        'mean_revert_buy': signals['mean_revert_long'],
        'mean_revert_sell': signals['mean_revert_short']
    }


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)
    n_periods = 100
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, n_periods)
    prices = [base_price]

    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Teste do indicador
    momo = MOMO15L()
    result = momo.calculate(prices)

    print(f"MOMO-1.5L Score: {result.score:.4f}")
    print(f"Signal Strength: {result.signal_strength}")
    print(f"Signal Direction: {result.signal_direction}")

    # Teste de sinais
    signals = momo.get_trading_signals(result)
    print(f"Long Signal: {signals['long_signal']}")
    print(f"Short Signal: {signals['short_signal']}")

    # Performance stats
    perf = momo.get_performance_stats()
    print(f"Calculation Time: {perf['avg_calc_time_ms']:.2f}ms")