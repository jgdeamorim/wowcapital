"""
VRP-fast - Volatility Risk Proxy Fast
Indicador avançado para proxy de risco de volatilidade em tempo real

Características:
- Detecta mudanças rápidas na volatilidade implícita vs realizada
- Calibração automática para diferentes regimes de mercado
- Performance target: <3ms por cálculo

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
import time
from dataclasses import dataclass
from enum import Enum


class VolatilityRegime(Enum):
    """Regimes de volatilidade"""
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"


@dataclass
class VRPFastConfig:
    """Configuração do VRP-fast"""
    fast_window: int = 10           # Janela rápida para volatilidade realizada
    slow_window: int = 30           # Janela lenta para baseline
    smoothing_factor: float = 0.3   # Alpha para suavização exponencial

    # Thresholds para regimes
    low_vol_threshold: float = 0.01     # 1% diário
    normal_vol_threshold: float = 0.02  # 2% diário
    high_vol_threshold: float = 0.04    # 4% diário

    # Calibração dinâmica
    auto_calibration: bool = True
    calibration_window: int = 100       # Janela para calibração


@dataclass
class VRPFastResult:
    """Resultado do cálculo VRP-fast"""
    vrp_score: float                    # Score principal (-1 a +1)
    realized_vol: float                 # Volatilidade realizada
    implied_vol_proxy: float            # Proxy volatilidade implícita
    vol_regime: VolatilityRegime        # Regime atual
    vol_percentile: float               # Percentil histórico (0-1)
    mean_reversion_signal: bool         # Sinal de reversão à média
    breakout_signal: bool               # Sinal de breakout
    confidence: float                   # Confiança no sinal (0-1)
    timestamp: float                    # Timestamp do cálculo


class VRPFast:
    """
    VRP-fast: Volatility Risk Proxy para detecção rápida de mudanças
    de volatilidade e sinais de reversão/breakout

    Método:
    1. Calcula volatilidade realizada (janela rápida vs lenta)
    2. Estima proxy de volatilidade implícita
    3. Calcula spread VRP = (IV - RV) / RV
    4. Determina regime e sinais
    """

    def __init__(self, config: Optional[VRPFastConfig] = None):
        self.config = config or VRPFastConfig()

        # Buffers para otimização
        self.price_buffer: List[float] = []
        self.returns_buffer: List[float] = []
        self.vol_history: List[float] = []

        # Estados internos
        self.ema_vol_fast = None
        self.ema_vol_slow = None
        self.calibration_stats = {
            'mean_vol': 0.02,
            'std_vol': 0.01,
            'percentiles': np.linspace(0.01, 0.06, 100)
        }

        # Performance tracking
        self.calculation_times: List[float] = []
        self.total_calculations = 0

    def calculate(self,
                 price_series: Union[pd.Series, np.ndarray],
                 volume_series: Optional[Union[pd.Series, np.ndarray]] = None) -> VRPFastResult:
        """
        Calcula VRP-fast para série de preços

        Args:
            price_series: Série de preços
            volume_series: Série de volumes (opcional, melhora accuracy)

        Returns:
            VRPFastResult: Resultado completo do cálculo
        """
        start_time = time.perf_counter()

        try:
            # Validação de entrada
            if len(price_series) < self.config.slow_window:
                raise ValueError(f"Insufficient data: need at least {self.config.slow_window} periods")

            # Converter para numpy
            prices = np.array(price_series) if not isinstance(price_series, np.ndarray) else price_series

            # 1. Calcular retornos
            returns = self._calculate_returns(prices)

            # 2. Volatilidade realizada (múltiplas janelas)
            realized_vol_fast = self._calculate_realized_volatility(returns, self.config.fast_window)
            realized_vol_slow = self._calculate_realized_volatility(returns, self.config.slow_window)

            # 3. Proxy de volatilidade implícita
            implied_vol_proxy = self._estimate_implied_volatility(returns, volume_series)

            # 4. Calcular VRP score
            vrp_score = self._calculate_vrp_score(realized_vol_fast, implied_vol_proxy)

            # 5. Determinar regime de volatilidade
            vol_regime = self._determine_volatility_regime(realized_vol_fast)

            # 6. Calcular percentil histórico
            vol_percentile = self._calculate_volatility_percentile(realized_vol_fast)

            # 7. Gerar sinais
            mean_reversion_signal = self._detect_mean_reversion_signal(vrp_score, vol_regime)
            breakout_signal = self._detect_breakout_signal(realized_vol_fast, vol_regime)

            # 8. Calcular confiança
            confidence = self._calculate_confidence(vrp_score, vol_percentile, vol_regime)

            # 9. Auto-calibração se habilitada
            if self.config.auto_calibration:
                self._update_calibration(realized_vol_fast)

            # Criar resultado
            result = VRPFastResult(
                vrp_score=vrp_score,
                realized_vol=realized_vol_fast,
                implied_vol_proxy=implied_vol_proxy,
                vol_regime=vol_regime,
                vol_percentile=vol_percentile,
                mean_reversion_signal=mean_reversion_signal,
                breakout_signal=breakout_signal,
                confidence=confidence,
                timestamp=time.time()
            )

            # Performance tracking
            calc_time = (time.perf_counter() - start_time) * 1000
            self.calculation_times.append(calc_time)
            self.total_calculations += 1

            return result

        except Exception as e:
            raise Exception(f"VRP-fast calculation error: {str(e)}")

    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calcula retornos log com tratamento de zeros"""
        prices_clean = np.where(prices <= 0, np.nan, prices)
        log_prices = np.log(prices_clean)
        returns = np.diff(log_prices)

        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        return returns

    def _calculate_realized_volatility(self, returns: np.ndarray, window: int) -> float:
        """
        Calcula volatilidade realizada usando janela especificada

        Fórmula: sqrt(sum(returns^2) / n) * sqrt(252) para anualização
        """
        if len(returns) < window:
            window = len(returns)

        recent_returns = returns[-window:]

        # Volatilidade realizada (anualizada)
        realized_vol = np.sqrt(np.sum(recent_returns ** 2) / len(recent_returns)) * np.sqrt(252)

        return realized_vol

    def _estimate_implied_volatility(self,
                                   returns: np.ndarray,
                                   volumes: Optional[np.ndarray] = None) -> float:
        """
        Estima proxy de volatilidade implícita usando métodos avançados

        Combina:
        1. GARCH-like prediction
        2. Volume-weighted volatility
        3. Regime adjustment
        """
        if len(returns) < 20:
            # Fallback simples
            return np.std(returns) * np.sqrt(252)

        # 1. Volatilidade GARCH simplificada
        # Vol(t) = alpha * return(t-1)^2 + beta * Vol(t-1)
        alpha, beta = 0.1, 0.85  # Parâmetros típicos GARCH(1,1)

        garch_vol = np.std(returns[-10:]) * np.sqrt(252)  # Initial estimate
        for i in range(min(10, len(returns))):
            garch_vol = alpha * (returns[-(i+1)] ** 2) * 252 + beta * garch_vol

        # 2. Ajuste por volume (se disponível)
        if volumes is not None and len(volumes) >= len(returns):
            recent_volumes = volumes[-len(returns):]
            volume_weight = np.corrcoef(np.abs(returns), recent_volumes[-len(returns):])[0, 1]

            # Se há correlação volume-volatilidade, ajustar
            if not np.isnan(volume_weight):
                volume_adj = 1 + 0.2 * abs(volume_weight)  # Máximo 20% adjustment
                garch_vol *= volume_adj

        # 3. Regime adjustment
        current_vol = self._calculate_realized_volatility(returns, 10)
        long_vol = self._calculate_realized_volatility(returns, 30)

        if current_vol > long_vol * 1.5:  # High vol regime
            garch_vol *= 1.1  # Expect continued high vol
        elif current_vol < long_vol * 0.7:  # Low vol regime
            garch_vol *= 0.95  # Expect some mean reversion

        return garch_vol

    def _calculate_vrp_score(self, realized_vol: float, implied_vol: float) -> float:
        """
        Calcula VRP score: (IV - RV) / RV

        Interpretação:
        - Positivo: IV > RV (volatilidade cara, expect mean reversion down)
        - Negativo: IV < RV (volatilidade barata, expect mean reversion up)
        """
        if realized_vol == 0:
            return 0.0

        vrp_raw = (implied_vol - realized_vol) / realized_vol

        # Clip para range razoável
        vrp_score = np.clip(vrp_raw, -1.0, 1.0)

        return vrp_score

    def _determine_volatility_regime(self, realized_vol: float) -> VolatilityRegime:
        """Determina regime de volatilidade atual"""

        if realized_vol <= self.config.low_vol_threshold:
            return VolatilityRegime.LOW_VOL
        elif realized_vol <= self.config.normal_vol_threshold:
            return VolatilityRegime.NORMAL_VOL
        elif realized_vol <= self.config.high_vol_threshold:
            return VolatilityRegime.HIGH_VOL
        else:
            return VolatilityRegime.EXTREME_VOL

    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """Calcula percentil da volatilidade atual vs histórico"""

        if len(self.vol_history) < 20:
            self.vol_history.append(current_vol)
            return 0.5  # Default 50th percentile

        # Manter histórico limitado
        if len(self.vol_history) >= self.config.calibration_window:
            self.vol_history = self.vol_history[-self.config.calibration_window:]

        self.vol_history.append(current_vol)

        # Calcular percentil
        percentile = (np.sum(np.array(self.vol_history) <= current_vol) /
                     len(self.vol_history))

        return percentile

    def _detect_mean_reversion_signal(self, vrp_score: float, regime: VolatilityRegime) -> bool:
        """
        Detecta sinal de reversão à média na volatilidade

        Lógica:
        - VRP muito positivo em regime alto: expect vol decrease
        - VRP muito negativo em regime baixo: expect vol increase
        """

        if regime in [VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL]:
            return vrp_score > 0.3  # IV cara, expect reversion down
        elif regime == VolatilityRegime.LOW_VOL:
            return vrp_score < -0.3  # IV barata, expect reversion up
        else:
            return abs(vrp_score) > 0.5  # Strong signal in normal regime

    def _detect_breakout_signal(self, realized_vol: float, regime: VolatilityRegime) -> bool:
        """
        Detecta sinal de breakout de volatilidade

        Lógica: Mudança súbita de regime
        """

        if len(self.vol_history) < 10:
            return False

        recent_avg_vol = np.mean(self.vol_history[-10:])

        # Breakout up
        if (regime in [VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL] and
            realized_vol > recent_avg_vol * 1.5):
            return True

        # Breakout down (collapse)
        if (regime == VolatilityRegime.LOW_VOL and
            realized_vol < recent_avg_vol * 0.6):
            return True

        return False

    def _calculate_confidence(self,
                            vrp_score: float,
                            vol_percentile: float,
                            regime: VolatilityRegime) -> float:
        """
        Calcula confiança no sinal baseado em múltiplos fatores
        """

        confidence_factors = []

        # 1. Força do VRP score
        vrp_strength = min(abs(vrp_score), 1.0)
        confidence_factors.append(vrp_strength)

        # 2. Extremidade do percentil
        percentile_extreme = 2 * abs(vol_percentile - 0.5)  # 0 = médio, 1 = extremo
        confidence_factors.append(percentile_extreme)

        # 3. Regime clarity
        regime_confidence = {
            VolatilityRegime.LOW_VOL: 0.8,
            VolatilityRegime.NORMAL_VOL: 0.6,
            VolatilityRegime.HIGH_VOL: 0.8,
            VolatilityRegime.EXTREME_VOL: 0.9
        }
        confidence_factors.append(regime_confidence[regime])

        # 4. Histórico de dados
        data_quality = min(len(self.vol_history) / 50, 1.0)
        confidence_factors.append(data_quality)

        # Média ponderada
        final_confidence = np.average(confidence_factors, weights=[0.3, 0.3, 0.2, 0.2])

        return final_confidence

    def _update_calibration(self, realized_vol: float):
        """Atualiza estatísticas de calibração"""

        if len(self.vol_history) >= 20:
            vol_array = np.array(self.vol_history)

            self.calibration_stats['mean_vol'] = np.mean(vol_array)
            self.calibration_stats['std_vol'] = np.std(vol_array)
            self.calibration_stats['percentiles'] = np.percentile(
                vol_array, np.arange(0, 101, 1)
            )

    def get_trading_signals(self, result: VRPFastResult) -> Dict[str, Union[bool, str, float]]:
        """
        Converte resultado VRP em sinais de trading práticos

        Returns:
            Dict com sinais estruturados
        """

        signals = {
            'timestamp': result.timestamp,
            'vrp_score': result.vrp_score,
            'volatility_regime': result.vol_regime.value,

            # Sinais de volatilidade
            'vol_mean_revert': result.mean_reversion_signal,
            'vol_breakout': result.breakout_signal,

            # Sinais de trading derivados
            'buy_vol_crush': (result.vrp_score > 0.4 and
                             result.vol_regime in [VolatilityRegime.HIGH_VOL, VolatilityRegime.EXTREME_VOL]),
            'sell_vol_spike': (result.vrp_score < -0.4 and
                              result.vol_regime == VolatilityRegime.LOW_VOL),

            # Risk management
            'reduce_size_high_vol': result.vol_regime == VolatilityRegime.EXTREME_VOL,
            'increase_size_low_vol': (result.vol_regime == VolatilityRegime.LOW_VOL and
                                     result.confidence > 0.7),

            # Meta informações
            'confidence': result.confidence,
            'vol_percentile': result.vol_percentile,
            'regime_stability': result.vol_regime != VolatilityRegime.EXTREME_VOL
        }

        return signals

    def get_performance_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de performance"""

        if not self.calculation_times:
            return {'avg_calc_time_ms': 0.0, 'total_calculations': 0}

        return {
            'avg_calc_time_ms': np.mean(self.calculation_times),
            'max_calc_time_ms': np.max(self.calculation_times),
            'min_calc_time_ms': np.min(self.calculation_times),
            'total_calculations': self.total_calculations,
            'target_met_3ms': np.mean(self.calculation_times) < 3.0
        }


# Funções utilitárias

def quick_vrp_fast(prices: Union[list, np.ndarray],
                   config: Optional[VRPFastConfig] = None) -> float:
    """
    Função utilitária para cálculo rápido do VRP-fast

    Returns:
        float: VRP score (-1 a +1)
    """
    vrp = VRPFast(config)
    result = vrp.calculate(prices)
    return result.vrp_score


def vrp_fast_regime(prices: Union[list, np.ndarray]) -> str:
    """
    Função utilitária para regime de volatilidade

    Returns:
        str: Regime de volatilidade
    """
    vrp = VRPFast()
    result = vrp.calculate(prices)
    return result.vol_regime.value


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)

    # Simular diferentes regimes de volatilidade
    low_vol_period = np.random.normal(0, 0.01, 50)  # Low vol
    high_vol_period = np.random.normal(0, 0.04, 20)  # High vol spike
    normal_vol_period = np.random.normal(0, 0.02, 30)  # Back to normal

    # Combinar em série de retornos
    all_returns = np.concatenate([low_vol_period, high_vol_period, normal_vol_period])

    # Converter para preços
    base_price = 50000
    prices = [base_price]
    for ret in all_returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    # Testar VRP-fast
    config = VRPFastConfig(
        fast_window=10,
        slow_window=30,
        auto_calibration=True
    )

    vrp = VRPFast(config)
    result = vrp.calculate(np.array(prices))

    print(f"VRP-fast Results:")
    print(f"VRP Score: {result.vrp_score:.4f}")
    print(f"Realized Vol: {result.realized_vol:.4f} ({result.realized_vol*100:.2f}%)")
    print(f"Implied Vol Proxy: {result.implied_vol_proxy:.4f}")
    print(f"Volatility Regime: {result.vol_regime.value}")
    print(f"Vol Percentile: {result.vol_percentile:.2f}")
    print(f"Mean Reversion Signal: {result.mean_reversion_signal}")
    print(f"Breakout Signal: {result.breakout_signal}")
    print(f"Confidence: {result.confidence:.3f}")

    # Trading signals
    signals = vrp.get_trading_signals(result)
    print(f"\nTrading Signals:")
    for key, value in signals.items():
        if isinstance(value, bool) and value:
            print(f"  ✅ {key}")
        elif isinstance(value, (int, float)):
            print(f"  📊 {key}: {value}")

    # Performance
    perf = vrp.get_performance_stats()
    print(f"\nPerformance: {perf['avg_calc_time_ms']:.2f}ms (target: <3ms)")
    print(f"Target Met: {'✅' if perf['target_met_3ms'] else '❌'}")