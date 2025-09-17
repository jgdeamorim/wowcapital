"""
High Aggression Score - Score Proprietário para Alta Agressividade
Indicador composto para identificação de oportunidades Pocket Explosion

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
import asyncio
from dataclasses import dataclass, field
import time
from enum import Enum


class RiskLevel(Enum):
    """Níveis de risco para operações"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class AggressionWeights:
    """Pesos para os componentes do score de agressividade"""
    momentum: float = 0.25
    volatility: float = 0.20
    volume: float = 0.20
    microstructure: float = 0.15
    sentiment: float = 0.10
    technical: float = 0.10


@dataclass
class MarketSignals:
    """Container para todos os sinais de mercado necessários"""
    # Dados básicos
    prices: np.ndarray
    volumes: np.ndarray
    timestamp: float

    # Indicadores técnicos
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bb_position: Optional[float] = None  # Posição nas Bollinger Bands

    # Microestrutura
    bid_ask_spread: Optional[float] = None
    order_book_imbalance: Optional[float] = None
    trade_flow: Optional[float] = None

    # Sentiment
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    fear_greed_index: Optional[float] = None

    # Volatilidade
    realized_volatility: Optional[float] = None
    implied_volatility: Optional[float] = None

    # Meta dados
    symbol: str = ""
    timeframe: str = "1m"


@dataclass
class AggressionResult:
    """Resultado do cálculo de High Aggression Score"""
    score: float                          # Score final (0-1)
    components: Dict[str, float]          # Scores por componente
    explosion_ready: bool                 # ≥ 0.92 threshold
    risk_level: RiskLevel                 # Nível de risco calculado
    conviction_level: str                 # high, medium, low
    confidence: float                     # Confiança no score (0-1)
    timestamp: float                      # Timestamp do cálculo
    warnings: List[str] = field(default_factory=list)  # Avisos/alertas


class HighAggressionScore:
    """
    Score proprietário para identificação de oportunidades de alta agressividade
    com alavancagem controlada (sistema Pocket Explosion)

    Características:
    - Score composto de 6 componentes principais
    - Threshold 0.92 para detonação automática
    - Multi-confirmação para reduzir falsos positivos
    - Performance target: <10ms cálculo completo
    """

    def __init__(self, weights: Optional[AggressionWeights] = None):
        self.weights = weights or AggressionWeights()
        self.explosion_threshold = 0.92
        self.confidence_threshold = 0.75

        # Buffers para cálculos otimizados
        self._momentum_buffer = []
        self._volatility_buffer = []
        self._volume_buffer = []

        # Cache para performance
        self.cache = {}
        self.calculation_times = []

    def calculate_aggression_score(self, market_signals: MarketSignals) -> AggressionResult:
        """
        Calcula score composto para agressividade

        Args:
            market_signals: Todos os sinais de mercado necessários

        Returns:
            AggressionResult: Resultado completo da análise
        """
        start_time = time.perf_counter()

        try:
            # Validação de entrada
            self._validate_market_signals(market_signals)

            # Calcular componentes individuais
            components = {}

            # 1. Momentum component
            components['momentum'] = self._score_momentum(market_signals)

            # 2. Volatility component
            components['volatility'] = self._score_volatility(market_signals)

            # 3. Volume component
            components['volume'] = self._score_volume(market_signals)

            # 4. Microstructure component
            components['microstructure'] = self._score_microstructure(market_signals)

            # 5. Sentiment component
            components['sentiment'] = self._score_sentiment(market_signals)

            # 6. Technical component
            components['technical'] = self._score_technical(market_signals)

            # 7. Calcular score final ponderado
            final_score = self._calculate_weighted_score(components)

            # 8. Determinar nível de risco
            risk_level = self._determine_risk_level(final_score, components)

            # 9. Calcular confiança do score
            confidence = self._calculate_confidence(components, market_signals)

            # 10. Determinar nível de convicção
            conviction_level = self._determine_conviction_level(final_score, confidence)

            # 11. Verificar warnings/alertas
            warnings = self._check_warnings(components, market_signals)

            # Criar resultado
            result = AggressionResult(
                score=final_score,
                components=components,
                explosion_ready=final_score >= self.explosion_threshold,
                risk_level=risk_level,
                conviction_level=conviction_level,
                confidence=confidence,
                timestamp=market_signals.timestamp,
                warnings=warnings
            )

            # Performance tracking
            calc_time = (time.perf_counter() - start_time) * 1000
            self.calculation_times.append(calc_time)

            return result

        except Exception as e:
            raise Exception(f"High Aggression Score calculation error: {str(e)}")

    def _score_momentum(self, signals: MarketSignals) -> float:
        """
        Score do componente momentum (0-1)

        Args:
            signals: Sinais de mercado

        Returns:
            float: Score de momentum normalizado
        """
        scores = []

        # Price momentum (últimos períodos)
        if len(signals.prices) >= 5:
            recent_returns = np.diff(signals.prices[-5:]) / signals.prices[-5:-1]
            momentum_strength = np.mean(recent_returns)

            # Normalizar para 0-1
            momentum_score = np.tanh(momentum_strength * 100) * 0.5 + 0.5
            scores.append(momentum_score)

        # RSI momentum
        if signals.rsi is not None:
            if signals.rsi > 70:  # Overbought momentum
                rsi_score = min(1.0, (signals.rsi - 70) / 20)
            elif signals.rsi < 30:  # Oversold momentum
                rsi_score = min(1.0, (30 - signals.rsi) / 20)
            else:
                rsi_score = 0.3  # Neutro
            scores.append(rsi_score)

        # MACD momentum
        if signals.macd is not None:
            macd_score = np.tanh(abs(signals.macd) * 10) * 0.5 + 0.5
            scores.append(macd_score)

        return np.mean(scores) if scores else 0.5

    def _score_volatility(self, signals: MarketSignals) -> float:
        """
        Score do componente volatilidade (0-1)
        Alta volatilidade = maior oportunidade para explosões
        """
        scores = []

        # Volatilidade realizada
        if signals.realized_volatility is not None:
            # Normalizar volatilidade (assumindo range 0-0.1 para cripto)
            vol_score = min(1.0, signals.realized_volatility / 0.05)
            scores.append(vol_score)

        # Volatilidade dos preços recentes
        if len(signals.prices) >= 10:
            recent_prices = signals.prices[-10:]
            price_std = np.std(recent_prices) / np.mean(recent_prices)
            vol_score = min(1.0, price_std / 0.02)  # 2% como referência
            scores.append(vol_score)

        # Spread bid-ask como proxy de volatilidade
        if signals.bid_ask_spread is not None:
            spread_score = min(1.0, signals.bid_ask_spread * 1000)  # Assumindo spread em %
            scores.append(spread_score)

        return np.mean(scores) if scores else 0.5

    def _score_volume(self, signals: MarketSignals) -> float:
        """
        Score do componente volume (0-1)
        Alto volume = maior liquidez para explosões
        """
        scores = []

        if len(signals.volumes) >= 20:
            recent_volume = np.mean(signals.volumes[-5:])
            avg_volume = np.mean(signals.volumes[-20:])

            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                # Volume 2x acima da média = score máximo
                volume_score = min(1.0, volume_ratio / 2.0)
                scores.append(volume_score)

        # Volume trend
        if len(signals.volumes) >= 10:
            volume_trend = np.polyfit(range(10), signals.volumes[-10:], 1)[0]
            trend_score = np.tanh(volume_trend / np.mean(signals.volumes[-10:])) * 0.5 + 0.5
            scores.append(trend_score)

        return np.mean(scores) if scores else 0.5

    def _score_microstructure(self, signals: MarketSignals) -> float:
        """
        Score do componente microestrutura (0-1)
        Desequilíbrios no orderbook indicam oportunidades
        """
        scores = []

        # Order book imbalance
        if signals.order_book_imbalance is not None:
            imbalance_score = abs(signals.order_book_imbalance)
            scores.append(min(1.0, imbalance_score * 2))

        # Trade flow direction
        if signals.trade_flow is not None:
            flow_score = abs(signals.trade_flow)
            scores.append(flow_score)

        # Bid-ask spread (menor spread = melhor microestrutura)
        if signals.bid_ask_spread is not None:
            spread_score = 1.0 - min(1.0, signals.bid_ask_spread * 500)
            scores.append(max(0.0, spread_score))

        return np.mean(scores) if scores else 0.5

    def _score_sentiment(self, signals: MarketSignals) -> float:
        """
        Score do componente sentiment (0-1)
        Sentimento extremo pode gerar oportunidades
        """
        scores = []

        # News sentiment
        if signals.news_sentiment is not None:
            # Sentimento extremo (muito positivo ou negativo) gera oportunidades
            sentiment_score = abs(signals.news_sentiment - 0.5) * 2
            scores.append(sentiment_score)

        # Social sentiment
        if signals.social_sentiment is not None:
            social_score = abs(signals.social_sentiment - 0.5) * 2
            scores.append(social_score)

        # Fear & Greed Index
        if signals.fear_greed_index is not None:
            # Extremos de medo (0-20) ou ganância (80-100) = oportunidade
            if signals.fear_greed_index <= 20:
                fear_greed_score = (20 - signals.fear_greed_index) / 20
            elif signals.fear_greed_index >= 80:
                fear_greed_score = (signals.fear_greed_index - 80) / 20
            else:
                fear_greed_score = 0.3  # Neutro
            scores.append(fear_greed_score)

        return np.mean(scores) if scores else 0.5

    def _score_technical(self, signals: MarketSignals) -> float:
        """
        Score do componente técnico (0-1)
        Indicadores técnicos em posições extremas
        """
        scores = []

        # RSI extremes
        if signals.rsi is not None:
            if signals.rsi <= 30 or signals.rsi >= 70:
                rsi_score = 0.8
            elif signals.rsi <= 40 or signals.rsi >= 60:
                rsi_score = 0.6
            else:
                rsi_score = 0.3
            scores.append(rsi_score)

        # Bollinger Bands position
        if signals.bb_position is not None:
            # Posição extrema nas bandas = oportunidade
            bb_score = abs(signals.bb_position)
            scores.append(min(1.0, bb_score * 1.5))

        # MACD divergence proxy
        if signals.macd is not None:
            macd_score = min(1.0, abs(signals.macd) * 5)
            scores.append(macd_score)

        return np.mean(scores) if scores else 0.5

    def _calculate_weighted_score(self, components: Dict[str, float]) -> float:
        """Calcula score final ponderado"""
        weighted_sum = 0.0
        total_weight = 0.0

        for component, score in components.items():
            if hasattr(self.weights, component):
                weight = getattr(self.weights, component)
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_risk_level(self, score: float, components: Dict[str, float]) -> RiskLevel:
        """Determina nível de risco baseado no score e componentes"""
        if score >= 0.92:
            return RiskLevel.EXTREME
        elif score >= 0.8:
            return RiskLevel.AGGRESSIVE
        elif score >= 0.6:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.CONSERVATIVE

    def _calculate_confidence(self, components: Dict[str, float], signals: MarketSignals) -> float:
        """
        Calcula confiança no score baseado na consistência dos componentes
        """
        if not components:
            return 0.0

        # Consistência entre componentes
        scores = list(components.values())
        consistency = 1.0 - (np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 1.0)

        # Qualidade dos dados
        data_quality = self._assess_data_quality(signals)

        # Confiança final
        confidence = (consistency * 0.7) + (data_quality * 0.3)
        return np.clip(confidence, 0.0, 1.0)

    def _assess_data_quality(self, signals: MarketSignals) -> float:
        """Avalia qualidade dos dados de entrada"""
        quality_factors = []

        # Completude dos dados de preço
        if len(signals.prices) >= 20:
            quality_factors.append(1.0)
        elif len(signals.prices) >= 10:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)

        # Disponibilidade de dados de volume
        if len(signals.volumes) > 0:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)

        # Disponibilidade de dados de microestrutura
        microstructure_available = sum([
            signals.bid_ask_spread is not None,
            signals.order_book_imbalance is not None,
            signals.trade_flow is not None
        ])
        quality_factors.append(microstructure_available / 3.0)

        return np.mean(quality_factors)

    def _determine_conviction_level(self, score: float, confidence: float) -> str:
        """Determina nível de convicção"""
        if score >= 0.8 and confidence >= 0.8:
            return "high"
        elif score >= 0.6 and confidence >= 0.6:
            return "medium"
        else:
            return "low"

    def _check_warnings(self, components: Dict[str, float], signals: MarketSignals) -> List[str]:
        """Verifica condições que podem gerar warnings"""
        warnings = []

        # Low data quality warning
        data_quality = self._assess_data_quality(signals)
        if data_quality < 0.5:
            warnings.append("Low data quality detected")

        # Inconsistent signals warning
        scores = list(components.values())
        if len(scores) > 1 and np.std(scores) > 0.3:
            warnings.append("Inconsistent component signals")

        # Low liquidity warning (baseado em volume)
        if len(signals.volumes) > 0 and np.mean(signals.volumes[-5:]) < np.mean(signals.volumes) * 0.5:
            warnings.append("Below average liquidity detected")

        return warnings

    def _validate_market_signals(self, signals: MarketSignals) -> None:
        """Valida sinais de mercado de entrada"""
        if len(signals.prices) < 5:
            raise ValueError("Insufficient price data: minimum 5 periods required")

        if signals.timestamp <= 0:
            raise ValueError("Invalid timestamp")

        if len(signals.volumes) > 0 and len(signals.volumes) != len(signals.prices):
            raise ValueError("Price and volume series length mismatch")

    async def calculate_async(self, market_signals: MarketSignals) -> AggressionResult:
        """Versão assíncrona do cálculo"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.calculate_aggression_score, market_signals)

    def get_performance_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de performance"""
        if not self.calculation_times:
            return {'avg_calc_time_ms': 0.0}

        return {
            'avg_calc_time_ms': np.mean(self.calculation_times),
            'max_calc_time_ms': np.max(self.calculation_times),
            'min_calc_time_ms': np.min(self.calculation_times),
            'total_calculations': len(self.calculation_times)
        }


# Funções utilitárias

def quick_aggression_score(prices: Union[list, np.ndarray],
                          volumes: Optional[Union[list, np.ndarray]] = None,
                          **kwargs) -> float:
    """
    Função utilitária para cálculo rápido do aggression score

    Args:
        prices: Lista/array de preços
        volumes: Lista/array de volumes (opcional)
        **kwargs: Outros sinais de mercado

    Returns:
        float: Score de agressividade (0-1)
    """
    signals = MarketSignals(
        prices=np.array(prices),
        volumes=np.array(volumes) if volumes else np.array([]),
        timestamp=time.time(),
        **kwargs
    )

    scorer = HighAggressionScore()
    result = scorer.calculate_aggression_score(signals)
    return result.score


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, 50)) + 50000
    volumes = np.random.exponential(1000, 50)

    # Criar sinais de mercado de exemplo
    market_signals = MarketSignals(
        prices=prices,
        volumes=volumes,
        timestamp=time.time(),
        rsi=75.0,  # RSI overbought
        macd=2.5,
        order_book_imbalance=0.8,  # Strong buy pressure
        news_sentiment=0.9,  # Very positive
        realized_volatility=0.03,  # 3% volatility
        symbol="BTC/USDT"
    )

    # Calcular aggression score
    scorer = HighAggressionScore()
    result = scorer.calculate_aggression_score(market_signals)

    print(f"Aggression Score: {result.score:.4f}")
    print(f"Explosion Ready: {result.explosion_ready}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Conviction: {result.conviction_level}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Components: {result.components}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    # Performance
    perf = scorer.get_performance_stats()
    print(f"Calculation Time: {perf['avg_calc_time_ms']:.2f}ms")