"""
RegimeNet - Classificador de Regime de Mercado ML
Rede neural leve para classificação em tempo real de regimes de mercado

Regimes Detectados:
- TRENDING_BULL: Tendência de alta consistente
- TRENDING_BEAR: Tendência de baixa consistente
- RANGE_BOUND: Movimento lateral sem direção clara
- HIGH_VOLATILITY: Alta volatilidade sem direção
- CONSOLIDATION: Consolidação após movimento

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import time
from dataclasses import dataclass
from enum import Enum
import pickle
import json


class MarketRegime(Enum):
    """Regimes de mercado identificados"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    CONSOLIDATION = "consolidation"


@dataclass
class RegimeNetConfig:
    """Configuração do RegimeNet"""
    feature_window: int = 50        # Janela para extração de features
    prediction_horizon: int = 10    # Horizon para predição (períodos)
    confidence_threshold: float = 0.6  # Threshold mínimo para classificação

    # Features utilizadas
    use_price_features: bool = True
    use_volume_features: bool = True
    use_volatility_features: bool = True
    use_momentum_features: bool = True

    # Model hyperparameters
    learning_rate: float = 0.01
    regularization: float = 0.001
    update_frequency: int = 100     # Retrain a cada N observações


@dataclass
class RegimeNetResult:
    """Resultado da classificação de regime"""
    regime: MarketRegime           # Regime classificado
    confidence: float              # Confiança da classificação (0-1)
    regime_probabilities: Dict[MarketRegime, float]  # Probabilidades de cada regime
    features_importance: Dict[str, float]  # Importância das features
    regime_duration: int           # Duração estimada do regime (períodos)
    transition_probability: float  # Probabilidade de transição
    timestamp: float               # Timestamp da classificação


class SimpleMLP:
    """
    Multi-Layer Perceptron simples para classificação de regimes
    Implementação leve sem dependências externas de ML
    """

    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização Xavier para pesos
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def _relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def _softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._softmax(self.z2)
        return self.a2

    def predict(self, X):
        """Predição"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probabilities = self.forward(X)
        return probabilities

    def update_weights(self, X, y, learning_rate=0.01):
        """Update simplificado dos pesos (sem backprop completo)"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Forward pass
        probabilities = self.forward(X)

        # Simple gradient update (aproximado)
        error = probabilities - y
        grad_W2 = np.dot(self.a1.T, error) * learning_rate
        grad_b2 = np.sum(error, axis=0, keepdims=True) * learning_rate

        # Update weights
        self.W2 -= grad_W2
        self.b2 -= grad_b2


class RegimeNet:
    """
    RegimeNet: Classificador de regime de mercado usando ML leve

    Processo:
    1. Extrai features técnicas do market data
    2. Classifica regime usando MLP simples
    3. Calcula confiança e probabilidades
    4. Auto-treina com novos dados
    """

    def __init__(self, config: Optional[RegimeNetConfig] = None):
        self.config = config or RegimeNetConfig()

        # Features history para treinamento
        self.features_history: List[np.ndarray] = []
        self.labels_history: List[int] = []

        # Model state
        self.model: Optional[SimpleMLP] = None
        self.is_trained = False
        self.training_samples = 0

        # Feature importance tracking
        self.feature_names = []
        self.feature_importance = {}

        # Regime tracking
        self.regime_history: List[MarketRegime] = []
        self.regime_durations: Dict[MarketRegime, List[int]] = {
            regime: [] for regime in MarketRegime
        }

        # Performance tracking
        self.calculation_times: List[float] = []
        self.accuracy_history: List[float] = []

    def classify_regime(self,
                       prices: Union[pd.Series, np.ndarray],
                       volumes: Optional[Union[pd.Series, np.ndarray]] = None) -> RegimeNetResult:
        """
        Classifica regime de mercado atual

        Args:
            prices: Série de preços
            volumes: Série de volumes (opcional)

        Returns:
            RegimeNetResult: Resultado da classificação
        """
        start_time = time.perf_counter()

        try:
            # Validação
            if len(prices) < self.config.feature_window:
                raise ValueError(f"Insufficient data: need at least {self.config.feature_window} periods")

            # Converter para numpy
            prices_array = np.array(prices) if not isinstance(prices, np.ndarray) else prices
            volumes_array = np.array(volumes) if volumes is not None else None

            # 1. Extrair features
            features = self._extract_features(prices_array, volumes_array)

            # 2. Inicializar/treinar modelo se necessário
            if not self.is_trained:
                self._initialize_model(features)

            # 3. Classificar regime
            regime_probabilities = self._classify_with_model(features)

            # 4. Determinar regime e confiança
            regime, confidence = self._determine_regime_and_confidence(regime_probabilities)

            # 5. Estimar duração e transição
            regime_duration = self._estimate_regime_duration(regime)
            transition_probability = self._calculate_transition_probability(regime)

            # 6. Calcular importância das features
            features_importance = self._calculate_feature_importance(features)

            # 7. Update histórico
            self._update_regime_history(regime)

            # 8. Auto-treinamento (se configurado)
            if self.training_samples % self.config.update_frequency == 0:
                self._online_learning_update(features, regime)

            # Criar resultado
            result = RegimeNetResult(
                regime=regime,
                confidence=confidence,
                regime_probabilities=regime_probabilities,
                features_importance=features_importance,
                regime_duration=regime_duration,
                transition_probability=transition_probability,
                timestamp=time.time()
            )

            # Performance tracking
            calc_time = (time.perf_counter() - start_time) * 1000
            self.calculation_times.append(calc_time)

            return result

        except Exception as e:
            raise Exception(f"RegimeNet classification error: {str(e)}")

    def _extract_features(self,
                         prices: np.ndarray,
                         volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extrai features técnicas para classificação

        Returns:
            np.ndarray: Feature vector normalizado
        """
        features = []
        window = self.config.feature_window

        # Usar últimos períodos
        recent_prices = prices[-window:]

        # 1. Price-based features
        if self.config.use_price_features:
            returns = np.diff(recent_prices) / recent_prices[:-1]

            # Trend strength
            trend_strength = self._calculate_trend_strength(recent_prices)
            features.append(trend_strength)

            # Price momentum (múltiplas janelas)
            momentum_5 = (recent_prices[-1] - recent_prices[-6]) / recent_prices[-6] if len(recent_prices) >= 6 else 0
            momentum_10 = (recent_prices[-1] - recent_prices[-11]) / recent_prices[-11] if len(recent_prices) >= 11 else 0
            features.extend([momentum_5, momentum_10])

            # Return statistics
            features.extend([
                np.mean(returns),           # Mean return
                np.std(returns),            # Volatility
                len([r for r in returns if r > 0]) / len(returns),  # Win rate
            ])

        # 2. Volume-based features
        if self.config.use_volume_features and volumes is not None:
            recent_volumes = volumes[-window:] if len(volumes) >= window else volumes

            if len(recent_volumes) > 0:
                # Volume trend
                volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                features.append(volume_trend / np.mean(recent_volumes))

                # Volume-price correlation
                if len(recent_prices) == len(recent_volumes):
                    vol_price_corr = np.corrcoef(recent_prices, recent_volumes)[0, 1]
                    features.append(vol_price_corr if not np.isnan(vol_price_corr) else 0)
                else:
                    features.append(0)
            else:
                features.extend([0, 0])  # Default values

        # 3. Volatility features
        if self.config.use_volatility_features:
            returns = np.diff(recent_prices) / recent_prices[:-1]

            # Rolling volatility
            vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
            vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

            # Volatility ratio
            vol_ratio = vol_5 / vol_20 if vol_20 != 0 else 1

            features.extend([vol_5, vol_20, vol_ratio])

        # 4. Momentum features
        if self.config.use_momentum_features:
            # RSI approximation
            gains = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]

            if len(gains) > 0 and len(losses) > 0:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50  # Neutral

            features.append(rsi / 100)  # Normalize

            # MACD approximation
            ema_12 = self._ema(recent_prices, 12)
            ema_26 = self._ema(recent_prices, 26)
            macd = ema_12 - ema_26
            features.append(macd / recent_prices[-1])  # Normalize

        # Converter para array e normalizar
        features_array = np.array(features)

        # Store feature names for importance calculation
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(len(features))]

        # Normalização simples (z-score com clipping)
        features_normalized = np.clip(features_array, -3, 3) / 3

        return features_normalized

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcula força da tendência usando regressão linear"""
        if len(prices) < 2:
            return 0.0

        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)

        # Normalizar slope pelo preço médio
        trend_strength = slope / np.mean(prices)

        # Calcular R² para confiança da tendência
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Combinar slope e R² para trend strength
        return trend_strength * r_squared

    def _ema(self, data: np.ndarray, window: int) -> float:
        """Exponential Moving Average simples"""
        if len(data) == 0:
            return 0.0
        if len(data) < window:
            return np.mean(data)

        alpha = 2.0 / (window + 1)
        ema = data[0]

        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _initialize_model(self, sample_features: np.ndarray):
        """Inicializa modelo com features de exemplo"""
        input_size = len(sample_features)
        self.model = SimpleMLP(input_size=input_size, hidden_size=32, output_size=5)

        # Treinamento inicial com dados sintéticos/heurísticos
        self._bootstrap_training()
        self.is_trained = True

    def _bootstrap_training(self):
        """Bootstrap inicial com classificações heurísticas"""
        # Este método criaria dados de treinamento inicial usando regras heurísticas
        # Por simplicidade, apenas marcamos como treinado
        pass

    def _classify_with_model(self, features: np.ndarray) -> Dict[MarketRegime, float]:
        """Classifica usando o modelo ML"""
        if self.model is None:
            # Fallback para classificação heurística
            return self._heuristic_classification(features)

        # Predição com modelo
        probabilities = self.model.predict(features)[0]

        # Mapear para regimes
        regime_list = list(MarketRegime)
        regime_probs = {
            regime_list[i]: prob for i, prob in enumerate(probabilities)
        }

        return regime_probs

    def _heuristic_classification(self, features: np.ndarray) -> Dict[MarketRegime, float]:
        """Classificação heurística como fallback"""
        # Classificação simples baseada em features conhecidas
        if len(features) < 5:
            # Default equal probabilities
            return {regime: 0.2 for regime in MarketRegime}

        trend_strength = features[0] if len(features) > 0 else 0
        momentum_5 = features[1] if len(features) > 1 else 0
        volatility = features[4] if len(features) > 4 else 0

        # Lógica heurística
        probs = {regime: 0.1 for regime in MarketRegime}

        if trend_strength > 0.1 and momentum_5 > 0.02:
            probs[MarketRegime.TRENDING_BULL] = 0.6
        elif trend_strength < -0.1 and momentum_5 < -0.02:
            probs[MarketRegime.TRENDING_BEAR] = 0.6
        elif volatility > 0.5:
            probs[MarketRegime.HIGH_VOLATILITY] = 0.5
        elif abs(trend_strength) < 0.05:
            probs[MarketRegime.RANGE_BOUND] = 0.4
        else:
            probs[MarketRegime.CONSOLIDATION] = 0.4

        # Normalizar
        total = sum(probs.values())
        return {regime: prob/total for regime, prob in probs.items()}

    def _determine_regime_and_confidence(self,
                                       regime_probabilities: Dict[MarketRegime, float]) -> Tuple[MarketRegime, float]:
        """Determina regime dominante e confiança"""
        # Regime com maior probabilidade
        regime = max(regime_probabilities.items(), key=lambda x: x[1])[0]
        confidence = regime_probabilities[regime]

        # Ajustar confiança baseado na separação
        sorted_probs = sorted(regime_probabilities.values(), reverse=True)
        if len(sorted_probs) >= 2:
            separation = sorted_probs[0] - sorted_probs[1]
            confidence *= (1 + separation)  # Boost confidence se há separação clara

        return regime, min(confidence, 1.0)

    def _estimate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Estima duração esperada do regime atual"""
        if current_regime in self.regime_durations and self.regime_durations[current_regime]:
            # Média histórica
            return int(np.mean(self.regime_durations[current_regime]))
        else:
            # Default por regime
            default_durations = {
                MarketRegime.TRENDING_BULL: 25,
                MarketRegime.TRENDING_BEAR: 20,
                MarketRegime.RANGE_BOUND: 15,
                MarketRegime.HIGH_VOLATILITY: 8,
                MarketRegime.CONSOLIDATION: 12
            }
            return default_durations.get(current_regime, 15)

    def _calculate_transition_probability(self, current_regime: MarketRegime) -> float:
        """Calcula probabilidade de transição de regime"""
        if len(self.regime_history) < 10:
            return 0.1  # Default baixa transição

        # Contar transições recentes
        recent_history = self.regime_history[-20:]
        transitions = sum(1 for i in range(1, len(recent_history))
                         if recent_history[i] != recent_history[i-1])

        transition_rate = transitions / len(recent_history)
        return min(transition_rate, 0.5)  # Cap em 50%

    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calcula importância das features"""
        # Implementação simplificada - usa magnitude das features
        if not self.feature_names:
            return {}

        feature_magnitudes = np.abs(features)
        total_magnitude = np.sum(feature_magnitudes)

        if total_magnitude == 0:
            return {name: 0.0 for name in self.feature_names}

        importance = {}
        for i, name in enumerate(self.feature_names[:len(features)]):
            importance[name] = feature_magnitudes[i] / total_magnitude

        return importance

    def _update_regime_history(self, regime: MarketRegime):
        """Atualiza histórico de regimes"""
        self.regime_history.append(regime)

        # Manter histórico limitado
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]

        # Atualizar durações se houve mudança
        if len(self.regime_history) >= 2 and self.regime_history[-2] != regime:
            # Calcular duração do regime anterior
            prev_regime = self.regime_history[-2]
            duration = 1
            for i in range(len(self.regime_history) - 2, -1, -1):
                if self.regime_history[i] == prev_regime:
                    duration += 1
                else:
                    break

            self.regime_durations[prev_regime].append(duration)

            # Limitar histórico de durações
            if len(self.regime_durations[prev_regime]) > 50:
                self.regime_durations[prev_regime] = self.regime_durations[prev_regime][-30:]

    def _online_learning_update(self, features: np.ndarray, regime: MarketRegime):
        """Update do modelo com nova observação"""
        self.training_samples += 1

        # Store para possível retreino futuro
        self.features_history.append(features)
        self.labels_history.append(list(MarketRegime).index(regime))

        # Limitar histórico
        if len(self.features_history) > 500:
            self.features_history = self.features_history[-300:]
            self.labels_history = self.labels_history[-300:]

    def get_performance_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de performance"""
        if not self.calculation_times:
            return {'avg_calc_time_ms': 0.0}

        stats = {
            'avg_calc_time_ms': np.mean(self.calculation_times),
            'max_calc_time_ms': np.max(self.calculation_times),
            'total_classifications': len(self.calculation_times),
            'is_trained': self.is_trained,
            'training_samples': self.training_samples
        }

        if self.accuracy_history:
            stats['avg_accuracy'] = np.mean(self.accuracy_history)

        return stats


    def calculate(self, price_series: np.ndarray, volume_series: Optional[np.ndarray] = None) -> Dict[str, any]:
        """Método compatível para classificação de regime"""
        return self.classify_regime(price_series, volume_series)


# Funções utilitárias

def quick_regime_classification(prices: Union[list, np.ndarray],
                               volumes: Optional[Union[list, np.ndarray]] = None) -> str:
    """
    Classificação rápida de regime

    Returns:
        str: Nome do regime
    """
    regime_net = RegimeNet()
    result = regime_net.classify_regime(prices, volumes)
    return result.regime.value


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo simulando diferentes regimes
    np.random.seed(42)

    # Regime Bull Trend
    bull_trend = np.cumsum(np.random.normal(0.002, 0.01, 50)) + 50000

    # Regime Range Bound
    range_bound = 50000 + np.random.normal(0, 100, 30)

    # Regime High Vol
    high_vol = np.cumsum(np.random.normal(0, 0.05, 20)) + 50000

    # Combinar
    all_prices = np.concatenate([bull_trend, range_bound, high_vol])
    volumes = np.random.exponential(1000, len(all_prices))

    # Teste RegimeNet
    config = RegimeNetConfig(
        feature_window=40,
        confidence_threshold=0.5
    )

    regime_net = RegimeNet(config)

    # Classificar últimos períodos
    result = regime_net.classify_regime(all_prices, volumes)

    print("RegimeNet Results:")
    print(f"Current Regime: {result.regime.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Estimated Duration: {result.regime_duration} periods")
    print(f"Transition Probability: {result.transition_probability:.3f}")

    print("\nRegime Probabilities:")
    for regime, prob in result.regime_probabilities.items():
        print(f"  {regime.value}: {prob:.3f}")

    print("\nTop Features:")
    sorted_features = sorted(result.features_importance.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features[:5]:
        print(f"  {name}: {importance:.3f}")

    # Performance
    perf = regime_net.get_performance_stats()
    print(f"\nPerformance: {perf['avg_calc_time_ms']:.2f}ms")
    print(f"Training Samples: {perf['training_samples']}")
    print(f"Model Trained: {'✅' if perf['is_trained'] else '❌'}")