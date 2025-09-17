"""
Strategy 1.5L - Low-Boost Momentum/Mean-Revert
Estratégia baseline de segurança com foco em estabilidade e drawdown controlado

Performance Target:
- Retorno diário médio: 0.65%
- Drawdown máximo: 3.8%
- Consistência: Alta

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
from time import time
import asyncio

# Core imports
from core.contracts import MarketSnapshot, TradingSignal, Position
from plugins.strategies.base_strategy import BaseStrategy

# Indicadores proprietários
from indicators.momentum.momo_1_5l import MOMO_1_5L
from indicators.composite.high_aggression_score import HighAggressionScore
from execution.pocket_explosion.core import PocketExplosion

# Imports padrão para indicadores técnicos
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class Strategy15LConfig:
    """Configuração da Strategy 1.5L"""

    def __init__(self):
        # Configurações de entrada
        self.long_threshold = 0.3           # MOMO-1.5L > 0.3 para long
        self.short_threshold = -0.3         # MOMO-1.5L < -0.3 para short
        self.mean_revert_threshold = 0.8    # Extremos para mean-reversion

        # Configurações de posição
        self.base_size_pct = 0.02          # 2% equity base por trade
        self.max_size_pct = 0.04           # 4% equity máximo
        self.max_concurrent_positions = 3   # Máximo 3 posições simultâneas


class Config(Strategy15LConfig):
    """Configuração compatível da Strategy 1.5L"""
    def __init__(self):
        super().__init__()
        # Risk management
        self.stop_loss_pct = 0.008         # 0.8% hard stop
        self.take_profit_pct = 0.015       # 1.5% take profit
        self.max_hold_time_mins = 240      # 4 horas máximo

        # Confirmações técnicas
        self.require_ema_confirmation = True
        self.require_volume_confirmation = True
        self.ema_period = 13
        self.volume_multiplier = 1.2

        # Performance thresholds
        self.daily_target_pct = 0.0065     # 0.65% target diário
        self.max_daily_dd_pct = 0.038      # 3.8% max drawdown diário


class Strategy15L(BaseStrategy):
    """
    Estratégia 1.5L - Low-Boost Momentum/Mean-Revert

    Características:
    - Hibridização: Momentum baixa intensidade + Mean reversion
    - Indicador principal: MOMO-1.5L
    - Controles rigorosos de risco
    - Foco em consistência vs agressividade
    """

    def __init__(self, config: Optional[Strategy15LConfig] = None):
        self.config = config or Strategy15LConfig()

        # Inicializar indicadores proprietários
        momo_config = MOMO15LConfig(
            lookback=21,
            smooth_factor=0.65,
            boost_limit=1.5,
            normalization_window=50
        )
        self.momo_1_5l = MOMO15L(momo_config)

        # Histórico de preços e volumes para cálculos
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.timestamp_history: List[int] = []

        # Estado da estratégia
        self.active_positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_time = 0

        # Performance tracking
        self.trade_history: List[Dict] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    def generate_signal(self, snapshot: MarketSnapshot, positions: List[Position], context: Dict[str, Any]) -> Optional[TradingSignal]:
        """Método compatível para geração de sinal"""
        decision = self.decide(snapshot)

        if decision and decision.get('side') != 'HOLD':
            from datetime import datetime
            return TradingSignal(
                symbol=snapshot.symbol,
                side=decision['side'],
                signal_strength="MEDIUM",
                confidence=0.75,
                position_pct=decision.get('qty', 0.02),
                entry_price=snapshot.mid,
                timestamp=datetime.now(),
                strategy_name="Strategy-1.5L",
                metadata=decision.get('meta', {})
            )
        return None

    @staticmethod
    def required_features() -> list[str]:
        """Features necessárias no MarketSnapshot"""
        return [
            'volume',           # Volume atual
            'equity_total',     # Equity total para sizing
            'open_positions',   # Posições abertas atuais
            'daily_pnl',        # PnL do dia
            'account'           # Conta de trading
        ]

    @staticmethod
    def warmup_bars() -> int:
        """Barras necessárias para warm-up"""
        return 50  # Suficiente para MOMO-1.5L com normalização

    def decide(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Lógica principal de decisão da estratégia 1.5L

        Args:
            snapshot: Dados de mercado atuais

        Returns:
            Dict: Ordem ou sinal de trading
        """
        try:
            # 1. Update histórico de dados
            self._update_market_history(snapshot)

            # 2. Verificar se temos dados suficientes
            if len(self.price_history) < self.warmup_bars():
                return self._no_action_response("Insufficient data for analysis")

            # 3. Calcular indicadores
            indicators = self._calculate_indicators(snapshot)

            # 4. Verificar condições de risk management
            risk_check = self._check_risk_conditions(snapshot)
            if not risk_check['allowed']:
                return self._no_action_response(f"Risk check failed: {risk_check['reason']}")

            # 5. Gerar sinal de trading
            signal = self._generate_trading_signal(indicators, snapshot)

            if signal['action'] == 'HOLD':
                return self._no_action_response(signal['reason'])

            # 6. Calcular tamanho da posição
            position_size = self._calculate_position_size(signal, snapshot)

            # 7. Criar ordem
            order = self._create_order(signal, position_size, snapshot)

            return order

        except Exception as e:
            return self._no_action_response(f"Strategy error: {str(e)}")

    def _update_market_history(self, snapshot: MarketSnapshot) -> None:
        """Atualiza histórico de dados de mercado"""
        self.price_history.append(snapshot.mid)
        self.volume_history.append(snapshot.features.get('volume', 0))
        self.timestamp_history.append(snapshot.ts_ns)

        # Manter apenas últimas N barras
        max_history = 100
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.timestamp_history = self.timestamp_history[-max_history:]

    def _calculate_indicators(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Calcula todos os indicadores necessários

        Returns:
            Dict: Valores dos indicadores
        """
        indicators = {}

        # 1. MOMO-1.5L (indicador principal)
        momo_result = self.momo_1_5l.calculate(
            price_series=np.array(self.price_history),
            volume_series=np.array(self.volume_history) if self.volume_history else None
        )
        indicators['momo_1_5l'] = momo_result
        indicators['momo_signals'] = self.momo_1_5l.get_trading_signals(momo_result)

        # 2. EMA para confirmação (se disponível talib)
        if TALIB_AVAILABLE and len(self.price_history) >= self.config.ema_period:
            prices_array = np.array(self.price_history, dtype=float)
            indicators['ema'] = talib.EMA(prices_array, timeperiod=self.config.ema_period)[-1]
            indicators['price_vs_ema'] = snapshot.mid - indicators['ema']
        else:
            # EMA simples se talib não disponível
            indicators['ema'] = np.mean(self.price_history[-self.config.ema_period:])
            indicators['price_vs_ema'] = snapshot.mid - indicators['ema']

        # 3. Volume analysis
        if len(self.volume_history) >= 20:
            avg_volume = np.mean(self.volume_history[-20:])
            current_volume = self.volume_history[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            indicators['volume_ratio'] = 1.0

        return indicators

    def _generate_trading_signal(self, indicators: Dict, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Gera sinal de trading baseado nos indicadores

        Args:
            indicators: Indicadores calculados
            snapshot: Dados de mercado

        Returns:
            Dict: Sinal de trading
        """
        momo_result = indicators['momo_1_5l']
        momo_signals = indicators['momo_signals']
        momo_score = momo_result.score

        # Verificar condições de entrada LONG
        long_conditions = [
            momo_score > self.config.long_threshold,
            momo_result.signal_strength in ['moderate', 'strong'],
            indicators['price_vs_ema'] > 0 if self.config.require_ema_confirmation else True,
            indicators['volume_ratio'] > self.config.volume_multiplier if self.config.require_volume_confirmation else True
        ]

        # Verificar condições de entrada SHORT
        short_conditions = [
            momo_score < self.config.short_threshold,
            momo_result.signal_strength in ['moderate', 'strong'],
            indicators['price_vs_ema'] < 0 if self.config.require_ema_confirmation else True,
            indicators['volume_ratio'] > self.config.volume_multiplier if self.config.require_volume_confirmation else True
        ]

        # Verificar mean-reversion signals
        mean_revert_long = momo_score < -self.config.mean_revert_threshold
        mean_revert_short = momo_score > self.config.mean_revert_threshold

        # Decidir ação
        if all(long_conditions):
            return {
                'action': 'LONG',
                'reason': f'MOMO-1.5L long signal: {momo_score:.4f}',
                'conviction': 'high' if momo_result.signal_strength == 'strong' else 'medium',
                'signal_type': 'momentum'
            }

        elif all(short_conditions):
            return {
                'action': 'SHORT',
                'reason': f'MOMO-1.5L short signal: {momo_score:.4f}',
                'conviction': 'high' if momo_result.signal_strength == 'strong' else 'medium',
                'signal_type': 'momentum'
            }

        elif mean_revert_long:
            return {
                'action': 'LONG',
                'reason': f'Mean-revert long: {momo_score:.4f}',
                'conviction': 'medium',
                'signal_type': 'mean_reversion'
            }

        elif mean_revert_short:
            return {
                'action': 'SHORT',
                'reason': f'Mean-revert short: {momo_score:.4f}',
                'conviction': 'medium',
                'signal_type': 'mean_reversion'
            }

        else:
            return {
                'action': 'HOLD',
                'reason': f'No signal conditions met. MOMO: {momo_score:.4f}',
                'conviction': 'none',
                'signal_type': 'none'
            }

    def _check_risk_conditions(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Verifica condições de risk management

        Returns:
            Dict: Status de risk check
        """
        # 1. Máximo de posições simultâneas
        current_positions = len(snapshot.features.get('open_positions', []))
        if current_positions >= self.config.max_concurrent_positions:
            return {'allowed': False, 'reason': 'Max concurrent positions reached'}

        # 2. Daily drawdown check
        daily_pnl_pct = snapshot.features.get('daily_pnl', 0.0)
        if daily_pnl_pct <= -self.config.max_daily_dd_pct:
            return {'allowed': False, 'reason': f'Daily drawdown limit reached: {daily_pnl_pct:.2%}'}

        # 3. Cooldown entre trades (evitar overtrading)
        time_since_last_trade = (snapshot.ts_ns - self.last_trade_time) / 1e9  # segundos
        if time_since_last_trade < 60:  # Mínimo 1 minuto entre trades
            return {'allowed': False, 'reason': 'Trade cooldown active'}

        # 4. Verificar liquidez mínima
        if snapshot.spread / snapshot.mid > 0.005:  # Spread > 0.5%
            return {'allowed': False, 'reason': 'Spread too wide - insufficient liquidity'}

        return {'allowed': True, 'reason': 'All risk checks passed'}

    def _calculate_position_size(self, signal: Dict, snapshot: MarketSnapshot) -> float:
        """
        Calcula tamanho da posição baseado no sinal e risk management

        Args:
            signal: Sinal de trading
            snapshot: Dados de mercado

        Returns:
            float: Tamanho da posição
        """
        equity_total = snapshot.features.get('equity_total', 10000.0)

        # Tamanho base
        if signal['conviction'] == 'high':
            size_pct = self.config.max_size_pct
        elif signal['conviction'] == 'medium':
            size_pct = (self.config.base_size_pct + self.config.max_size_pct) / 2
        else:
            size_pct = self.config.base_size_pct

        # Ajuste para mean-reversion (mais conservador)
        if signal['signal_type'] == 'mean_reversion':
            size_pct *= 0.7

        # Calcular quantidade em USD
        position_value = equity_total * size_pct

        # Converter para quantity do ativo
        position_qty = position_value / snapshot.mid

        return position_qty

    def _create_order(self, signal: Dict, size: float, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Cria ordem de trading

        Args:
            signal: Sinal de trading
            size: Tamanho da posição
            snapshot: Dados de mercado

        Returns:
            Dict: Ordem formatada
        """
        side = "BUY" if signal['action'] == 'LONG' else "SELL"

        # Calcular preços de stop e take profit
        if side == "BUY":
            stop_price = snapshot.mid * (1 - self.config.stop_loss_pct)
            take_profit_price = snapshot.mid * (1 + self.config.take_profit_pct)
        else:
            stop_price = snapshot.mid * (1 + self.config.stop_loss_pct)
            take_profit_price = snapshot.mid * (1 - self.config.take_profit_pct)

        # Update tracking
        self.last_trade_time = snapshot.ts_ns
        self.daily_trades += 1

        return {
            "symbol": snapshot.symbol,
            "side": side,
            "qty": round(size, 6),  # 6 decimais para cripto
            "order_type": "MARKET",
            "client_id": "strategy.1.5l",
            "idempotency_key": f"s15l-{snapshot.symbol}-{int(time())}",
            "meta": {
                "class": snapshot.class_,
                "account": snapshot.features.get("account", "acc#1"),
                "strategy": "1.5L",
                "signal_type": signal['signal_type'],
                "conviction": signal['conviction'],
                "momo_score": str(signal.get('momo_score', 0)),
                "stop_price": str(stop_price),
                "take_profit_price": str(take_profit_price),
                "reason": signal['reason']
            }
        }

    def _no_action_response(self, reason: str) -> Dict[str, Any]:
        """Resposta quando nenhuma ação é tomada"""
        return {
            "action": "HOLD",
            "reason": reason,
            "meta": {
                "strategy": "1.5L",
                "timestamp": int(time())
            }
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Retorna métricas de performance da estratégia"""
        if self.performance_metrics['total_trades'] == 0:
            return self.performance_metrics

        win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
        avg_return = self.performance_metrics['total_pnl'] / self.performance_metrics['total_trades']

        return {
            **self.performance_metrics,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'profit_factor': self._calculate_profit_factor(),
            'trades_today': self.daily_trades
        }

    def _calculate_profit_factor(self) -> float:
        """Calcula profit factor (gross profit / gross loss)"""
        if not self.trade_history:
            return 0.0

        gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))

        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def reset_daily_stats(self):
        """Reset estatísticas diárias"""
        self.daily_pnl = 0.0
        self.daily_trades = 0

    async def async_decide(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Versão assíncrona da decisão para integração com sistema"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.decide, snapshot)


# Funções utilitárias

def create_strategy_1_5l(config_dict: Optional[Dict] = None) -> Strategy15L:
    """
    Factory function para criar Strategy15L

    Args:
        config_dict: Configurações customizadas

    Returns:
        Strategy15L: Instância configurada
    """
    if config_dict:
        config = Strategy15LConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return Strategy15L(config)
    else:
        return Strategy15L()


# Exemplo de uso e testes
if __name__ == "__main__":
    # Configuração de teste
    test_config = Strategy15LConfig()
    strategy = Strategy15L(test_config)

    # Dados de exemplo
    from backend.core.contracts import MarketSnapshot

    test_snapshot = MarketSnapshot(
        class_="CRYPTO",
        symbol="BTC/USDT",
        ts_ns=int(time() * 1e9),
        bid=50000.0,
        ask=50010.0,
        mid=50005.0,
        spread=10.0,
        features={
            'volume': 1000.0,
            'equity_total': 10000.0,
            'open_positions': [],
            'daily_pnl': 0.001,  # 0.1%
            'account': 'test_acc'
        }
    )

    # Simular histórico de preços
    base_price = 50000
    for i in range(60):
        price_change = np.random.normal(0, 0.001)  # 0.1% volatilidade
        new_price = base_price * (1 + price_change)

        snapshot = MarketSnapshot(
            class_="CRYPTO",
            symbol="BTC/USDT",
            ts_ns=int(time() * 1e9),
            bid=new_price - 5,
            ask=new_price + 5,
            mid=new_price,
            spread=10.0,
            features={
                'volume': 1000 + np.random.exponential(500),
                'equity_total': 10000.0,
                'open_positions': [],
                'daily_pnl': 0.001,
                'account': 'test_acc'
            }
        )

        decision = strategy.decide(snapshot)
        if decision.get('side'):
            print(f"Decision: {decision['side']} {decision['qty']} {decision['symbol']}")
            print(f"Reason: {decision['meta']['reason']}")
            break

        base_price = new_price

    # Performance stats
    perf = strategy.get_performance_metrics()
    print(f"Performance: {perf}")