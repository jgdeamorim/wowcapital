#!/usr/bin/env python3
"""
AI Orchestrator Monitor - Sistema de Observabilidade
Monitora e registra atividades do AI Orchestrator para demo trading

Funcionalidades:
- Rastreia decisões de trading
- Monitora performance em tempo real
- Coleta métricas de precisão
- Alerta sobre anomalias
- Logging estruturado

Autor: WOW Capital Trading System
Data: 2024-09-16
"""

import time
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import os

@dataclass
class TradingDecision:
    """Representa uma decisão de trading do AI"""
    timestamp: float
    symbol: str
    side: str  # BUY/SELL/HOLD
    confidence: float
    reasoning: str
    market_conditions: Dict[str, Any]
    risk_score: float
    expected_pnl_pct: float
    strategy_used: str
    execution_time_ms: float

@dataclass
class TradingExecution:
    """Representa execução real de uma ordem"""
    decision_timestamp: float
    execution_timestamp: float
    symbol: str
    side: str
    quantity: float
    price: float
    fees: float
    success: bool
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Métricas de performance do sistema"""
    total_decisions: int
    total_executions: int
    success_rate: float
    avg_decision_time_ms: float
    avg_execution_time_ms: float
    profit_loss_usd: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_profit_per_trade: float

class AIObservabilityMonitor:
    """Monitor principal de observabilidade do AI Orchestrator"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()

        # Storage for metrics
        self.decisions: deque = deque(maxlen=1000)  # Last 1000 decisions
        self.executions: deque = deque(maxlen=1000)  # Last 1000 executions
        self.performance_history: List[PerformanceMetrics] = []

        # Real-time metrics
        self.current_pnl = 0.0
        self.initial_capital = 10000.0  # Demo starting capital
        self.current_capital = self.initial_capital

        # Alerts and thresholds
        self.alert_thresholds = {
            'max_drawdown_pct': 0.05,  # 5%
            'min_success_rate': 0.60,   # 60%
            'max_decision_latency_ms': 1000,  # 1 second
            'daily_loss_limit_pct': 0.02  # 2% daily loss limit
        }

        # Demo mode safety
        self.demo_mode = True
        self.max_position_size_usd = 100.0  # Maximum $100 per position in demo

        self.logger.info("AI Orchestrator Monitor initialized in DEMO mode")

    def _setup_logging(self) -> logging.Logger:
        """Configura sistema de logging estruturado"""
        logger = logging.getLogger('ai_orchestrator_monitor')
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # File handler with JSON formatting
        file_handler = logging.FileHandler(f'logs/ai_orchestrator_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    async def log_trading_decision(self, decision: TradingDecision):
        """Registra uma decisão de trading do AI"""
        self.decisions.append(decision)

        # Log structured decision
        decision_log = {
            'event': 'trading_decision',
            'timestamp': decision.timestamp,
            'symbol': decision.symbol,
            'side': decision.side,
            'confidence': decision.confidence,
            'risk_score': decision.risk_score,
            'strategy': decision.strategy_used,
            'execution_time_ms': decision.execution_time_ms
        }

        self.logger.info(f"TRADING_DECISION: {json.dumps(decision_log)}")

        # Real-time alerts
        await self._check_decision_alerts(decision)

    async def log_trading_execution(self, execution: TradingExecution):
        """Registra execução de uma ordem"""
        self.executions.append(execution)

        # Update P&L tracking
        if execution.success:
            if execution.side.upper() == 'BUY':
                self.current_capital -= (execution.quantity * execution.price + execution.fees)
            elif execution.side.upper() == 'SELL':
                self.current_capital += (execution.quantity * execution.price - execution.fees)

        self.current_pnl = self.current_capital - self.initial_capital

        # Log structured execution
        execution_log = {
            'event': 'trading_execution',
            'timestamp': execution.execution_timestamp,
            'symbol': execution.symbol,
            'side': execution.side,
            'quantity': execution.quantity,
            'price': execution.price,
            'success': execution.success,
            'fees': execution.fees,
            'latency_ms': execution.latency_ms,
            'current_pnl': self.current_pnl
        }

        self.logger.info(f"TRADING_EXECUTION: {json.dumps(execution_log)}")

        # Real-time alerts
        await self._check_execution_alerts(execution)

    async def _check_decision_alerts(self, decision: TradingDecision):
        """Verifica alertas relacionados a decisões"""

        # High latency alert
        if decision.execution_time_ms > self.alert_thresholds['max_decision_latency_ms']:
            await self._send_alert(
                'HIGH_LATENCY',
                f"Decision latency {decision.execution_time_ms}ms exceeds threshold",
                {'symbol': decision.symbol, 'latency_ms': decision.execution_time_ms}
            )

        # Low confidence alert
        if decision.confidence < 0.5 and decision.side != 'HOLD':
            await self._send_alert(
                'LOW_CONFIDENCE',
                f"Low confidence decision: {decision.confidence}",
                {'symbol': decision.symbol, 'confidence': decision.confidence, 'side': decision.side}
            )

    async def _check_execution_alerts(self, execution: TradingExecution):
        """Verifica alertas relacionados a execuções"""

        # Execution failure
        if not execution.success:
            await self._send_alert(
                'EXECUTION_FAILURE',
                f"Failed to execute order: {execution.error_message}",
                {'symbol': execution.symbol, 'side': execution.side, 'error': execution.error_message}
            )

        # Drawdown alert
        current_return_pct = (self.current_pnl / self.initial_capital) * 100
        if current_return_pct < -self.alert_thresholds['max_drawdown_pct'] * 100:
            await self._send_alert(
                'MAX_DRAWDOWN',
                f"Maximum drawdown reached: {current_return_pct:.2f}%",
                {'current_pnl': self.current_pnl, 'return_pct': current_return_pct}
            )

    async def _send_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """Envia alerta crítico"""
        alert = {
            'event': 'ALERT',
            'type': alert_type,
            'timestamp': time.time(),
            'message': message,
            'data': data,
            'demo_mode': self.demo_mode
        }

        self.logger.warning(f"ALERT: {json.dumps(alert)}")

        # In production, this would integrate with monitoring systems
        print(f"🚨 ALERT [{alert_type}]: {message}")

    def calculate_current_metrics(self) -> PerformanceMetrics:
        """Calcula métricas atuais de performance"""

        if not self.decisions:
            return PerformanceMetrics(
                total_decisions=0, total_executions=0, success_rate=0.0,
                avg_decision_time_ms=0.0, avg_execution_time_ms=0.0,
                profit_loss_usd=0.0, return_pct=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, avg_profit_per_trade=0.0
            )

        # Basic counts
        total_decisions = len(self.decisions)
        total_executions = len([e for e in self.executions if e.success])

        # Success rate
        success_rate = (total_executions / total_decisions) if total_decisions > 0 else 0.0

        # Average times
        avg_decision_time = statistics.mean([d.execution_time_ms for d in self.decisions])
        avg_execution_time = statistics.mean([e.latency_ms for e in self.executions if e.latency_ms]) if self.executions else 0.0

        # P&L metrics
        profit_loss_usd = self.current_pnl
        return_pct = (profit_loss_usd / self.initial_capital) * 100

        # Win rate (simplified - in practice would track individual trade outcomes)
        winning_trades = len([e for e in self.executions if e.success])
        win_rate = (winning_trades / total_executions) if total_executions > 0 else 0.0

        # Simplified metrics (would be more sophisticated in production)
        sharpe_ratio = return_pct / 10 if return_pct > 0 else 0.0  # Simplified
        max_drawdown = min(0, return_pct)  # Simplified
        avg_profit_per_trade = profit_loss_usd / total_executions if total_executions > 0 else 0.0

        return PerformanceMetrics(
            total_decisions=total_decisions,
            total_executions=total_executions,
            success_rate=success_rate,
            avg_decision_time_ms=avg_decision_time,
            avg_execution_time_ms=avg_execution_time,
            profit_loss_usd=profit_loss_usd,
            return_pct=return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_profit_per_trade=avg_profit_per_trade
        )

    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Retorna dashboard em tempo real"""
        metrics = self.calculate_current_metrics()

        return {
            'timestamp': datetime.now().isoformat(),
            'demo_mode': self.demo_mode,
            'system_status': 'OPERATIONAL',
            'current_metrics': asdict(metrics),
            'capital_status': {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'unrealized_pnl': self.current_pnl,
                'return_pct': metrics.return_pct
            },
            'recent_activity': {
                'decisions_last_hour': len([d for d in self.decisions if time.time() - d.timestamp < 3600]),
                'executions_last_hour': len([e for e in self.executions if time.time() - e.execution_timestamp < 3600]),
                'last_decision_symbol': self.decisions[-1].symbol if self.decisions else None,
                'last_decision_time': datetime.fromtimestamp(self.decisions[-1].timestamp).isoformat() if self.decisions else None
            }
        }

    async def generate_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Gera relatório detalhado de performance"""
        cutoff_time = time.time() - (hours_back * 3600)

        # Filter recent data
        recent_decisions = [d for d in self.decisions if d.timestamp > cutoff_time]
        recent_executions = [e for e in self.executions if e.execution_timestamp > cutoff_time]

        # Strategy breakdown
        strategy_breakdown = defaultdict(int)
        for decision in recent_decisions:
            strategy_breakdown[decision.strategy_used] += 1

        # Symbol breakdown
        symbol_breakdown = defaultdict(int)
        for decision in recent_decisions:
            symbol_breakdown[decision.symbol] += 1

        metrics = self.calculate_current_metrics()

        report = {
            'report_timestamp': datetime.now().isoformat(),
            'period_hours': hours_back,
            'demo_mode': self.demo_mode,
            'summary_metrics': asdict(metrics),
            'activity_summary': {
                'total_decisions': len(recent_decisions),
                'total_executions': len(recent_executions),
                'strategies_used': dict(strategy_breakdown),
                'symbols_traded': dict(symbol_breakdown)
            },
            'performance_analysis': {
                'profit_factor': abs(metrics.avg_profit_per_trade) if metrics.avg_profit_per_trade != 0 else 1.0,
                'risk_return_ratio': abs(metrics.return_pct / max(abs(metrics.max_drawdown), 1)),
                'system_efficiency': metrics.success_rate * (metrics.win_rate or 0.5),
                'latency_analysis': {
                    'avg_decision_latency': metrics.avg_decision_time_ms,
                    'avg_execution_latency': metrics.avg_execution_time_ms
                }
            }
        }

        return report

# Demo testing function
async def demo_ai_monitoring():
    """Demonstra funcionalidades do monitor"""
    print("🤖 AI Orchestrator Monitor - Demo")
    print("=" * 50)

    monitor = AIObservabilityMonitor()

    # Simulate some trading decisions and executions
    print("\n📊 Simulando atividade de trading...")

    # Decision 1
    decision1 = TradingDecision(
        timestamp=time.time(),
        symbol="BTC/USDT",
        side="BUY",
        confidence=0.85,
        reasoning="Strong bullish momentum with RSI oversold recovery",
        market_conditions={"rsi": 35, "volume_ratio": 1.8, "price_change_pct": 2.3},
        risk_score=0.3,
        expected_pnl_pct=1.5,
        strategy_used="Strategy-1.5L",
        execution_time_ms=245
    )
    await monitor.log_trading_decision(decision1)

    # Execution 1
    execution1 = TradingExecution(
        decision_timestamp=decision1.timestamp,
        execution_timestamp=time.time(),
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.001,
        price=50000.0,
        fees=0.5,
        success=True,
        latency_ms=156
    )
    await monitor.log_trading_execution(execution1)

    # Decision 2 (with alert condition)
    decision2 = TradingDecision(
        timestamp=time.time() + 60,
        symbol="ETH/USDT",
        side="SELL",
        confidence=0.42,  # Low confidence - will trigger alert
        reasoning="Weak bearish signal with mixed indicators",
        market_conditions={"rsi": 55, "volume_ratio": 0.9, "price_change_pct": -0.8},
        risk_score=0.7,
        expected_pnl_pct=-0.5,
        strategy_used="Strategy-1.6",
        execution_time_ms=1200  # High latency - will trigger alert
    )
    await monitor.log_trading_decision(decision2)

    # Get real-time dashboard
    print("\n📈 Dashboard em tempo real:")
    dashboard = monitor.get_real_time_dashboard()
    print(json.dumps(dashboard, indent=2))

    # Generate performance report
    print("\n📋 Relatório de Performance:")
    report = await monitor.generate_performance_report(hours_back=1)
    print(json.dumps(report, indent=2))

    print("\n✅ Demo do AI Monitor concluída")

if __name__ == "__main__":
    asyncio.run(demo_ai_monitoring())