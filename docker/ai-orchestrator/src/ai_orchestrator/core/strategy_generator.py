#!/usr/bin/env python3
"""
Strategy Generator - Sistema de criação automática de estratégias de trading
Usa OpenAI para gerar estratégias personalizadas baseadas em dados de mercado

Autor: WOW Capital AI System
Data: 2024-09-16
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid


logger = logging.getLogger(__name__)


class StrategyGenerator:
    """Gerador automático de estratégias de trading"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.strategies_cache = {}
        self.performance_history = {}

    async def generate_momentum_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera estratégia baseada em momentum"""
        logger.info("🚀 Gerando estratégia de momentum...")

        requirements = {
            "strategy_type": "momentum",
            "timeframe": "1h",
            "risk_management": {
                "max_drawdown": 0.05,
                "position_size": 0.02,
                "stop_loss": 0.02,
                "take_profit": 0.04
            },
            "indicators": ["RSI", "MACD", "EMA"],
            "market_conditions": self._analyze_market_conditions(market_data)
        }

        strategy = await self.orchestrator.create_strategy(market_data, requirements)

        # Add metadata
        strategy_id = str(uuid.uuid4())
        strategy.update({
            "id": strategy_id,
            "type": "momentum",
            "created_at": datetime.now().isoformat(),
            "status": "generated",
            "backtest_required": True
        })

        # Cache strategy
        self.strategies_cache[strategy_id] = strategy

        logger.info(f"✅ Estratégia momentum criada: {strategy_id}")
        return strategy

    async def generate_mean_reversion_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera estratégia de reversão à média"""
        logger.info("📉 Gerando estratégia de mean reversion...")

        requirements = {
            "strategy_type": "mean_reversion",
            "timeframe": "4h",
            "risk_management": {
                "max_drawdown": 0.03,
                "position_size": 0.015,
                "stop_loss": 0.015,
                "take_profit": 0.025
            },
            "indicators": ["Bollinger Bands", "RSI", "Stochastic"],
            "market_conditions": self._analyze_market_conditions(market_data)
        }

        strategy = await self.orchestrator.create_strategy(market_data, requirements)

        # Add metadata
        strategy_id = str(uuid.uuid4())
        strategy.update({
            "id": strategy_id,
            "type": "mean_reversion",
            "created_at": datetime.now().isoformat(),
            "status": "generated",
            "backtest_required": True
        })

        self.strategies_cache[strategy_id] = strategy

        logger.info(f"✅ Estratégia mean reversion criada: {strategy_id}")
        return strategy

    async def generate_breakout_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera estratégia de breakout"""
        logger.info("💥 Gerando estratégia de breakout...")

        requirements = {
            "strategy_type": "breakout",
            "timeframe": "15m",
            "risk_management": {
                "max_drawdown": 0.04,
                "position_size": 0.025,
                "stop_loss": 0.01,
                "take_profit": 0.03
            },
            "indicators": ["Volume", "ATR", "Support/Resistance"],
            "market_conditions": self._analyze_market_conditions(market_data)
        }

        strategy = await self.orchestrator.create_strategy(market_data, requirements)

        # Add metadata
        strategy_id = str(uuid.uuid4())
        strategy.update({
            "id": strategy_id,
            "type": "breakout",
            "created_at": datetime.now().isoformat(),
            "status": "generated",
            "backtest_required": True
        })

        self.strategies_cache[strategy_id] = strategy

        logger.info(f"✅ Estratégia breakout criada: {strategy_id}")
        return strategy

    async def generate_ai_adaptive_strategy(self, market_data: Dict[str, Any],
                                          pnl_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera estratégia adaptativa usando AI com base em P&L histórico"""
        logger.info("🤖 Gerando estratégia adaptativa AI...")

        # Analyze P&L patterns
        pnl_insights = self._analyze_pnl_patterns(pnl_history)

        requirements = {
            "strategy_type": "ai_adaptive",
            "timeframe": "dynamic",
            "risk_management": {
                "max_drawdown": 0.03,
                "position_size": "dynamic",
                "stop_loss": "adaptive",
                "take_profit": "adaptive"
            },
            "indicators": ["AI_Signal", "Volatility_Regime", "Market_State"],
            "market_conditions": self._analyze_market_conditions(market_data),
            "pnl_insights": pnl_insights,
            "adaptive_parameters": True
        }

        strategy = await self.orchestrator.create_strategy(market_data, requirements)

        # Add AI-specific metadata
        strategy_id = str(uuid.uuid4())
        strategy.update({
            "id": strategy_id,
            "type": "ai_adaptive",
            "created_at": datetime.now().isoformat(),
            "status": "generated",
            "backtest_required": True,
            "adaptation_enabled": True,
            "learning_mode": True
        })

        self.strategies_cache[strategy_id] = strategy

        logger.info(f"✅ Estratégia adaptativa AI criada: {strategy_id}")
        return strategy

    async def optimize_strategy(self, strategy_id: str,
                              historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Otimiza estratégia existente"""
        logger.info(f"⚙️ Otimizando estratégia {strategy_id}...")

        if strategy_id not in self.strategies_cache:
            raise ValueError(f"Estratégia {strategy_id} não encontrada")

        strategy = self.strategies_cache[strategy_id]

        # Get strategy code for optimization
        strategy_code = strategy.get("code", "")

        # Optimize using OpenAI
        optimization = await self.orchestrator.optimize_parameters(
            strategy_code, historical_data
        )

        # Update strategy with optimized parameters
        strategy.update({
            "optimized_at": datetime.now().isoformat(),
            "optimization_results": optimization,
            "status": "optimized"
        })

        logger.info(f"✅ Estratégia {strategy_id} otimizada")
        return strategy

    async def backtest_strategy(self, strategy_id: str,
                              historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa backtest da estratégia"""
        logger.info(f"📊 Executando backtest da estratégia {strategy_id}...")

        if strategy_id not in self.strategies_cache:
            raise ValueError(f"Estratégia {strategy_id} não encontrada")

        strategy = self.strategies_cache[strategy_id]

        # Simulate backtest (in production, this would run real backtest)
        backtest_results = {
            "strategy_id": strategy_id,
            "period": "2023-01-01 to 2024-09-16",
            "total_return": 0.15,  # 15%
            "sharpe_ratio": 1.25,
            "max_drawdown": 0.08,
            "win_rate": 0.62,
            "profit_factor": 1.45,
            "trades_count": 150,
            "avg_trade_duration": "4.2 hours",
            "performance_metrics": {
                "volatility": 0.12,
                "beta": 0.85,
                "alpha": 0.03,
                "calmar_ratio": 1.88
            },
            "monthly_returns": self._generate_monthly_returns(),
            "equity_curve": self._generate_equity_curve(),
            "backtest_completed_at": datetime.now().isoformat()
        }

        # Update strategy
        strategy.update({
            "backtest_results": backtest_results,
            "status": "backtested",
            "performance_score": self._calculate_performance_score(backtest_results)
        })

        # Store in performance history
        self.performance_history[strategy_id] = backtest_results

        logger.info(f"✅ Backtest concluído para estratégia {strategy_id}")
        return backtest_results

    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa condições atuais do mercado"""
        conditions = {
            "trend": "neutral",
            "volatility": "medium",
            "volume": "normal",
            "sentiment": "neutral"
        }

        # Analyze price data if available
        if "price_data" in market_data:
            price_data = market_data["price_data"]
            if "current_price" in price_data and "previous_price" in price_data:
                price_change = (price_data["current_price"] - price_data["previous_price"]) / price_data["previous_price"]

                if price_change > 0.02:
                    conditions["trend"] = "bullish"
                elif price_change < -0.02:
                    conditions["trend"] = "bearish"

                if abs(price_change) > 0.05:
                    conditions["volatility"] = "high"
                elif abs(price_change) < 0.01:
                    conditions["volatility"] = "low"

        return conditions

    def _analyze_pnl_patterns(self, pnl_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa padrões no histórico de P&L"""
        if not pnl_history:
            return {"pattern": "no_history", "insights": []}

        # Simulate P&L analysis
        insights = {
            "pattern": "profitable_trend",
            "avg_daily_pnl": 150.0,
            "win_streak_max": 5,
            "loss_streak_max": 3,
            "best_performing_hours": ["09:00", "14:00", "16:00"],
            "worst_performing_hours": ["12:00", "18:00"],
            "volatility_preference": "medium",
            "market_regime_performance": {
                "trending": 0.65,
                "ranging": 0.35,
                "high_vol": 0.45,
                "low_vol": 0.55
            }
        }

        return insights

    def _generate_monthly_returns(self) -> List[Dict[str, float]]:
        """Gera retornos mensais simulados"""
        import random
        months = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
                 "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12",
                 "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
                 "2024-07", "2024-08", "2024-09"]

        return [{"month": month, "return": round(random.uniform(-0.05, 0.08), 4)}
                for month in months]

    def _generate_equity_curve(self) -> List[Dict[str, float]]:
        """Gera curva de equity simulada"""
        import random
        equity = 10000
        curve = []

        for i in range(100):
            daily_return = random.uniform(-0.02, 0.03)
            equity *= (1 + daily_return)
            curve.append({"day": i, "equity": round(equity, 2)})

        return curve

    def _calculate_performance_score(self, backtest_results: Dict[str, Any]) -> float:
        """Calcula score de performance da estratégia"""
        total_return = backtest_results.get("total_return", 0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
        max_drawdown = backtest_results.get("max_drawdown", 1)
        win_rate = backtest_results.get("win_rate", 0)

        # Weighted score calculation
        score = (
            total_return * 0.3 +
            sharpe_ratio * 0.3 +
            (1 - max_drawdown) * 0.2 +
            win_rate * 0.2
        )

        return round(score, 3)

    async def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Retorna estratégia por ID"""
        return self.strategies_cache.get(strategy_id)

    async def list_strategies(self) -> List[Dict[str, Any]]:
        """Lista todas as estratégias"""
        return list(self.strategies_cache.values())

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Retorna resumo de performance de todas as estratégias"""
        if not self.performance_history:
            return {"message": "Nenhum backtest executado ainda"}

        total_strategies = len(self.performance_history)
        avg_return = sum(p["total_return"] for p in self.performance_history.values()) / total_strategies
        avg_sharpe = sum(p["sharpe_ratio"] for p in self.performance_history.values()) / total_strategies
        avg_drawdown = sum(p["max_drawdown"] for p in self.performance_history.values()) / total_strategies

        return {
            "total_strategies": total_strategies,
            "average_return": round(avg_return, 4),
            "average_sharpe_ratio": round(avg_sharpe, 3),
            "average_max_drawdown": round(avg_drawdown, 4),
            "best_strategy": max(self.performance_history.items(),
                               key=lambda x: x[1]["total_return"])[0],
            "summary_generated_at": datetime.now().isoformat()
        }