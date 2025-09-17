#!/usr/bin/env python3
"""
API Routes - Rotas da API para orquestração AI
Define endpoints para interação com o sistema de orquestração

Autor: WOW Capital AI System
Data: 2024-09-16
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel


logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class StrategyRequest(BaseModel):
    strategy_type: str
    market_data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None


class IndicatorRequest(BaseModel):
    indicator_type: str
    market_data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None


class PluginActivationRequest(BaseModel):
    plugin_id: str
    parameters: Optional[Dict[str, Any]] = None


class ParameterUpdateRequest(BaseModel):
    plugin_id: str
    new_parameters: Dict[str, Any]


class MarketAnalysisRequest(BaseModel):
    market_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


def create_routes(app, ai_system):
    """Cria todas as rotas da API"""

    # Strategy routes
    @app.post("/api/strategies/generate")
    async def generate_strategy(request: StrategyRequest):
        """Gera uma nova estratégia"""
        try:
            if request.strategy_type == "momentum":
                strategy = await ai_system.strategy_generator.generate_momentum_strategy(
                    request.market_data
                )
            elif request.strategy_type == "mean_reversion":
                strategy = await ai_system.strategy_generator.generate_mean_reversion_strategy(
                    request.market_data
                )
            elif request.strategy_type == "breakout":
                strategy = await ai_system.strategy_generator.generate_breakout_strategy(
                    request.market_data
                )
            elif request.strategy_type == "ai_adaptive":
                pnl_history = request.market_data.get("pnl_history", [])
                strategy = await ai_system.strategy_generator.generate_ai_adaptive_strategy(
                    request.market_data, pnl_history
                )
            else:
                raise HTTPException(status_code=400, detail=f"Strategy type {request.strategy_type} not supported")

            return {"status": "success", "strategy": strategy}

        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/strategies")
    async def list_strategies():
        """Lista todas as estratégias"""
        try:
            strategies = await ai_system.strategy_generator.list_strategies()
            return {"strategies": strategies}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/strategies/{strategy_id}")
    async def get_strategy(strategy_id: str):
        """Obtém estratégia específica"""
        try:
            strategy = await ai_system.strategy_generator.get_strategy(strategy_id)
            if not strategy:
                raise HTTPException(status_code=404, detail="Strategy not found")
            return {"strategy": strategy}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/strategies/{strategy_id}/optimize")
    async def optimize_strategy(strategy_id: str, historical_data: Dict[str, Any]):
        """Otimiza uma estratégia"""
        try:
            result = await ai_system.strategy_generator.optimize_strategy(
                strategy_id, historical_data
            )
            return {"status": "success", "optimization": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/strategies/{strategy_id}/backtest")
    async def backtest_strategy(strategy_id: str, historical_data: Dict[str, Any]):
        """Executa backtest de uma estratégia"""
        try:
            result = await ai_system.strategy_generator.backtest_strategy(
                strategy_id, historical_data
            )
            return {"status": "success", "backtest": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/strategies/performance/summary")
    async def get_strategies_performance():
        """Obtém resumo de performance das estratégias"""
        try:
            summary = await ai_system.strategy_generator.get_performance_summary()
            return {"performance_summary": summary}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Indicator routes
    @app.post("/api/indicators/generate")
    async def generate_indicator(request: IndicatorRequest):
        """Gera um novo indicador"""
        try:
            if request.indicator_type == "ai_momentum":
                indicator = await ai_system.indicator_factory.create_ai_momentum_indicator(
                    request.market_data
                )
            elif request.indicator_type == "volatility_regime":
                indicator = await ai_system.indicator_factory.create_volatility_regime_indicator(
                    request.market_data
                )
            elif request.indicator_type == "microstructure":
                indicator = await ai_system.indicator_factory.create_market_microstructure_indicator(
                    request.market_data
                )
            elif request.indicator_type == "sentiment_fusion":
                sentiment_data = request.market_data.get("sentiment_data", {})
                indicator = await ai_system.indicator_factory.create_sentiment_fusion_indicator(
                    request.market_data, sentiment_data
                )
            elif request.indicator_type == "support_resistance":
                indicator = await ai_system.indicator_factory.create_adaptive_support_resistance(
                    request.market_data
                )
            else:
                raise HTTPException(status_code=400, detail=f"Indicator type {request.indicator_type} not supported")

            return {"status": "success", "indicator": indicator}

        except Exception as e:
            logger.error(f"Error generating indicator: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/indicators")
    async def list_indicators():
        """Lista todos os indicadores"""
        try:
            indicators = await ai_system.indicator_factory.list_indicators()
            return {"indicators": indicators}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/indicators/{indicator_id}")
    async def get_indicator(indicator_id: str):
        """Obtém indicador específico"""
        try:
            indicator = await ai_system.indicator_factory.get_indicator(indicator_id)
            if not indicator:
                raise HTTPException(status_code=404, detail="Indicator not found")
            return {"indicator": indicator}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/indicators/{indicator_id}/optimize")
    async def optimize_indicator(indicator_id: str, historical_data: Dict[str, Any]):
        """Otimiza um indicador"""
        try:
            result = await ai_system.indicator_factory.optimize_indicator(
                indicator_id, historical_data
            )
            return {"status": "success", "optimization": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/indicators/{indicator_id}/backtest")
    async def backtest_indicator(indicator_id: str, historical_data: Dict[str, Any]):
        """Executa backtest de um indicador"""
        try:
            result = await ai_system.indicator_factory.backtest_indicator(
                indicator_id, historical_data
            )
            return {"status": "success", "backtest": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/indicators/performance/summary")
    async def get_indicators_performance():
        """Obtém resumo de performance dos indicadores"""
        try:
            summary = await ai_system.indicator_factory.get_performance_summary()
            return {"performance_summary": summary}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Market RAG routes
    @app.post("/api/market/analyze")
    async def analyze_market(request: MarketAnalysisRequest):
        """Analisa sentimento e condições de mercado"""
        try:
            analysis = await ai_system.market_rag.analyze_market_sentiment(
                request.market_data
            )
            return {"status": "success", "analysis": analysis}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/market/pnl-insights")
    async def get_pnl_insights(pnl_data: List[Dict[str, Any]]):
        """Gera insights sobre padrões de P&L"""
        try:
            insights = await ai_system.market_rag.generate_pnl_insights(pnl_data)
            return {"status": "success", "insights": insights}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/market/insights")
    async def get_contextual_insights(query: str, market_data: Optional[Dict[str, Any]] = None):
        """Obtém insights contextuais"""
        try:
            insights = await ai_system.market_rag.get_contextual_insights(query, market_data)
            return {"status": "success", "insights": insights}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/market/predict")
    async def predict_market_movement(market_data: Dict[str, Any], timeframe: str = "1h"):
        """Prediz movimento de mercado"""
        try:
            prediction = await ai_system.market_rag.predict_market_movement(
                market_data, timeframe
            )
            return {"status": "success", "prediction": prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Plugin management routes
    @app.get("/api/plugins/available")
    async def get_available_plugins():
        """Lista plugins disponíveis"""
        try:
            plugins = await ai_system.plugin_manager.get_available_plugins()
            return {"plugins": plugins}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/plugins/active")
    async def get_active_plugins():
        """Lista plugins ativos"""
        try:
            active = await ai_system.plugin_manager.get_active_plugins()
            return {"active_plugins": active}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/strategies/activate")
    async def activate_strategy_plugin(request: PluginActivationRequest):
        """Ativa uma estratégia"""
        try:
            result = await ai_system.plugin_manager.activate_strategy(
                request.plugin_id, request.parameters
            )
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/strategies/deactivate")
    async def deactivate_strategy_plugin(plugin_id: str):
        """Desativa uma estratégia"""
        try:
            result = await ai_system.plugin_manager.deactivate_strategy(plugin_id)
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/indicators/activate")
    async def activate_indicator_plugin(request: PluginActivationRequest):
        """Ativa um indicador"""
        try:
            result = await ai_system.plugin_manager.activate_indicator(
                request.plugin_id, request.parameters
            )
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/indicators/deactivate")
    async def deactivate_indicator_plugin(plugin_id: str):
        """Desativa um indicador"""
        try:
            result = await ai_system.plugin_manager.deactivate_indicator(plugin_id)
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/strategies/update-parameters")
    async def update_strategy_parameters(request: ParameterUpdateRequest):
        """Atualiza parâmetros de estratégia em runtime"""
        try:
            result = await ai_system.plugin_manager.update_strategy_parameters(
                request.plugin_id, request.new_parameters
            )
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/indicators/update-parameters")
    async def update_indicator_parameters(request: ParameterUpdateRequest):
        """Atualiza parâmetros de indicador em runtime"""
        try:
            result = await ai_system.plugin_manager.update_indicator_parameters(
                request.plugin_id, request.new_parameters
            )
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/plugins/{plugin_id}/status")
    async def get_plugin_status(plugin_id: str):
        """Obtém status de um plugin"""
        try:
            status = await ai_system.plugin_manager.get_plugin_status(plugin_id)
            return {"status": status}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/plugins/execute")
    async def execute_active_plugins(market_data: Dict[str, Any]):
        """Executa todos os plugins ativos"""
        try:
            # Execute strategies
            strategy_signals = await ai_system.plugin_manager.execute_strategy_signals(market_data)

            # Calculate indicators
            indicator_values = await ai_system.plugin_manager.calculate_indicator_values(market_data)

            return {
                "status": "success",
                "strategy_signals": strategy_signals,
                "indicator_values": indicator_values,
                "execution_summary": {
                    "total_strategies_executed": strategy_signals.get("successful_executions", 0),
                    "total_indicators_calculated": indicator_values.get("successful_calculations", 0),
                    "total_errors": len(strategy_signals.get("errors", [])) + len(indicator_values.get("errors", []))
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/plugins/performance")
    async def get_plugins_performance():
        """Obtém resumo de performance dos plugins"""
        try:
            performance = await ai_system.plugin_manager.get_performance_summary()
            return {"performance": performance}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # System monitoring routes
    @app.get("/api/system/status")
    async def get_system_status():
        """Obtém status completo do sistema"""
        try:
            return {
                "system": "AI Orchestrator",
                "status": "operational",
                "components": {
                    "openai_orchestrator": ai_system.openai_orchestrator is not None,
                    "strategy_generator": ai_system.strategy_generator is not None,
                    "indicator_factory": ai_system.indicator_factory is not None,
                    "market_rag": ai_system.market_rag is not None,
                    "plugin_manager": ai_system.plugin_manager is not None
                },
                "active_plugins": await ai_system.plugin_manager.get_active_plugins(),
                "model": ai_system.openai_orchestrator.model if ai_system.openai_orchestrator else None
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/system/test")
    async def test_system():
        """Executa teste completo do sistema"""
        try:
            # Test market data
            test_market_data = {
                "price": 50000.0,
                "volume": 1000000,
                "timestamp": "2024-09-16T10:00:00Z",
                "volatility": 0.15,
                "trend": "bullish"
            }

            # Generate test strategy
            test_strategy = await ai_system.strategy_generator.generate_momentum_strategy(test_market_data)

            # Generate test indicator
            test_indicator = await ai_system.indicator_factory.create_ai_momentum_indicator(test_market_data)

            # Test market analysis
            test_analysis = await ai_system.market_rag.analyze_market_sentiment(test_market_data)

            return {
                "status": "success",
                "test_results": {
                    "strategy_generation": "passed",
                    "indicator_creation": "passed",
                    "market_analysis": "passed",
                    "system_health": "operational"
                },
                "test_data": {
                    "strategy_id": test_strategy.get("id"),
                    "indicator_id": test_indicator.get("id"),
                    "analysis_timestamp": test_analysis.get("timestamp")
                }
            }
        except Exception as e:
            logger.error(f"System test failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"System test failed: {str(e)}")

    return app