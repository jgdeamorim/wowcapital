#!/usr/bin/env python3
"""
AI Orchestrator Main - Sistema de Orquestração GPT-4.1-mini
Cria estratégias e indicadores automaticamente com base em insights de mercado

Autor: WOW Capital AI System
Data: 2024-09-16
"""

import asyncio
import logging
import os
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .core.openai_orchestrator import OpenAIOrchestrator
from .core.strategy_generator import StrategyGenerator
from .core.indicator_factory import IndicatorFactory
from .core.market_rag import MarketRAG
from .core.plugin_manager import PluginManager
from .api.routes import create_routes


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIOrchestrationSystem:
    """Sistema principal de orquestração AI"""

    def __init__(self):
        self.openai_orchestrator: OpenAIOrchestrator = None
        self.strategy_generator: StrategyGenerator = None
        self.indicator_factory: IndicatorFactory = None
        self.market_rag: MarketRAG = None
        self.plugin_manager: PluginManager = None

    async def initialize(self):
        """Inicializa todos os componentes do sistema"""
        logger.info("🚀 Inicializando AI Orchestration System...")

        try:
            # Initialize OpenAI orchestrator
            self.openai_orchestrator = OpenAIOrchestrator(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini"  # Updated model name
            )
            await self.openai_orchestrator.initialize()

            # Initialize strategy generator
            self.strategy_generator = StrategyGenerator(
                orchestrator=self.openai_orchestrator
            )

            # Initialize indicator factory
            self.indicator_factory = IndicatorFactory(
                orchestrator=self.openai_orchestrator
            )

            # Initialize market RAG system
            self.market_rag = MarketRAG(
                orchestrator=self.openai_orchestrator
            )
            await self.market_rag.initialize()

            # Initialize plugin manager
            self.plugin_manager = PluginManager()
            await self.plugin_manager.initialize()

            logger.info("✅ AI Orchestration System inicializado com sucesso!")

        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {str(e)}")
            raise

    async def shutdown(self):
        """Finaliza todos os componentes"""
        logger.info("🛑 Finalizando AI Orchestration System...")

        if self.plugin_manager:
            await self.plugin_manager.shutdown()
        if self.market_rag:
            await self.market_rag.shutdown()
        if self.openai_orchestrator:
            await self.openai_orchestrator.shutdown()

        logger.info("✅ Sistema finalizado com sucesso!")


# Global system instance
ai_system = AIOrchestrationSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciamento do ciclo de vida da aplicação"""
    # Startup
    await ai_system.initialize()
    yield
    # Shutdown
    await ai_system.shutdown()


# Create FastAPI application
app = FastAPI(
    title="WOW Capital AI Orchestrator",
    description="Sistema de orquestração AI para criação automática de estratégias e indicadores",
    version="1.0.0",
    lifespan=lifespan
)

# Add routes
create_routes(app, ai_system)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "AI Orchestrator",
        "components": {
            "openai_orchestrator": ai_system.openai_orchestrator is not None,
            "strategy_generator": ai_system.strategy_generator is not None,
            "indicator_factory": ai_system.indicator_factory is not None,
            "market_rag": ai_system.market_rag is not None,
            "plugin_manager": ai_system.plugin_manager is not None
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WOW Capital AI Orchestrator",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/strategies",
            "/api/indicators",
            "/api/market-insights",
            "/api/plugins"
        ]
    }


async def main():
    """Função principal"""
    logger.info("🎯 Iniciando WOW Capital AI Orchestrator...")

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY não configurada!")
        return

    # Run server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())