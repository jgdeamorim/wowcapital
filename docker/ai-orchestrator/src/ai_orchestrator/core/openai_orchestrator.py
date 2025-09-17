#!/usr/bin/env python3
"""
OpenAI Orchestrator - Sistema de orquestração usando GPT-4.1-mini
Gerencia todas as chamadas para OpenAI e coordena criação de estratégias e indicadores

Autor: WOW Capital AI System
Data: 2024-09-16
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import aiohttp
from openai import AsyncOpenAI
import tiktoken


logger = logging.getLogger(__name__)


class OpenAIOrchestrator:
    """Orquestrador principal da OpenAI"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client: AsyncOpenAI = None
        self.tokenizer = None

        # Rate limiting
        self.requests_per_minute = 50
        self.tokens_per_minute = 150000
        self.request_count = 0
        self.token_count = 0

        # Context management
        self.max_context_tokens = 128000  # GPT-4o-mini context window
        self.system_prompts = {}

    async def initialize(self):
        """Inicializa o orquestrador"""
        logger.info("🤖 Inicializando OpenAI Orchestrator...")

        try:
            # Initialize OpenAI client
            self.client = AsyncOpenAI(api_key=self.api_key)

            # Initialize tokenizer
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

            # Test connection
            await self.test_connection()

            # Load system prompts
            self._load_system_prompts()

            logger.info(f"✅ OpenAI Orchestrator inicializado (Model: {self.model})")

        except Exception as e:
            logger.error(f"❌ Erro inicializando OpenAI: {str(e)}")
            raise

    async def test_connection(self):
        """Testa conexão com OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a trading system test."},
                    {"role": "user", "content": "Respond with 'OK' if you can process this message."}
                ],
                max_tokens=10
            )

            if response.choices[0].message.content.strip() == "OK":
                logger.info("✅ Conexão OpenAI testada com sucesso")
            else:
                logger.warning("⚠️ Resposta inesperada da OpenAI")

        except Exception as e:
            logger.error(f"❌ Erro testando conexão OpenAI: {str(e)}")
            raise

    def _load_system_prompts(self):
        """Carrega prompts do sistema"""
        self.system_prompts = {
            "strategy_creator": """You are an expert quantitative trading strategy developer for WOW Capital.

Your role:
1. Analyze market data and conditions
2. Create sophisticated trading strategies based on technical analysis
3. Generate Python code for strategy implementation
4. Ensure strategies include risk management
5. Provide clear documentation and parameters

Requirements:
- All strategies must be in Python
- Include entry/exit conditions
- Add position sizing logic
- Implement risk management (stop-loss, take-profit)
- Use only standard libraries (pandas, numpy, ta)
- Provide backtesting capability

Output format: JSON with strategy code, parameters, and documentation.""",

            "indicator_creator": """You are an expert technical analysis indicator developer for WOW Capital.

Your role:
1. Create custom technical indicators based on market analysis
2. Generate Python functions for indicator calculations
3. Optimize indicators for different market conditions
4. Provide mathematical explanations
5. Include visualization code

Requirements:
- All indicators must be in Python
- Use pandas and numpy for calculations
- Include parameter optimization
- Add signal generation logic
- Provide plotting capabilities
- Mathematical accuracy is critical

Output format: JSON with indicator code, parameters, and documentation.""",

            "market_analyst": """You are an expert market analyst and financial advisor for WOW Capital.

Your role:
1. Analyze current market conditions
2. Identify trading opportunities
3. Assess market sentiment and trends
4. Provide risk assessments
5. Generate actionable insights

Focus areas:
- Technical analysis patterns
- Market volatility analysis
- Correlation studies
- Risk/reward assessments
- Economic indicators impact

Output format: Structured analysis with actionable recommendations."""
        }

    async def create_strategy(self, market_data: Dict[str, Any],
                            requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Cria uma estratégia de trading personalizada"""
        logger.info("📊 Criando estratégia personalizada...")

        prompt = f"""
        Based on the following market data and requirements, create a comprehensive trading strategy:

        Market Data:
        {json.dumps(market_data, indent=2)}

        Requirements:
        {json.dumps(requirements, indent=2)}

        Create a complete strategy including:
        1. Strategy logic and rules
        2. Entry and exit conditions
        3. Risk management parameters
        4. Python implementation code
        5. Backtesting framework
        6. Performance metrics

        Ensure the strategy is:
        - Mathematically sound
        - Risk-controlled
        - Implementable in production
        - Well-documented
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompts["strategy_creator"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )

            strategy_content = response.choices[0].message.content

            # Parse and validate strategy
            strategy = self._parse_strategy_response(strategy_content)

            logger.info("✅ Estratégia criada com sucesso")
            return strategy

        except Exception as e:
            logger.error(f"❌ Erro criando estratégia: {str(e)}")
            raise

    async def create_indicator(self, indicator_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Cria um indicador técnico personalizado"""
        logger.info("📈 Criando indicador personalizado...")

        prompt = f"""
        Create a custom technical indicator based on the following specification:

        Specification:
        {json.dumps(indicator_spec, indent=2)}

        Requirements:
        1. Mathematical implementation in Python
        2. Parameter optimization capabilities
        3. Signal generation logic
        4. Visualization code
        5. Documentation and usage examples
        6. Performance considerations

        The indicator should be:
        - Numerically stable
        - Computationally efficient
        - Configurable
        - Well-tested
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompts["indicator_creator"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.1
            )

            indicator_content = response.choices[0].message.content

            # Parse and validate indicator
            indicator = self._parse_indicator_response(indicator_content)

            logger.info("✅ Indicador criado com sucesso")
            return indicator

        except Exception as e:
            logger.error(f"❌ Erro criando indicador: {str(e)}")
            raise

    async def analyze_market(self, market_data: Dict[str, Any],
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analisa condições de mercado e gera insights"""
        logger.info("🔍 Analisando condições de mercado...")

        prompt = f"""
        Analyze the following market data and provide comprehensive insights:

        Market Data:
        {json.dumps(market_data, indent=2)}

        Context (if available):
        {json.dumps(context or {}, indent=2)}

        Provide analysis including:
        1. Current market trends and patterns
        2. Volatility assessment
        3. Support/resistance levels
        4. Risk factors
        5. Trading opportunities
        6. Recommended actions
        7. Market sentiment
        8. Technical indicators signals

        Focus on actionable insights for trading decisions.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompts["market_analyst"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.2
            )

            analysis_content = response.choices[0].message.content

            # Parse analysis
            analysis = self._parse_analysis_response(analysis_content)

            logger.info("✅ Análise de mercado concluída")
            return analysis

        except Exception as e:
            logger.error(f"❌ Erro na análise de mercado: {str(e)}")
            raise

    async def optimize_parameters(self, strategy_code: str,
                                 historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Otimiza parâmetros de estratégia usando AI"""
        logger.info("⚙️ Otimizando parâmetros de estratégia...")

        prompt = f"""
        Optimize the parameters for this trading strategy based on historical data:

        Strategy Code:
        ```python
        {strategy_code}
        ```

        Historical Data Summary:
        {json.dumps(historical_data, indent=2)}

        Provide:
        1. Optimal parameter values
        2. Reasoning for each parameter choice
        3. Expected performance metrics
        4. Risk assessment
        5. Sensitivity analysis
        6. Alternative parameter sets for different market conditions

        Focus on risk-adjusted returns and drawdown minimization.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative analyst specializing in strategy optimization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.1
            )

            optimization_content = response.choices[0].message.content

            # Parse optimization results
            optimization = self._parse_optimization_response(optimization_content)

            logger.info("✅ Otimização de parâmetros concluída")
            return optimization

        except Exception as e:
            logger.error(f"❌ Erro na otimização: {str(e)}")
            raise

    def _parse_strategy_response(self, content: str) -> Dict[str, Any]:
        """Parse da resposta de criação de estratégia"""
        try:
            # Try to extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
                return json.loads(json_content)
            else:
                # Fallback: create structured response
                return {
                    "name": "AI Generated Strategy",
                    "description": "Strategy created by AI orchestrator",
                    "code": content,
                    "parameters": {},
                    "risk_management": {},
                    "performance_metrics": {}
                }
        except Exception as e:
            logger.warning(f"⚠️ Erro parsing strategy response: {str(e)}")
            return {"content": content, "parsed": False}

    def _parse_indicator_response(self, content: str) -> Dict[str, Any]:
        """Parse da resposta de criação de indicador"""
        try:
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
                return json.loads(json_content)
            else:
                return {
                    "name": "AI Generated Indicator",
                    "description": "Indicator created by AI orchestrator",
                    "code": content,
                    "parameters": {},
                    "usage": {}
                }
        except Exception as e:
            logger.warning(f"⚠️ Erro parsing indicator response: {str(e)}")
            return {"content": content, "parsed": False}

    def _parse_analysis_response(self, content: str) -> Dict[str, Any]:
        """Parse da resposta de análise de mercado"""
        return {
            "timestamp": asyncio.get_event_loop().time(),
            "analysis": content,
            "insights": [],
            "recommendations": [],
            "risk_level": "medium"
        }

    def _parse_optimization_response(self, content: str) -> Dict[str, Any]:
        """Parse da resposta de otimização"""
        return {
            "timestamp": asyncio.get_event_loop().time(),
            "optimization": content,
            "optimal_parameters": {},
            "performance_metrics": {},
            "risk_metrics": {}
        }

    def count_tokens(self, text: str) -> int:
        """Conta tokens no texto"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text.split()) * 1.3  # Rough estimation

    async def shutdown(self):
        """Finaliza o orquestrador"""
        logger.info("🛑 Finalizando OpenAI Orchestrator...")
        if self.client:
            await self.client.close()
        logger.info("✅ OpenAI Orchestrator finalizado")