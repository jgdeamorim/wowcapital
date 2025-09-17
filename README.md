# 🚀 WOW Capital - Backend Trading System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Integration-orange.svg)](https://openai.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

Sistema de trading quantitativo com orquestração AI, integração multi-exchange e observabilidade completa.

## 📋 Índice

- [🏗️ Arquitetura](#️-arquitetura)
- [⚡ Quick Start](#-quick-start)
- [🔧 Instalação](#-instalação)
- [🏃 Execução](#-execução)
- [🧪 Testes](#-testes)
- [🐳 Docker](#-docker)
- [📊 Monitoramento](#-monitoramento)
- [🤖 AI Integration](#-ai-integration)
- [🏦 Exchanges](#-exchanges)
- [📚 Documentação](#-documentação)

---

## 🏗️ Arquitetura

### **Componentes Principais**

```
backend/
├── 🌐 api_gateway/          # FastAPI REST API + WebSocket
├── 🔄 orchestrator/         # Plugin Manager & Strategy Engine
├── 🏦 exchanges/           # Multi-exchange adapters (Binance, Kraken, Bybit)
├── 🤖 ai_gateway/          # AI Integration & RAG System
├── 🎯 execution/           # Order Router & Trade Execution
├── 📊 observability/       # Prometheus Metrics & Monitoring
├── 🔐 vault/              # Secure Credentials Management
├── 📈 indicators/         # Technical Analysis Library
├── 🧪 tests/              # Comprehensive Test Suite
├── 🧰 scripts/            # Automation & Validation Tools
└── ⚙️ config/             # Configuration Management
```

### **Stack Tecnológico**

| Componente | Tecnologia | Versão |
|------------|------------|---------|
| **API** | FastAPI + Uvicorn | 0.110.0 |
| **AI** | OpenAI GPT-4o-mini | 1.45.0 |
| **Database** | MongoDB + Redis | Motor 3.5.1 |
| **Vector DB** | Qdrant | 1.9.0 |
| **Analysis** | TA-Lib + Pandas | 0.4.28 |
| **Monitoring** | Prometheus | 0.20.0 |
| **Container** | Docker + Compose | Latest |

---

## ⚡ Quick Start

### **1. Clone & Setup**
```bash
# Clone o repositório
git clone <repository-url>
cd backend/

# Criar ambiente virtual
python3 -m venv tui_env
source tui_env/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### **2. Configurar Variáveis**
```bash
# Copiar template de configuração
cp .env.example .env

# Editar com suas credenciais
export OPENAI_API_KEY="sua-chave-openai"
export BINANCE_API_KEY="sua-chave-binance"
export KRAKEN_API_KEY="sua-chave-kraken"
```

### **3. Iniciar Serviços**
```bash
# Iniciar Redis & MongoDB
docker run -d --name redis -p 6379:6379 redis:7-alpine
docker run -d --name mongo -p 27017:27017 mongo:6

# Iniciar Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### **4. Executar Sistema**
```bash
# Iniciar TUI (Terminal UI)
python3 run_tui.py

# OU iniciar API Gateway
uvicorn api_gateway.app:app --reload --port 8080
```

---

## 🔧 Instalação

### **Pré-requisitos**
- Python 3.11+
- Docker & Docker Compose
- Git

### **Instalação Completa**

```bash
# 1. Ambiente Python
python3 -m venv tui_env
source tui_env/bin/activate

# 2. Dependências principais
pip install -r requirements.txt

# 3. Dependências de desenvolvimento (opcional)
pip install -r requirements-dev.txt

# 4. Dependências de teste (opcional)
pip install -r requirements-test.txt

# 5. TA-Lib (análise técnica)
# Ubuntu/Debian:
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make && sudo make install
pip install ta-lib

# macOS:
brew install ta-lib
pip install ta-lib
```

### **Configuração de Ambiente**

```bash
# Copiar template
cp config/.env.example .env

# Configurar variáveis essenciais
cat >> .env << EOF
# OpenAI
OPENAI_API_KEY=sua-chave-aqui
OPENAI_ORGANIZATION=sua-org-id

# Exchanges
BINANCE_API_KEY=sua-chave-binance
BINANCE_API_SECRET=seu-secret-binance
KRAKEN_API_KEY=sua-chave-kraken
KRAKEN_PRIVATE_KEY=sua-chave-privada-kraken
BYBIT_API_KEY=sua-chave-bybit
BYBIT_API_SECRET=seu-secret-bybit

# Sistema
TRADING_MODE=demo
REDIS_URL=redis://localhost:6379/0
MONGO_URI=mongodb://localhost:27017/wowcapital
QDRANT_URL=http://localhost:6333
EOF
```

---

## 🏃 Execução

### **Modos de Execução**

#### **1. 🖥️ Terminal UI (Recomendado)**
```bash
# Ativar ambiente
source tui_env/bin/activate

# Iniciar TUI completa
python3 run_tui.py
```
- Interface de terminal interativa
- Monitoramento em tempo real
- Execução de estratégias
- Observabilidade integrada

#### **2. 🌐 API Gateway**
```bash
# Desenvolvimento
uvicorn api_gateway.app:app --reload --port 8080

# Produção
uvicorn api_gateway.app:app --host 0.0.0.0 --port 8080
```
- REST API completa
- WebSocket para dados em tempo real
- Documentação automática em `/docs`

#### **3. 🤖 AI Orchestrator**
```bash
# Demo AI Orchestration
python3 scripts/demo_trading_validator.py

# AI com observabilidade completa
python3 orchestrator/ai_orchestrator.py
```

### **Endpoints Principais**

| Endpoint | Método | Descrição |
|----------|---------|-----------|
| `/` | GET | Health check |
| `/docs` | GET | Documentação Swagger |
| `/metrics` | GET | Métricas Prometheus |
| `/api/v1/strategies` | GET | Listar estratégias |
| `/api/v1/exchanges` | GET | Status exchanges |
| `/api/v1/execute` | POST | Executar estratégia |
| `/ws/market-data` | WS | Dados de mercado |

---

## 🧪 Testes

### **Suite de Testes Completa**

```bash
# Executar todos os testes
cd scripts/
python3 run_all_tests.py

# Testes específicos
python3 -m pytest tests/test_binance_adapter.py -v
python3 -m pytest tests/test_ai_orchestrator.py -v
python3 -m pytest tests/integration/ -v

# Teste OpenAI real
python3 test_openai_real.py

# Validação demo trading
python3 scripts/demo_trading_validator.py
```

### **Testes por Categoria**

#### **🔗 Testes de Integração**
```bash
# Prioridade 1 (core)
python3 tests/integration_test_priority_1.py

# Sistema completo
python3 tests/integration_test_100_percent.py
```

#### **🏦 Testes de Exchange**
```bash
# Binance
python3 tests/test_binance_adapter.py

# Kraken
python3 tests/test_kraken_adapter.py

# Bybit
python3 tests/test_bybit_adapter.py
```

#### **🤖 Testes de AI**
```bash
# AI Orchestrator
python3 tests/test_ai_orchestrator.py

# OpenAI Integration
python3 test_openai_real.py
```

### **Cobertura de Testes**

| Componente | Cobertura | Status |
|------------|-----------|---------|
| Exchange Adapters | 95% | ✅ |
| AI Orchestrator | 100% | ✅ |
| Strategy Engine | 90% | ✅ |
| API Gateway | 85% | ✅ |
| Risk Management | 100% | ✅ |

---

## 🐳 Docker

### **Docker Compose (Recomendado)**

```bash
# Serviços básicos
docker-compose up redis mongo qdrant -d

# Sistema completo
docker-compose -f docker-compose.test.yml up --build

# Ambiente de desenvolvimento
cd deploy/
docker-compose up --build
```

### **Containers Individuais**

```bash
# Redis (Cache)
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# MongoDB (Dados)
docker run -d --name mongo \
  -p 27017:27017 \
  -v mongo_data:/data/db \
  mongo:6

# Qdrant (Vector DB)
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest

# API Gateway
docker build -f Dockerfile.api -t wowcapital-api .
docker run -p 8080:8080 wowcapital-api
```

### **Docker Files**

| Arquivo | Propósito |
|---------|-----------|
| `Dockerfile.api` | API Gateway |
| `Dockerfile.test` | Ambiente de testes |
| `docker-compose.test.yml` | Suite completa |

---

## 📊 Monitoramento

### **Métricas Prometheus**

```bash
# Acessar métricas
curl http://localhost:8080/metrics

# Exemplos de métricas disponíveis:
# - trading_decisions_total
# - order_execution_duration_seconds
# - exchange_api_requests_total
# - ai_orchestrator_latency_seconds
```

### **Observabilidade**

#### **Logs Estruturados**
```bash
# Logs do sistema
tail -f logs/wowcapital.log

# Logs de AI
tail -f logs/ai_orchestrator.log

# Logs de trading
tail -f logs/trading_decisions.log
```

#### **Health Checks**
```bash
# Sistema geral
curl http://localhost:8080/health

# Exchanges
curl http://localhost:8080/api/v1/exchanges/status

# AI Services
curl http://localhost:8080/api/v1/ai/health
```

### **Alertas & Monitoring**

- **Latência alta**: > 1000ms
- **Taxa de erro**: > 5%
- **Uso de memória**: > 80%
- **Conectividade exchanges**: Down > 30s

---

## 🤖 AI Integration

### **OpenAI GPT-4o-mini**

```python
# Configuração
OPENAI_API_KEY = "sua-chave"
MODEL = "gpt-4o-mini"
CONTEXT_WINDOW = 128000  # tokens
RATE_LIMIT = "50 req/min"
```

### **Funcionalidades AI**

#### **1. Strategy Generation**
```bash
# Gerar estratégia automaticamente
python3 -c "
from orchestrator.ai_orchestrator import AIOrchestrator
ai = AIOrchestrator()
strategy = ai.generate_strategy('momentum', 'BTCUSDT')
print(strategy)
"
```

#### **2. Market Analysis**
```python
# Análise de mercado com RAG
from ai_gateway.rag_system import RAGSystem
rag = RAGSystem()
analysis = rag.analyze_market('BTCUSDT', 'current trends')
```

#### **3. Risk Assessment**
```python
# Avaliação de risco via AI
from orchestrator.risk_ai import RiskAI
risk_ai = RiskAI()
risk_score = risk_ai.assess_risk(portfolio, market_data)
```

### **RAG System (Qdrant)**

```bash
# Testar RAG
python3 -c "
from ai_gateway.rag_system import RAGSystem
rag = RAGSystem('http://localhost:6333')
rag.create_collection('trading_knowledge')
response = rag.query('What is momentum trading?')
print(response)
"
```

---

## 🏦 Exchanges

### **Exchanges Suportadas**

| Exchange | Status | Features |
|----------|--------|----------|
| **Binance** | ✅ Operacional | Spot, Futures, WebSocket |
| **Kraken** | ✅ Operacional | Spot, Margin, REST API |
| **Bybit** | ✅ Operacional | Derivatives, Spot |

### **Configuração de Exchanges**

#### **Binance**
```python
# config/exchanges.yaml
binance:
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
  testnet: true  # Para desenvolvimento
  endpoints:
    spot: https://api.binance.com
    futures: https://fapi.binance.com
```

#### **Kraken**
```python
# Configuração Kraken
kraken:
  api_key: ${KRAKEN_API_KEY}
  private_key: ${KRAKEN_PRIVATE_KEY}
  environment: sandbox  # sandbox | live
```

#### **Bybit**
```python
# Configuração Bybit
bybit:
  api_key: ${BYBIT_API_KEY}
  api_secret: ${BYBIT_API_SECRET}
  testnet: true
  category: spot  # spot | linear | inverse
```

### **Teste de Conectividade**

```bash
# Testar todas as exchanges
python3 scripts/demo_trading_validator.py

# Teste específico Binance
python3 tests/test_binance_detailed.py

# Teste específico Kraken
python3 tests/test_kraken_adapter.py
```

---

## 📚 Documentação

### **Documentação Adicional**

| Documento | Descrição |
|-----------|-----------|
| [`docs/ai-orchestration-system.md`](../docs/ai-orchestration-system.md) | Sistema AI completo |
| [`docs/final-system-validation.md`](../docs/final-system-validation.md) | Validação final |
| [`docs/SISTEMA_COMPLETO_VALIDADO.md`](../docs/SISTEMA_COMPLETO_VALIDADO.md) | Sistema validado |
| [`scripts/README.md`](scripts/README.md) | Documentação scripts |

### **Arquitetura Detalhada**

#### **Plugin System**
```
plugins/
├── strategies/          # Estratégias de trading
│   ├── strategy_1_5l.py    # Estratégia 1.5L
│   ├── strategy_1_6.py     # Estratégia 1.6
│   └── strategy_1_6pp_r.py # Estratégia 1.6pp-R
├── indicators/         # Indicadores técnicos
│   ├── ob_score.py         # OrderBook Score
│   ├── ob_flow.py          # OrderBook Flow
│   └── squeeze_sigma.py    # Volatility Compression
└── examples/           # Exemplos e templates
```

#### **Configuration System**
```
config/
├── models.yaml         # Configuração AI models
├── rag.yaml           # RAG system config
├── orchestrator.yaml  # Orchestrator settings
├── plugins.yaml       # Plugin management
├── exchanges.yaml     # Exchange configurations
└── accounts.yaml      # Account settings
```

### **APIs e Integrações**

#### **REST API Endpoints**
```
GET    /                      # Health check
GET    /docs                  # Swagger documentation
GET    /metrics               # Prometheus metrics
GET    /api/v1/strategies     # List strategies
POST   /api/v1/execute       # Execute strategy
GET    /api/v1/exchanges     # Exchange status
POST   /api/v1/orders        # Create order
GET    /api/v1/positions     # Get positions
```

#### **WebSocket Channels**
```
/ws/market-data/{symbol}     # Market data stream
/ws/orders                   # Order updates
/ws/positions                # Position updates
/ws/trades                   # Trade execution
```

---

## 🔐 Segurança

### **Credentials Management**

```bash
# Configurar vault de credenciais
python3 -c "
from vault.credentials_vault import CredentialsVault
vault = CredentialsVault()
vault.store_credential('binance', 'api_key', 'sua-chave')
vault.store_credential('binance', 'api_secret', 'seu-secret')
"
```

### **Rate Limiting**

```python
# Configuração automática de rate limits
BINANCE_RATE_LIMIT = "1200/min"
KRAKEN_RATE_LIMIT = "15/min"
BYBIT_RATE_LIMIT = "100/min"
```

### **Risk Management**

```python
# Limites de segurança configuráveis
MAX_POSITION_SIZE_USD = 1000  # Demo mode
MAX_DAILY_LOSS_USD = 500
ENABLE_REAL_TRADING = False   # Sempre False em dev
```

---

## 🚀 Deploy

### **Ambiente de Produção**

```bash
# Build imagem de produção
docker build -f Dockerfile.api -t wowcapital:latest .

# Deploy com docker-compose
cd deploy/
docker-compose up -d

# Verificar status
docker-compose ps
docker-compose logs -f api
```

### **Monitoramento Produção**

```bash
# Métricas Prometheus
curl http://production-host:8080/metrics

# Health check
curl http://production-host:8080/health

# Logs
docker-compose logs --tail=100 api
```

---

## 🤝 Contribuição

### **Desenvolvimento**

```bash
# Setup desenvolvimento
git clone <repo>
cd backend/
python3 -m venv tui_env
source tui_env/bin/activate
pip install -r requirements-dev.txt

# Executar testes
python3 scripts/run_all_tests.py

# Lint & format
flake8 .
black .
isort .
```

### **Estrutura de Commits**

```
feat: nova funcionalidade
fix: correção de bug
docs: atualização documentação
test: adição de testes
refactor: refatoração de código
```

---

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 📞 Suporte

- **Documentação**: [`docs/`](../docs/)
- **Issues**: Reporte problemas no repositório
- **Discussões**: Use as discussões do GitHub

---

**🎯 WOW Capital Trading System - Construído com excelência para trading quantitativo profissional.**

*Última atualização: 2025-09-17*