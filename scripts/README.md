# 🧰 WOW Capital - Scripts Directory

Este diretório contém scripts utilitários e ferramentas de administração do sistema WOW Capital.

## 📋 Scripts Disponíveis

### 🧪 **run_all_tests.py**
**Descrição**: Executa todos os testes do sistema de forma sequencial
**Uso**:
```bash
cd scripts/
python3 run_all_tests.py
```
**Testes incluídos**:
- Testes de integração prioritários
- Testes completos do sistema
- Testes de endpoints das exchanges
- Testes do AI Orchestrator

### 🎯 **demo_trading_validator.py**
**Descrição**: Validador completo do sistema de trading demo
**Uso**:
```bash
cd scripts/
python3 demo_trading_validator.py
```
**Funcionalidades**:
- Integração Binance com demo trading
- Validação AI Orchestrator com observabilidade
- Simulação de cenários de trading realistas
- Verificação de limites de segurança

### 🐳 **run_docker_test.sh**
**Descrição**: Script para executar testes em ambiente Docker
**Uso**:
```bash
cd scripts/
./run_docker_test.sh
```
**Funcionalidades**:
- Build e execução de containers de teste
- Ambiente isolado para validação
- Limpeza automática após execução

## 🔧 Configuração

### Pré-requisitos
- Python 3.11+
- Ambiente virtual ativado (`source ../tui_env/bin/activate`)
- Variáveis de ambiente configuradas (OpenAI, exchanges)

### Variáveis de Ambiente Necessárias
```bash
export OPENAI_API_KEY="sua-chave-openai"
export BINANCE_API_KEY="sua-chave-binance"
export BINANCE_API_SECRET="seu-secret-binance"
export KRAKEN_API_KEY="sua-chave-kraken"
export KRAKEN_PRIVATE_KEY="sua-chave-privada-kraken"
```

## 📊 Uso dos Scripts

### Execução Sequencial de Testes
```bash
# Ativar ambiente
source ../tui_env/bin/activate

# Navegar para scripts
cd scripts/

# Executar todos os testes
python3 run_all_tests.py
```

### Validação Demo Trading
```bash
cd scripts/
python3 demo_trading_validator.py
```

### Testes Docker
```bash
cd scripts/
./run_docker_test.sh
```

## ⚠️ Notas Importantes

- **Todos os scripts foram corrigidos** para usar `python3` em vez de `python`
- **Caminhos relativos ajustados** para funcionar do diretório `scripts/`
- **Scripts são executáveis** e têm permissões apropriadas
- **Ambiente virtual deve estar ativo** antes da execução

## 🎯 Status dos Scripts

| Script | Status | Funcionalidade |
|--------|--------|----------------|
| `run_all_tests.py` | ✅ Corrigido | Execução completa de testes |
| `demo_trading_validator.py` | ✅ Funcional | Validação demo trading |
| `run_docker_test.sh` | ✅ Funcional | Testes em Docker |

## 🏁 Resultado Esperado

Todos os scripts agora estão organizados profissionalmente e funcionam corretamente do diretório `scripts/`, contribuindo para uma estrutura de projeto mais limpa e maintível.