# WOW Capital — Plano de Testes Integrados

## 0. Preparação Geral
- [ ] Garantir `.env` atualizado com credenciais demo/testnet e segredos (Redis, Mongo, Qdrant, Grafana, OpenAI).
- [ ] Exportar `.env` para o ambiente (`set -a && source .env && set +a`).
- [ ] `docker compose -f deploy/docker-compose.yml up -d --build` com todos os serviços saudáveis (`docker compose ps`).
- [ ] Criar ambiente virtual local (`python3 -m venv .venv && source .venv/bin/activate`) ou usar o container `api` para execuções.
- [ ] Instalar dependências de teste (`pip install -r requirements-test.txt` ou `docker compose exec api pip install -r requirements-test.txt`).
- [ ] Preparar diretório `logs/test_runs/<data>` para arquivar saídas relevantes (curl, pytest, scripts, dashboards).

## 1. Infraestrutura & Segurança
- [ ] `docker compose ps` confirmando API, Redis, Mongo, Qdrant, Prometheus e Grafana.
- [ ] `docker compose exec redis redis-cli -a "$REDIS_PASSWORD" PING` → `PONG`.
- [ ] `docker compose exec mongo mongosh --username "$MONGO_ROOT_USERNAME" --password "$MONGO_ROOT_PASSWORD" --quiet --eval 'db.runCommand({ ping: 1 })'`.
- [ ] `docker compose exec qdrant curl -sSf http://localhost:6333/healthz`.
- [ ] Validar ausência de portas públicas: `ss -tlnp | grep -E '6379|27017|6333'` deve mostrar binding apenas em `0.0.0.0` interno do container.
- [ ] Verificar Grafana e Prometheus em `127.0.0.1:3000` / `9090` com credenciais definidas.

## 2. API Gateway & Endpoints Principais
- [ ] `curl -sSf http://127.0.0.1:8080/healthz` → `{ "status": "ok" }`.
- [ ] `curl -I http://127.0.0.1:8080/docs` → status 200.
- [ ] `curl -sSf http://127.0.0.1:8080/metrics/ | head` confirmando export de métricas.
- [ ] `docker compose exec api python scripts/connection_tests.py` (Redis/Mongo/Qdrant/API OK).
- [ ] `curl -sSf http://127.0.0.1:8080/md/quote?symbol=BTCUSDT\&venue=binance` → resposta válida.
- [ ] Exercitar `/exec/place` com payload simulado (via Postman ou `curl` com token) e validar `order_ack` com `accepted=False` em modo demo.
- [ ] Testar `/orchestrator/place` com snapshot sintético e confirmar registro em `backend/var/audit/trading.ndjson` + métricas `ai_orchestrator_*`.

## 3. AI Gateway & Orquestração
- [ ] `curl -X POST http://127.0.0.1:8080/ai/orchestrate -H 'X-API-Key: <key>' -d '{"prompt": "Status do mercado?"}'` (execução falsa, apenas leitura).
- [ ] `curl -X POST http://127.0.0.1:8080/ai/orchestrate -H 'X-API-Key: <key>' -d '{"prompt": "Gerar ordem demo", "execute": true}'` e validar tool-call.
- [ ] Conferir métricas `ai_tool_calls_total`, `ai_orchestrator_decisions_total`, `ai_orchestrator_execution_total` no Prometheus.
- [ ] Validar logs do monitor (`logs/ai_orchestrator_*.log`) para cada decisão/execução.

## 4. Adaptadores de Exchanges (Demo/Testnet)
- [ ] `docker compose exec api python tests/test_binance_adapter.py`.
- [ ] `docker compose exec api python tests/test_bybit_adapter.py`.
- [ ] `docker compose exec api python tests/test_kraken_adapter.py`.
- [ ] Registrar saldos/limites retornados (capturar stdout).
- [ ] Se disponível, validar posição demo com `docker compose exec api python scripts/connection_tests.py` (seções adicionais se adaptadas).

## 5. Estratégias, Indicadores e PnL
- [ ] `docker compose exec api python tests/integration_test_priority_1.py` (verificar relatório final 100%).
- [ ] `docker compose exec api python orchestrator/ai_orchestrator.py` (run curta monitorando decisões e métricas).
- [ ] Revisar indicadores no Redis (`docker compose exec api redis-cli -a "$REDIS_PASSWORD" keys 'md:*' | head`).
- [ ] Conferir agregados em Grafana (“WOW Capital API Overview”) – print ou export do painel.

## 6. IA Orquestradora — Cenários Avançados
- [ ] Executar `docker compose exec api python scripts/demo_trading_validator.py` (esperado `Status Geral: PASS`).
- [ ] Avaliar alertas gerados (baja confiança, etc.) no log do monitor.
- [ ] Testar fallback de cache (rodar `/ai/orchestrate` duas vezes com `execute=false` e confirmar hit redis).
- [ ] Ajustar overrides via Redis (`redis-cli -a "$REDIS_PASSWORD" hgetall perf:plugin:<id>`).

## 7. Observabilidade & Alertas
- [ ] Confirmar que `/metrics/` inclui os labels esperados (`ai_orchestrator_*`, `http_request_*`).
- [ ] Verificar alertas em Grafana (se houver no painel) ou configurar notificações.
- [ ] Arquivar screenshot/JSON do dashboard atualizado.

## 8. Endpoints Auxiliares & Ferramentas
- [ ] `docker compose exec api python tests/test_openai_real.py` (rotacionar chave após uso).
- [ ] `docker compose exec api python scripts/connection_tests.py --verbose` (se houver flag extra).
- [ ] Validar scripts em `scripts/` (ex.: `run_all_tests.py`, `connection_tests.py`).
- [ ] Testar CLI/TUI: `python run_tui.py` (se aplicável) em sessão isolada.

## 9. Limpeza & Documentação
- [ ] Consolidar logs e outputs em `logs/test_runs/<data>`.
- [ ] Atualizar `docs/OPERATIONS_REPORT.md` com resultados e data/hora.
- [ ] Se necessário, girar credenciais (OpenAI & exchanges) e atualizar `.env`.
- [ ] `docker compose down` quando concluir os testes.

## Registro dos Resultados
- Para cada checklist, registrar **Data/Hora**, **Responsável**, **Saída obtida**, **Status (OK/KO)**.
- Anexar prints ou JSONs relevantes (ex.: Grafana dashboard, responses de API, relatórios do validador).
- Manter histórico em `logs/test_runs/<data>/README.md` com resumo da rodada.

> Recomenda-se executar as fases na ordem apresentada. Caso algum passo falhe, documentar o erro, aplicar correção e repetir a fase antes de avançar.

