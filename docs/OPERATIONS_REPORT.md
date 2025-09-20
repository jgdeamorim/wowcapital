# WOW Capital — Operações no Servidor `ny4-1`

## Visão Geral
- Repositório: `/opt/wowcapital`
- Stack: Docker Compose (API, Redis, MongoDB, Qdrant)
- Imagem atual da API: `wowcapital-api:latest` (Python 3.11, TA-Lib C 0.4.0, wrapper 0.4.26)
- Arquivo Compose: `/opt/wowcapital/deploy/docker-compose.yml`
- `.env`: `/opt/wowcapital/.env` (não versionar; preencher com chaves reais)

### Comandos padrão
```bash
# Build / rebuild
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml build

# Subir/derrubar
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml up -d
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml down

# Status & logs
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml ps
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml logs -f api
```

### Status atual (2025-09-20 12:10 UTC)
```
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml ps
NAME                IMAGE                   STATUS      PORTS
wowcapital-api      wowcapital-api:latest   Up 23 mins  127.0.0.1:8080->8080/tcp
wowcapital-mongo    mongo:6                 Up 23 mins
wowcapital-qdrant   qdrant/qdrant:latest    Up 23 mins
wowcapital-redis    redis:7-alpine          Up 23 mins
wowcapital-prometheus prom/prometheus:latest Up 23 mins  127.0.0.1:9090->9090/tcp
wowcapital-grafana  grafana/grafana:10.4.5   Up 23 mins  127.0.0.1:3000->3000/tcp
```
*(Redis/Mongo/Qdrant acessíveis somente via rede interna do Compose. API HTTP, Prometheus e Grafana expostos apenas em `127.0.0.1`.)*

## Testes de conectividade internos
```
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec api python scripts/connection_tests.py
== Redis ==         PING: True
== Mongo ==         PING: {'ok': 1.0}
== Qdrant ==        HEALTH: healthz check passed
== API ==           /docs status: 200
```
- `/usr/bin/curl -sSf http://localhost:8080/docs | head -n 1` → `<!DOCTYPE html>`
- `/usr/bin/curl -sSf http://localhost:6333/healthz` → `healthz check passed`
- `/usr/bin/docker compose ... exec redis redis-cli PING` → `PONG`
- `/usr/bin/docker compose ... exec mongo mongosh --eval "db.runCommand({ ping: 1 })"` → `{ ok: 1 }`

Os logs recentes da API aparecem limpos após o rebuild (ver `docker compose logs -f api`).

## Testes funcionais avançados (exchanges & OpenAI)
### Rodada 2025-09-20 11:12 UTC
| Comando (dentro do container `api`) | Resultado |
|------------------------------------|-----------|
| `python tests/test_binance_adapter.py` | ✅ Sucesso com chaves `BINANCE_*` atuais (testnet). |
| `python tests/test_kraken_adapter.py` | ✅ Sucesso com `KRAKEN_ENVIRONMENT=sandbox`. |
| `python tests/test_bybit_adapter.py` | ✅ Sucesso com `BYBIT_TESTNET=1`. |
| `python tests/integration_test_priority_1.py` | ✅ Passou (100% dos componentes prioridade 1 aprovados). |
| `python tests/test_openai_real.py` | ✅ Sucesso (`sk-proj-…90A` validou completions e embeddings). |
| `PYTHONPATH=/app:/app/backend python scripts/demo_trading_validator.py` | ✅ Sucesso (relatório `PASS` com credenciais demo/testnet ativas). |

### Próximos passos sugeridos
1. **Trocar/garantir credenciais válidas**
   - Confirmar chaves OpenAI em https://platform.openai.com/account/api-keys.
   - Verificar se as exchanges estão habilitadas para uso demo (serverTime, endpoints testnet).
   - Após ajustes, repetir os comandos acima.
2. **Fluxo end-to-end**
   - Com conexões validadas, execute também:
     ```bash
     /usr/bin/docker compose exec api python orchestrator/ai_orchestrator.py
     ```
   - Acompanhe métricas (`/metrics`) e auditoria (`backend/var/audit/trading.ndjson`).
3. **Registro e rotação**
   - Salve saídas relevantes no histórico e rotacione todas as chaves utilizadas após os testes.

## Portas e rede
- API HTTP: `127.0.0.1:8080`
- Prometheus: `127.0.0.1:9090`
- Grafana: `127.0.0.1:3000`
- Redis/Mongo/Qdrant: somente rede interna do Compose (autenticados)

Para depuração use `docker compose exec`; em produção mantenha as senhas/API keys fora do repositório.

## Observações adicionais
- Formulação da imagem compila TA-Lib C e instala `TA-Lib==0.4.26` com flags de compatibilidade contra GCC 13.
- `requirements.txt` já ajustado para `starlette>=0.36.3,<0.37.0` (compatível com FastAPI 0.110.0).
- Quando rodar `docker compose`, use sempre o path completo (`/usr/bin/docker compose ...`) para evitar conflitos com wrappers/aliases.
- Serviços internos agora exigem segredo: `redis` usa `REDIS_PASSWORD`, `mongo` usa `MONGO_ROOT_USERNAME`/`MONGO_ROOT_PASSWORD` e `qdrant` valida `QDRANT_API_KEY`.
- Prometheus (`127.0.0.1:9090`) e Grafana (`127.0.0.1:3000`) foram adicionados com provisionamento automático de dashboard “WOW Capital API Overview”.
- Métricas novas (`ai_orchestrator_*`) acompanham decisões/execuções automaticamente via Prometheus.
- LangChain/LangSmith não estão habilitados; a telemetria usa monitor próprio + Prometheus.
- Após testes com credenciais reais, rotacione chaves e remova dados sensíveis de logs (`docker logs`, `tests/` etc.).
