# Testes de Conexão

## Verificações rápidas (sem sair do host)
```bash
/usr/bin/curl -sSf http://127.0.0.1:8080/docs | head -n 1    # espera <!DOCTYPE html>
```
*(Redis/Mongo/Qdrant não expõem portas públicas; utilize `docker compose exec`.)*

## Via Docker Compose
```bash
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec redis redis-cli -a "$REDIS_PASSWORD" PING
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec mongo mongosh --username "$MONGO_ROOT_USERNAME" --password "$MONGO_ROOT_PASSWORD" --quiet --eval "db.runCommand({ ping: 1 })"
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec qdrant curl -sSf http://localhost:6333/healthz
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec api python scripts/connection_tests.py
```
Saída esperada do script (2025-09-20):
```
== Redis ==          PING: True
== Mongo ==          PING: {'ok': 1.0}
== Qdrant ==         HEALTH: healthz check passed
== API ==            /docs status: 200
```

## Endpoints úteis
- `/healthz` → retorna `{ "status": "ok" }`
- `/docs` / `/openapi.json` → documentação Swagger/OpenAPI
- `/metrics` → métricas Prometheus (disponível via `curl http://localhost:8080/metrics`)
- Prometheus UI: `http://127.0.0.1:9090`
- Grafana UI: `http://127.0.0.1:3000`

## Testes avançados
Para validar exchanges e integrações reais (requer `.env` com chaves):
```bash
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec api python tests/test_binance_adapter.py
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec api python tests/test_kraken_adapter.py
/usr/bin/docker compose -f /opt/wowcapital/deploy/docker-compose.yml exec api python tests/test_openai_real.py
```
Documente os resultados e rotacione credenciais após cada rodada.
