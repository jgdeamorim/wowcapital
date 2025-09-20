# Deploy com Docker Compose

## Pré-requisitos
- Docker CE + plugin Compose já instalados (verificado com `/usr/bin/docker --version` e `/usr/bin/docker compose version`).
- Repositório em `/opt/wowcapital` e `.env` preenchido (não versionar) com `REDIS_PASSWORD`, `MONGO_ROOT_USERNAME`/`MONGO_ROOT_PASSWORD`, `QDRANT_API_KEY`, chaves OpenAI e exchanges.

## Build e subida
```bash
cd /opt/wowcapital/deploy
/usr/bin/docker compose -f docker-compose.yml build
/usr/bin/docker compose -f docker-compose.yml up -d
```
*(A CLI emite um aviso sobre `version:` obsoleto; pode ser ignorado ou remover a chave do YAML futuramente).* 

### Monitoramento
```bash
/usr/bin/docker compose -f docker-compose.yml ps
/usr/bin/docker compose -f docker-compose.yml logs -f api
```

### Encerramento / limpeza
```bash
/usr/bin/docker compose -f docker-compose.yml down      # mantém volumes
/usr/bin/docker compose -f docker-compose.yml down -v   # remove volumes qdrant/mongo
```

## Testes rápidos pós-deploy
```bash
/usr/bin/docker compose -f docker-compose.yml exec api python scripts/connection_tests.py
/usr/bin/docker compose -f docker-compose.yml exec qdrant curl -sSf http://localhost:6333/healthz
/usr/bin/docker compose -f docker-compose.yml exec redis redis-cli -a "$REDIS_PASSWORD" PING
/usr/bin/docker compose -f docker-compose.yml exec mongo mongosh --username "$MONGO_ROOT_USERNAME" --password "$MONGO_ROOT_PASSWORD" --quiet --eval 'db.runCommand({ ping: 1 })'
```
Resultados esperados (2025-09-20): `PING: True`, `{ ok: 1 }`, `healthz check passed`, `/docs status: 200`.

## Portas expostas (modo atual)
- API HTTP: `127.0.0.1:8080`
- Prometheus: `127.0.0.1:9090`
- Grafana: `127.0.0.1:3000`
- Redis/MongoDB/Qdrant: somente rede interna do Compose

> Para depuração local use `docker compose exec`. Em produção, mantenha senhas e API keys fora do repositório (ver `PRODUCTION_GUIDE.md`).

## Documentação complementar
- `OPERATIONS_REPORT.md`: status da stack, outputs de testes e passos para validar exchanges/OpenAI.
- `CONNECTION_TESTS.md`: comandos detalhados caso queira repetir ou automatizar as verificações.
