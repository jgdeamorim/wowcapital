# Guia de Produção

## Segredos e variáveis de ambiente
- `.env` em `/opt/wowcapital` deve conter apenas credenciais reais; nunca versionar.
- Defina obrigatoriamente `REDIS_PASSWORD`, `MONGO_ROOT_USERNAME`, `MONGO_ROOT_PASSWORD`, `QDRANT_API_KEY`, `OPENAI_API_KEY` e chaves das exchanges.
- Preferir Docker Secrets ou `docker compose --env-file` para ambientes multi-equipe.
- Rotacione chaves de exchanges/OpenAI sempre após testes e monitore limites de requisição.

## Endurecimento de rede
- Remova mapeamentos externos de Redis/Mongo/Qdrant no `docker-compose.yml` quando for expor apenas a API (o template atual já restringe o acesso e deixa a API mapeada apenas em `127.0.0.1:8080`).
- Ative autenticação:
  - **Redis**: habilite `requirepass` (ou ACL) e restrinja `bind` a `127.0.0.1`/rede interna.
  - **MongoDB**: iniciar com `--auth`, criar usuário admin e definir roles específicas para a aplicação.
  - **Qdrant**: defina `QDRANT__SERVICE__API_KEY` e considere TLS reverso via Nginx/Traefik.
- Configure firewall (UFW/iptables) permitindo apenas 22 e 8080 conforme necessário.

## Observabilidade e logs
- API: `docker compose logs -f api`
- Redis/Mongo/Qdrant: `docker compose logs -f <service>`
- Métricas: `http://localhost:8080/metrics`
- Auditoria: arquivo `backend/var/audit/trading.ndjson` dentro do container (`docker compose exec api tail -f backend/var/audit/trading.ndjson`).

## Backups
- Volumes mapeados: `mongo_data`, `qdrant_data`.
  - Mongo: use `mongodump`/`mongorestore` via `docker compose exec mongo ...`.
  - Qdrant: snapshot direto do volume (`docker run --rm -v deploy_qdrant_data:/data busybox tar czf - /data`).
- Preserve `.env`, manifests em `plugins/` e arquivos de configuração em `config/` em um cofre seguro.

## Testes com credenciais reais
1. **Sandbox/Demo das exchanges**
   - Preencha `.env` com chaves demo (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET=true`, etc.).
   - Rode os testes em `tests/integration/exchanges/` conforme instruções em `OPERATIONS_REPORT.md`.
   - Monitore logs para capturar erros de autenticação ou throttle.
   - Consulte `SERVER_NY_OVERVIEW.md` para restrições geográficas (NY/Brasil) e priorize exchanges liberadas para o IP atual.
2. **OpenAI / Orquestração completa**
   - Configure `OPENAI_API_KEY`, `OPENAI_ORGANIZATION`.
   - Execute `tests/test_openai_real.py` e `scripts/demo_trading_validator.py` dentro do container API.
   - Observe consumo e custos adicionais.

## Segurança operacional
- Após execuções reais, limpe logs que possam conter payloads sensíveis (`docker compose logs`, arquivos em `backend/var/audit`).
- Aplique atualizações de sistema: `sudo apt update && sudo apt upgrade` (manter Docker/Compose atualizados).
- Considere usar reverse proxy (Nginx) com TLS para expor a API publicamente, delegando autenticação (OAuth/API key).
