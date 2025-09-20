# Servidor WOW Capital – Nova York (NY4/Nova Jérsei)

## Panorama do Servidor
- **Local**: Região metropolitana de Nova York (Equinix NY4/NYC área) – IP residente dos EUA.
- **IP público fixo**: `74.201.72.154`.
- **Objetivo**: Hospedar o stack WOW Capital (API, orquestrador, observabilidade) e operar testes/demo conectados a exchanges que aceitam clientes de NY.
- **Serviços ativos**: `api`, `redis`, `mongo` (auth habilitado), `qdrant`, `prometheus`, `grafana`, `ai-gateway`.
- **Ferramentas de monitoramento**: Prometheus (127.0.0.1:9090) e Grafana (127.0.0.1:3000) com dashboard “WOW Capital API Overview”.
- **Segurança**: Acesso restrito via SSH; `.env` com credenciais sensíveis (Exchange + OpenAI) armazenadas localmente.

## Conectividade e Bloqueios Atuais
- **Binance Spot Testnet**: bloqueio geográfico (HTTP 451). Aguardando liberação de IP pelo suporte.
- **Bybit Spot Testnet**: retornando HTTP 403 (CloudFront). Pedido de liberação pendente.
- **Kraken Futures Sandbox**: responde, mas as ordens retornam `['EAPI:Invalid key']`; precisa confirmar habilitação da nova chave demo.
- **Kraken VIP (vip.futures.kraken.com)**: conexão TLS estabelece, porém não há resposta HTTP – provável falta de whitelist.

## Check-list Operacional
1. Confirmar carregamento de variáveis no container:
   ```bash
   /usr/bin/docker compose -f deploy/docker-compose.yml exec api printenv | grep -E 'KRAKEN|BYBIT|BINANCE'
   ```
2. Testes rápidos de saúde:
   ```bash
   /usr/bin/curl -sSf http://127.0.0.1:8080/healthz
   /usr/bin/docker compose -f deploy/docker-compose.yml exec api python scripts/connection_tests.py
   ```
3. Observabilidade:
   - Prometheus: http://127.0.0.1:9090
   - Grafana: http://127.0.0.1:3000 (usuário/senha definidos em `.env`).
4. Execução de ordens demo (após liberação das exchanges): usar comandos cadastrados em `docs/TEST_PLAN.md` e registrar logs em `logs/test_runs/`.

## Exchanges Permitidas para Nova York

### Top 5 (spot/custódia regulados em NY)
| Exchange | Regulação | Produtos liberados | Observações de API |
|----------|-----------|--------------------|--------------------|
| Coinbase | BitLicense, SEC/FinCEN | Spot, custódia | API Advanced Trade (REST/WebSocket), ideal para trading automatizado |
| Gemini | Trust Charter (NYDFS) | Spot, custódia, GUSD | APIs REST/WS robustas |
| Paxos / itBit | Paxos Trust (NYDFS) | Spot via itBit, stablecoins reguladas | FIX/REST institucionais |
| Robinhood Crypto | BitLicense | Spot varejo | Sem API pública HFT (foco app) |
| PayPal Crypto | Licença NYDFS | Compra/venda + custódia limitada | Sem API aberta; integração via SDK |

### Brokers/Exchanges com presença em NY4 / baixa latência
| Segmento | Broker | Regulação | Infraestrutura | APIs |
|----------|--------|-----------|----------------|------|
| FX varejo | OANDA (US) | CFTC/NFA | Presença NY4 | REST v20, FIX |
| FX varejo | FOREX.com (StoneX) | CFTC/NFA | NY4 | REST + WebSocket, FIX |
| FX varejo | IG US | CFTC/NFA | NY4 | REST + streaming |
| Equities/ETFs | Interactive Brokers | SEC/FINRA, CFTC | Cross-connects em NY/NJ | TWS API, FIX, Client Portal |
| Equities/Options | TradeStation | SEC/FINRA | Datacenters EUA (latência baixa) | REST/WS |
| Equities/Options | Lightspeed Trading | SEC/FINRA | Linhas diretas NYSE/NASDAQ | FIX/SDK |
| Futuros institucionais | Advantage Futures | CFTC/NFA | CME Aurora + cross NY4 | FIX, iLink |
| Futuros institucionais | RJO’Brien | CFTC/NFA | NY/NJ, CME Aurora | FIX |
| Futuros institucionais | Wedbush Futures | CFTC/NFA | NY4 + CME | FIX |

> **Importante**: Binance, Bybit e Kraken global continuam bloqueados para IP norte-americano sem liberação explícita. Para operações regulares em NY, priorize exchanges/brokers licenciados acima.

## Procedimentos Recomendados
- **Whitelist de IP**: Requisitar liberação do IP `74.201.72.154` nos portais de suporte de Binance Testnet, Bybit Testnet e Kraken Futures.
- **Rotação de chaves**: Substituir API keys após cada bateria de testes, documentando-as em cofre seguro.
- **Documentação**: Atualizar resultados dos testes em `docs/OPERATIONS_REPORT.md` e marcar checklist em `docs/TEST_PLAN.md`.
- **Estratégias demo**: Ao liberar execução, usar plugins `jerico_*` ou estratégia controlada para validar ordens BUY/SELL e monitorar PnL.
- **Conformidade**: Reforçar que operações reais devem respeitar regulamentos estaduais (NYDFS) e políticas de cada broker.

## Próximos Passos
1. Confirmar liberação das exchanges bloqueadas e repetir testes de ordens demo.
2. Registrar métricas de execução no Prometheus/Grafana e anexar screenshots ao relatório operacional.
3. Avaliar adoção de exchanges reguladas (Coinbase/Gemini) para complementar o ambiente demo com execução permitida em NY.
4. Revisar hardening: reintroduzir usuário não-root no Dockerfile, validar regras de firewall e backups do `.env` em local seguro.

## Referência – Exchanges com IP brasileiro

| Exchange | Situação com IP BR | Observações |
|----------|--------------------|-------------|
| Binance | Spot permitido; derivativos (futuros/margem/opções) bloqueados desde 2023/2024 por acordo com a CVM. | API de derivativos recusa chamadas de IP brasileiro, inclusive em modo demo. |
| Bybit | Spot permitido; derivativos suspensos desde setembro/2022. | Endpoints de futuros retornam bloqueio (HTTP 403/451) para IP BR. |
| Kraken | Spot liberado; staking/margin/derivatives dependem de habilitação na conta. | IP brasileiro não é bloqueado; produtos restritos retornam erro apenas se desativados. |
| Exchanges locais (Mercado Bitcoin, Foxbit, BitPreço, Novadax) | Operação normal com IP BR; reguladas por CVM/BC. | Oferecem principalmente spot, eventualmente staking/tokenização simples. |

> Use esta matriz quando precisar deslocar workloads para servidores no Brasil: priorize brokers/exchanges licenciadas localmente ou que tenham liberado explicitamente o IP brasileiro para o produto desejado.
