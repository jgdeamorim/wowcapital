# 📦 Backup e Restauração da Inteligência Orquestradora

> **Objetivo:**  
> Garantir que toda a **inteligência acumulada** do sistema (dados + embeddings + contexto + orquestração + monitoramento) possa ser salva e restaurada em outro servidor, sem perda de informações críticas.

---

## 🔹 1. Qdrant (Banco Vetorial)

- **Salvar**:
  - Snapshot binário (`qdrant snapshot create`).  
  - Exportação JSON/Parquet via API (`/collections/{name}/points`).  

- **Arquivos**:
  - `snapshot.snapshot`
  - `export.json`

---

## 🔹 2. MongoDB (Documentos Originais + Logs)

- **Salvar**:
  - `mongodump` → arquivos BSON completos.  
  - `mongoexport` → JSON para versionamento.  

- **Arquivos**:
  - `dump.bson`
  - `export.json`

---

## 🔹 3. Redis (Cache / Memória Efêmera)

- **Salvar**:
  - `dump.rdb` (snapshotting).  
  - `appendonly.aof` (histórico de comandos).  

- **Arquivos**:
  - `dump.rdb`
  - `appendonly.aof`

---

## 🔹 4. LangChain (Orquestração)

- **Salvar**:
  - Prompts templates (`.yaml` ou `.json`).  
  - Chains / Agents configs.  

- **Arquivos**:
  - `prompts.yaml`
  - `chains.json`
  - `agents.yaml`

---

## 🔹 5. LangSmith (Observabilidade e Debug)

- **Salvar**:
  - Logs de execuções via API (`runs.json`).  
  - Configurações de projetos (`projects.json`).  

- **Arquivos**:
  - `runs.json`
  - `projects.json`

---

## 🔹 6. OpenAI GPT-4.1-mini (ou outro LLM)

- **Se local**:
  - Pesos do modelo (se licenciado).  
  - Configurações do servidor (`model_config.json`).  
- **Se API**:
  - Prompts system usados.  
  - Parâmetros de inferência.  

- **Arquivos**:
  - `model_config.json`
  - `system_prompts.json`

---

## 🔹 7. Infraestrutura (Glue)

- **Salvar**:
  - Scripts de ingestão (ETL, sincronização).  
  - Configs de deploy (Docker, K8s).  
  - Variáveis de ambiente (`.env`).  

- **Arquivos**:
  - `docker-compose.yaml`
  - `.env`
  - `etl_scripts.tar.gz`

---

## 🔹 8. Prometheus (Métricas e Alertas)

- **O que salvar**:
  - **Banco de dados TSDB** (armazenamento de séries temporais).  
  - Configurações de scrape (`prometheus.yml`).  
  - Regras de alertas (`rules.yml`).  

- **Arquivos**:
  - `data/` (diretório TSDB do Prometheus).  
  - `prometheus.yml`  
  - `rules.yml`

---

## 🔹 9. Grafana (Dashboards e Alertas Visuais)

- **O que salvar**:
  - Configurações do Grafana (`grafana.ini`).  
  - Banco de dados do Grafana (se SQLite: `grafana.db`; se MySQL/Postgres: dump do DB).  
  - Dashboards exportados (`.json`).  
  - Alertas configurados.  

- **Arquivos**:
  - `grafana.ini`
  - `grafana.db` (ou dump SQL se DB externo).  
  - `dashboards/*.json`

---

## 🔹 Estrutura Recomendada no MinIO

```bash
s3://inteligencia-backups/
   /2025-09-22/
       qdrant/
           snapshot.snapshot
           export.json
       mongodb/
           dump.bson
           export.json
       redis/
           dump.rdb
           appendonly.aof
       langchain/
           prompts.yaml
           chains.json
           agents.yaml
       langsmith/
           runs.json
           projects.json
       gpt/
           model_config.json
           system_prompts.json
       infra/
           docker-compose.yaml
           .env
           etl_scripts.tar.gz
       prometheus/
           prometheus.yml
           rules.yml
           tsdb/                # diretório com dados das métricas
       grafana/
           grafana.ini
           grafana.db
           dashboards/*.json
🔹 Processo de Backup
Pausar ingestão.

Criar snapshot do Qdrant.

Executar mongodump no MongoDB.

Exportar Redis (dump.rdb e appendonly.aof).

Exportar configs do LangChain/LangSmith.

Copiar dados/configs do Prometheus.

Exportar dashboards e configs do Grafana.

Compactar e enviar para MinIO (mc cp ou boto3).

🔹 Processo de Restauração
Subir instâncias limpas (Qdrant, MongoDB, Redis, Prometheus, Grafana).

Importar snapshots/dumps.

Restaurar configs do LangChain/LangSmith.

Reaplicar system prompts e configs do modelo GPT.

Recarregar dashboards do Grafana.

Reiniciar ingestão → sistema volta no mesmo estado.

