# AUTONOMA Architecture

## Overview

AUTONOMA is a self-hosted MLOps platform built around a closed-loop feedback cycle: collect data, train a model, serve predictions, observe performance, detect degradation, and recover — without manual intervention. The AI agent is the novel component that converts observations into actions, replacing the human on-call for routine recovery scenarios.

The system is designed to be run entirely on a single machine (Docker Compose) for development and demoing, and on a Kubernetes cluster for production.

---

## System Components

### Inference Service (`src/autonoma/api`)
FastAPI application serving the prediction endpoint, health checks, and Prometheus metrics. Stateless — scales horizontally. Depends on the model registry (MLflow) to load the active model version at startup.

### Feature Pipeline (`src/autonoma/pipelines`)
Scheduled process that fetches OHLCV candles from the Binance REST/WebSocket API, engineers features (returns, volatility, RSI, MACD), and writes them to PostgreSQL and Redis (for low-latency inference).

### Model Registry (MLflow)
Tracks experiment runs, stores model artifacts, and exposes a model registry with stage promotion (`Staging` → `Production`). The inference service queries the registry at startup to load the current `Production` model.

### Monitoring (`src/autonoma/monitoring`)
Two layers:
1. **Infrastructure metrics** — Prometheus scrapes the API's `/metrics` endpoint for request latency, throughput, and error rates.
2. **Model metrics** — A background job computes Population Stability Index (PSI) and KS-test scores against the training distribution to detect input drift; prediction distribution entropy flags output drift.

Grafana dashboards visualise both layers side-by-side.

### AI Agent (`src/autonoma/agent`)
A Groq-hosted Llama 3 model with a structured tool-use interface. The agent runs on a schedule (every N minutes) and also on alert webhook triggers. It receives a compressed system snapshot (recent metrics, active alerts, model registry state) as context, reasons about root cause, selects a recovery action from a defined action space, and executes it.

**Action space (Week 4+):**
- `retrigger_pipeline` — re-run the feature and retraining pipeline
- `swap_model_version` — promote a different model version to Production
- `scale_replicas` — adjust API replica count
- `open_incident` — log a structured incident record and notify

---

## Data Flow

```
Binance API
    │
    ▼
Feature Pipeline ──▶ PostgreSQL (features table)
                 ──▶ Redis (latest features, TTL 60s)
                          │
                          ▼
                   Inference Service
                   POST /predict ──▶ Response (prediction, confidence)
                          │
                          ▼
                   Prometheus ──▶ Grafana
                          │
                          ▼
                   Drift Monitor ──▶ Alert (if drift > threshold)
                                          │
                                          ▼
                                    AI Agent
                                    (observe → reason → act)
                                          │
                                          ▼
                                    Recovery Action
```

---

## Agent Decision Loop

```
1. OBSERVE   Read Prometheus metrics, drift scores, MLflow registry
2. REASON    LLM prompt: "Given this system state, what is wrong and what should I do?"
3. SELECT    Agent outputs a structured action from the allowed action space
4. EXECUTE   Python tool executes the action (API call, DB write, subprocess)
5. VERIFY    Poll relevant metric for 2 minutes; check if issue resolved
6. LOG       Write incident record to PostgreSQL with full reasoning trace
```

The reasoning trace is stored verbatim, making agent decisions auditable and debuggable.

---

## Tech Choices Rationale

- **FastAPI over Flask/Django** — native async, Pydantic v2 integration, zero-config OpenAPI docs. The auto-generated `/docs` UI is useful for demo purposes.
- **XGBoost over deep learning** — tabular crypto features are well-suited to gradient boosting; fast training enables frequent retraining without GPU.
- **Groq over OpenAI** — sub-100ms LLM inference allows the agent to act within the same request cycle as an alert. Free tier is sufficient for the demo's call volume.
- **Prometheus + Grafana over managed observability** — fully self-hosted, no data leaves the machine, integrates natively with the Docker Compose stack.
- **PostgreSQL over SQLite** — production-grade from day one; MLflow's backend store requires a real database for concurrent access.

---

## Future Considerations

- **Multi-asset support** — parameterise the pipeline for ETH, SOL, and other pairs.
- **Online learning** — replace batch retraining with incremental updates for faster adaptation.
- **Agent memory** — store past incidents and outcomes in a vector database so the agent learns from history.
- **Multi-agent coordination** — separate agents for data quality, model health, and infrastructure, coordinated by a supervisor.
- **Cost accounting** — track compute cost per prediction to enable cost-aware scaling decisions.
