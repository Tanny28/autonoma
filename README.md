# ⚡ AUTONOMA

**A self-healing MLOps platform with an AI agent that monitors, optimizes, and recovers your ML systems autonomously.**

[![CI](https://github.com/tanmayshinde/autonoma/actions/workflows/ci.yml/badge.svg)](https://github.com/tanmayshinde/autonoma/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

---

## Why AUTONOMA?

Production ML systems break in subtle ways: data drift degrades accuracy silently, infrastructure spikes cause latency blowups, and models stale out while engineers sleep. Traditional MLOps tooling surfaces these problems — but leaves the response to humans.

AUTONOMA closes the loop. An embedded AI agent watches your metrics in real time, interprets anomalies, and executes targeted recovery actions — retrigger pipelines, swap model versions, scale resources — without waiting for a pagerduty alert at 3 AM. The showcase use case is live BTC/USD price direction prediction, where the agent must adapt to a market that never stops changing.

---

## Features

| Feature | Status |
|---|---|
| FastAPI inference service with `/predict` and `/health` endpoints | ✅ Shipped |
| Structured JSON logging via Loguru | ✅ Shipped |
| Prometheus metrics middleware + `/metrics` endpoint | ✅ Shipped |
| Pydantic v2 settings with `.env` support | ✅ Shipped |
| Full Docker Compose stack (Postgres, Redis, MLflow, Prometheus, Grafana) | ✅ Shipped |
| CI pipeline (lint → test → docker build) | ✅ Shipped |
| Live BTC feature pipeline (Binance public API) | 📋 Planned (Week 2) |
| Trained XGBoost direction classifier | 📋 Planned (Week 2) |
| MLflow experiment tracking + model registry | 📋 Planned (Week 2) |
| Drift detection (PSI, KS-test) | 📋 Planned (Week 3) |
| Grafana dashboards for model + infra metrics | 📋 Planned (Week 3) |
| Autonomous AI agent (Groq LLM) with tool use | 📋 Planned (Week 4) |
| Automated retraining trigger on drift detection | 📋 Planned (Week 5) |
| Kubernetes manifests + Helm chart | 📋 Planned (Week 6) |
| Chaos engineering suite | 📋 Planned (Week 7) |
| Full demo video + blog post | 📋 Planned (Week 8) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      AUTONOMA Stack                      │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Binance  │───▶│ Feature  │───▶│  XGBoost Model   │  │
│  │  API     │    │ Pipeline │    │  (MLflow reg.)   │  │
│  └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                           │             │
│  ┌──────────────────────────────────────▼──────────┐   │
│  │              FastAPI Inference Service           │   │
│  │         /predict  /health  /metrics              │   │
│  └──────────────────────────────────────────────────┘   │
│          │                    │                          │
│  ┌───────▼──────┐    ┌───────▼──────┐                  │
│  │  Prometheus  │    │   Postgres   │                   │
│  │  (metrics)   │    │  + Redis     │                   │
│  └───────┬──────┘    └──────────────┘                   │
│          │                                               │
│  ┌───────▼──────┐    ┌──────────────────────────────┐  │
│  │   Grafana    │    │       AI Agent (Groq LLM)    │  │
│  │ (dashboards) │    │  observe → reason → act      │  │
│  └──────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart

**Prerequisites:** Docker Desktop, Python 3.11+, Git

1. **Clone the repo**
   ```bash
   git clone https://github.com/tanmayshinde/autonoma.git
   cd autonoma
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for local dev)
   ```

3. **Install Python dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

4. **Run tests to verify the install**
   ```bash
   pytest -v
   ```

5. **Spin up the full stack**
   ```bash
   docker compose up -d
   ```
   Services:
   - API → http://localhost:8000/docs
   - MLflow → http://localhost:5000
   - Prometheus → http://localhost:9090
   - Grafana → http://localhost:3000 (admin / admin)

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Inference API | FastAPI + Uvicorn | Async, fast, auto-docs |
| ML framework | XGBoost + scikit-learn | Tabular prediction, fast inference |
| Experiment tracking | MLflow | Industry standard, self-hostable |
| Metrics | Prometheus + Grafana | Pull-based, mature ecosystem |
| Agent LLM | Groq (Llama 3) | Ultra-low latency inference, free tier |
| Database | PostgreSQL 16 | MLflow backend, app state |
| Cache / queue | Redis 7 | Feature cache, task queue |
| Containerisation | Docker + Compose | Reproducible local stack |
| Orchestration | Kubernetes + Helm | Production deploy (Week 6) |
| CI | GitHub Actions | Lint → test → build on every push |
| Language | Python 3.11 | Ecosystem, type hints |

---

## Roadmap

| Week | Theme | Deliverables |
|---|---|---|
| 1 | **Foundation** | Project scaffold, FastAPI stub, full Docker stack, CI |
| 2 | **Data + Model** | Binance pipeline, XGBoost classifier, MLflow tracking |
| 3 | **Observability** | Drift detection, Grafana dashboards, alerting |
| 4 | **AI Agent v1** | Groq-powered agent that interprets metrics and recommends actions |
| 5 | **Self-healing** | Automated retraining trigger, model swap, incident logs |
| 6 | **Kubernetes** | Helm chart, HPA, resource limits, production hardening |
| 7 | **Chaos + Resilience** | Chaos engineering, graceful degradation, SLA targets |
| 8 | **Showcase** | End-to-end demo, blog post, LinkedIn launch |

---

## Showcase Use Case

AUTONOMA's demo uses **live BTC/USD price data** from the Binance public WebSocket and REST API (no API key required) to train and serve a binary direction classifier: will the price be higher in the next 15 minutes?

> ⚠️ **Disclaimer:** This is a **platform demonstration only**. The model output is not financial advice and should not be used to make trading decisions. Cryptocurrency markets are highly volatile and unpredictable. AUTONOMA's purpose is to demonstrate autonomous MLOps capabilities, not to provide alpha.

---

## Contributing

AUTONOMA is currently a solo portfolio project built in public. Issues and discussions are welcome. PRs will be considered after Week 4 once the core architecture stabilises.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Tanmay Shinde](https://github.com/tanmayshinde)*
