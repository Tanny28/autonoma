# Contributing to AUTONOMA

## Setup

```bash
cp .env.example .env        # fill in GROQ_API_KEY only if you work on the agent
python -m venv .venv
make install
make test
```

`make up` starts the full Docker stack (Postgres, Redis, MLflow, Prometheus, Grafana, serving API).

## Who owns what

| Layer | Folder | Owner |
|---|---|---|
| Diagnostic agent, remediation (O2, O3) | `src/autonoma/agent/`, `src/autonoma/remediation/` | Tanmay Shinde |
| Model training and serving | `src/autonoma/serving/` | Dushyant Dhote |
| Drift detection and signals (O1) | `src/autonoma/signals/`, `src/autonoma/injection/` | Pranav Shimpi |
| Observability and infrastructure | `src/autonoma/observability/`, `infra/`, `docker-compose.yml` | Namrata Shinde |

Shared code lives in `src/autonoma/core/`. Changes there affect every layer, so they need a review from Tanmay.

## Workflow

1. Pick an issue assigned to you, or ask for one.
2. Branch from `main` using `feature/<short-name>`, for example `feature/injection-framework`.
3. Keep commits small and messages descriptive.
4. Open a PR into `main` and link its issue with `Closes #N`.
5. CI (lint and tests) must pass, and one reviewer must approve. `main` is protected; nobody pushes to it directly.

## Rules that matter for the research

- **Use the taxonomy.** Degradation classes and remediation actions come from `core/taxonomy.py`. Never write `"concept_drift"` or `"retrain"` as a raw string.
- **The action space is closed.** Do not add a sixth remediation action without discussing it with the team first. The closed action space is one of the project's stated guardrails.
- **Every arm sees the same signals.** Rule-based, classifier and LLM arms all consume the identical signal bundle. If one arm gets extra information, the comparison is invalid.
- **Keep the baselines.** The rule and classifier arms are the evidence for the research question. They are never optional, even under deadline pressure.

## What never goes in the repo

- `.env`, API keys, tokens or passwords. `.env` is gitignored; keep it that way.
- Raw datasets. Commit the download script under `data/`, not the data.
- Large generated files: model artifacts, MLflow runs, notebook outputs with big tables.
- Personal data of any kind.

If you commit a secret by accident, tell the team immediately and rotate the key. Deleting the commit is not enough, because the key has already been exposed.

## Code style

`make lint` runs ruff (line length 100). `make format` fixes most issues automatically.
