# AUTONOMA Architecture

Five layers, described in order of dependency: each layer down the list
consumes the output of the ones above it, so build order follows this list.
See the [README](../README.md) for the project's research framing and the
[synopsis](figs2/) for the reviewed diagram set.

## 1. Replay & injection (`src/autonoma/injection`) — O1

Replays a public dataset (Elec2, Airlines, UCI Credit) through the serving
layer as a live stream — one record per HTTP request, in temporal order. The
first 30% of each dataset trains the baseline model and fixes the reference
distribution used by the signal layer; the remainder is replayed sequentially.

At an experimenter-chosen index, the injector applies one of the five
degradation classes defined in
[`core/taxonomy.py`](../src/autonoma/core/taxonomy.py) and records the
ground-truth `(cause, onset_index)` pair. That ground truth is what makes
every downstream metric — diagnostic accuracy, harmful action rate, detection
latency — computable at all.

## 2. Serving (`src/autonoma/serving`)

FastAPI application exposing `/predict`, `/health`, `/health/ready`, and
`/metrics`. Logs every request (features, prediction, confidence) and ingests
true labels after a configurable delay, mirroring the label lag of a real
production system. Stateless; the model itself is loaded from the MLflow
registry rather than baked into the image, so a rollback is a registry
pointer change, not a redeploy.

## 3. Signal extraction (`src/autonoma/signals`) — O1

Three independent families, all computed from the request log:

- **Distributional** — KS-test, Population Stability Index, chi-square for
  categoricals, ADWIN for the streaming case.
- **Performance** — rolling accuracy / F1 delta, available once delayed
  labels arrive.
- **Integrity** — null rate, cardinality change, range/unit violation,
  staleness, schema-order change. This family has no equivalent in prior
  work; it is what lets `upstream_pipeline_fault` be told apart from genuine
  drift, since a pipeline fault can leave the *distributional* signals nearly
  unchanged while corrupting the data outright.

Output is a fixed-shape signal bundle, consumed identically by every
evaluation arm — the ablation in layer 4 depends on that identical input.

## 4. Diagnostic agent (`src/autonoma/agent`) — O2

Given a signal bundle plus relevant Incident Memory, produces a cause
classification (one of the five), a confidence score, a selected action, and
a natural-language justification.

Three arms live in this layer on the same interface, since the ablation is
the point:

- **Threshold rules** — a hand-written deterministic policy (arm 3).
- **Trained classifier** — a gradient-boosted model trained on labelled
  scenarios (arm 4).
- **LLM agent** — LangGraph orchestration over a Groq-hosted Llama model
  (arm 5), with an incident-memory-augmented variant (arm 6) that reads and
  writes a PostgreSQL store of past `(cause, action, outcome)` triples.

## 5. Remediation (`src/autonoma/remediation`) — O3

Executes the selected action. Guardrails are implemented before the agent
that depends on them, not after:

- Closed, enumerated action space (`retrain` / `rollback` / `recalibrate` /
  `suppress` / `escalate`) — the agent cannot invent new actions.
- Validation gate: a retrained candidate must beat the incumbent on held-out
  data before it replaces it in the MLflow registry.
- Every action is reversible and logged with the agent's stated reasoning.
- Rate limiting against oscillation between model versions.
- Human approval configurable for high-impact actions.

## Cross-cutting: observability & audit (`src/autonoma/observability`)

Prometheus scrapes `/metrics`; Grafana renders model and infrastructure
dashboards side by side. Every autonomous action is written to a PostgreSQL
decision log together with the full reasoning trace, making agent behaviour
auditable independent of the live agent process.

## Data flow

```
dataset ──▶ injector ──▶ serving (/predict) ──▶ request log
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                             distributional      performance        integrity
                                    │                 │                 │
                                    └────────── signal bundle ─────────┘
                                                      │
                                        ┌─────────────┼─────────────┐
                                        ▼             ▼             ▼
                                   rules (3)   classifier (4)   LLM agent (5/6)
                                        │             │             │
                                        └───── selected action ─────┘
                                                      │
                                                      ▼
                                          remediation executor (O3)
                                                      │
                                                      ▼
                                     decision log ──▶ Grafana / Prometheus
```
