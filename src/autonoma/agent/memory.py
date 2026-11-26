"""Incident memory for arm 6 (#28, hypothesis H2).

After an episode closes, the agent stores what it saw, what it did and how
that turned out. On a new window it recalls the most similar past incidents
and gets them as context. Recall compares a short numeric signature of the
signal bundle, so it works for any dataset.

Any SQLAlchemy URL works: Postgres in the stack (`settings.database_url`),
SQLite in tests and offline experiments.
"""

import math

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
)

from autonoma.agent.arms import Decision
from autonoma.core.bundle import SignalBundle
from autonoma.core.taxonomy import CORRECT_ACTION, Cause, is_harmful

_metadata = MetaData()
incidents = Table(
    "incident_memory",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("dataset", String, nullable=False),
    Column("signature", JSON, nullable=False),
    Column("diagnosed_cause", String),
    Column("action", String),
    Column("confirmed_cause", String, nullable=False),
    Column("outcome", String, nullable=False),
)


def signature(bundle: SignalBundle) -> dict[str, float]:
    """The handful of numbers that tell the five causes apart."""
    d, p, i = bundle.distributional, bundle.performance, bundle.integrity
    window_len = bundle.window_end - bundle.window_start + 1
    return {
        "max_psi": max(d.psi.values(), default=0.0),
        "min_ks_pvalue": min(d.ks_pvalue.values(), default=1.0),
        "accuracy_delta": p.accuracy_delta or 0.0,
        "max_null_rate": max(i.null_rate.values(), default=0.0),
        "max_range_violation": max(i.range_violation_rate.values(), default=0.0),
        "max_stale_fraction": max(i.stale_run_length.values(), default=0) / window_len,
        "min_cardinality_ratio": min(i.cardinality_ratio.values(), default=1.0),
        "schema_order_changed": float(i.schema_order_changed),
    }


def outcome(confirmed: Cause, decision: Decision) -> str:
    if decision.action is CORRECT_ACTION[confirmed]:
        return "correct"
    if decision.action is not None and is_harmful(confirmed, decision.action):
        return "harmful"
    return "wrong"


class IncidentMemory:
    def __init__(self, url: str) -> None:
        self._engine = create_engine(url)
        _metadata.create_all(self._engine)

    def record(self, bundle: SignalBundle, decision: Decision, confirmed: Cause) -> None:
        """Store a closed incident once its true cause is known."""
        with self._engine.begin() as conn:
            conn.execute(
                insert(incidents).values(
                    dataset=bundle.dataset,
                    signature=signature(bundle),
                    diagnosed_cause=decision.cause.value if decision.cause else None,
                    action=decision.action.value if decision.action else None,
                    confirmed_cause=confirmed.value,
                    outcome=outcome(confirmed, decision),
                )
            )

    def recall(self, bundle: SignalBundle, k: int = 3) -> list[dict]:
        """The k past incidents on this dataset whose signals look most like `bundle`."""
        # ponytail: full scan per call, fine for thousands of incidents; pgvector past that.
        now = signature(bundle)
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(incidents).where(incidents.c.dataset == bundle.dataset)
            ).mappings()
            scored = [
                (math.dist(list(now.values()), _vector(r["signature"], now)), r) for r in rows
            ]
        scored.sort(key=lambda s: s[0])
        return [
            {
                "signals": row["signature"],
                "diagnosed_cause": row["diagnosed_cause"],
                "action": row["action"],
                "confirmed_cause": row["confirmed_cause"],
                "outcome": row["outcome"],
            }
            for _, row in scored[:k]
        ]


def _vector(stored: dict, like: dict) -> list[float]:
    return [stored.get(key, 0.0) for key in like]
