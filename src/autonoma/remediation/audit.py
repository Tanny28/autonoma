
"""Decision audit logging for remediation (#29)."""

import sqlalchemy as sa

from autonoma.agent.arms import Decision
from autonoma.core.config import settings
from autonoma.remediation.executor import Outcome


metadata = sa.MetaData()

decision_log = sa.Table(
    "decision_log",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column(
        "timestamp",
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    ),
    sa.Column("episode", sa.String, nullable=False),
    sa.Column("diagnosed_cause", sa.String, nullable=True),
    sa.Column("confidence", sa.Float, nullable=True),
    sa.Column("action", sa.String, nullable=True),
    sa.Column("justification", sa.Text, nullable=False),
    sa.Column("outcome", sa.Text, nullable=False),
)


class AuditLogger:
    """Store remediation decisions and their execution outcomes."""

    def __init__(self, database_url: str | None = None) -> None:
        self.engine = sa.create_engine(database_url or settings.database_url)
        metadata.create_all(self.engine)

    def record(
        self,
        episode_id: str,
        decision: Decision,
        outcome: Outcome,
    ) -> None:
        """Save one decision and its outcome to the audit log."""

        values = {
            "episode": episode_id,
            "diagnosed_cause": (
                decision.cause.value if decision.cause is not None else None
            ),
            "confidence": decision.confidence,
            "action": (
                decision.action.value if decision.action is not None else None
            ),
            "justification": decision.justification,
            "outcome": f"{outcome.status.value}: {outcome.detail}",
        }

        with self.engine.begin() as connection:
            connection.execute(sa.insert(decision_log).values(**values))
