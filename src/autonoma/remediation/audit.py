
"""Decision audit logging for remediation (#29)."""

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
)

from autonoma.agent.arms import Decision
from autonoma.core.config import settings
from autonoma.remediation.executor import Outcome


metadata = MetaData()

decision_log = Table(
    "decision_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now()),
    Column("episode", String, nullable=False),
    Column("diagnosed_cause", String, nullable=True),
    Column("confidence", Float, nullable=True),
    Column("action", String, nullable=True),
    Column("justification", Text, nullable=False),
    Column("outcome", Text, nullable=False),
)


class AuditLogger:
    """Store remediation decisions and their execution outcomes."""

    def __init__(self, database_url: str | None = None) -> None:
        self.engine = create_engine(database_url or settings.database_url)
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
            connection.execute(insert(decision_log).values(**values))
