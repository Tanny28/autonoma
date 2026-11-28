
"""Guard the executor with remediation rate limiting (#29)."""

from autonoma.agent.arms import Decision
from autonoma.remediation.audit import AuditLogger
from autonoma.remediation.executor import Executor, Outcome, Status
from autonoma.remediation.rate_limiter import RateLimiter


class GuardedExecutor:
    """Apply cooldown checks and audit logging to remediation decisions."""

    def __init__(
        self,
        executor: Executor,
        rate_limiter: RateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.executor = executor
        self.rate_limiter = rate_limiter or RateLimiter()
        self.audit_logger = audit_logger

    def execute(
        self,
        decision: Decision,
        episode_id: str | None = None,
    ) -> Outcome:
        # Require an episode ID whenever audit logging is enabled.
        if self.audit_logger is not None and not episode_id:
            raise ValueError(
                "episode_id is required when audit logging is enabled"
            )

        action = decision.action

        # Block model-changing actions during the cooldown.
        if action is not None and not self.rate_limiter.is_allowed(action):
            outcome = Outcome(
                action=action,
                status=Status.RATE_LIMITED,
                detail=(
                    f"{action} blocked: remediation cooldown "
                    f"of {self.rate_limiter.cooldown_seconds} seconds "
                    "has not expired"
                ),
                justification=decision.justification,
            )
        else:
            # Execute the decision using the existing executor.
            outcome = self.executor.execute(decision)

            # Start cooldown only after successful execution.
            if outcome.status is Status.APPLIED and action is not None:
                self.rate_limiter.record_success(action)

        # Record both successful and unsuccessful outcomes.
        if self.audit_logger is not None:
            self.audit_logger.record(
                episode_id=episode_id,
                decision=decision,
                outcome=outcome,
            )

        return outcome
