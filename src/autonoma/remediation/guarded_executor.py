
"""Guard the executor with remediation rate limiting (#29)."""

from autonoma.agent.arms import Decision
from autonoma.remediation.executor import Executor, Outcome, Status
from autonoma.remediation.rate_limiter import RateLimiter


class GuardedExecutor:
    """Apply cooldown checks before executing remediation decisions."""

    def __init__(
        self,
        executor: Executor,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.executor = executor
        self.rate_limiter = rate_limiter or RateLimiter()

    def execute(self, decision: Decision) -> Outcome:
        action = decision.action

        # Block model-changing actions during the cooldown.
        if action is not None and not self.rate_limiter.is_allowed(action):
            return Outcome(
                action=action,
                status=Status.RATE_LIMITED,
                detail=(
                    f"{action} blocked: remediation cooldown "
                    f"of {self.rate_limiter.cooldown_seconds} seconds "
                    "has not expired"
                ),
                justification=decision.justification,
            )

        # Execute the decision using the existing executor.
        outcome = self.executor.execute(decision)

        # Start cooldown only after successful execution.
        if outcome.status is Status.APPLIED and action is not None:
            self.rate_limiter.record_success(action)

        return outcome
