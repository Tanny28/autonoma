
"""Rate limiting for model-changing remediation actions (#29)."""

from collections.abc import Callable
from time import monotonic

from autonoma.core.taxonomy import Action
from autonoma.core.config import settings

MODEL_CHANGING_ACTIONS = frozenset(
    {
        Action.RETRAIN,
        Action.RECALIBRATE,
        Action.ROLLBACK,
    }
)


class RateLimiter:
    """Prevent repeated model-changing actions within a cooldown period."""

       def __init__(
        self,
        cooldown_seconds: float | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if cooldown_seconds is None:
            cooldown_seconds = settings.remediation_cooldown_seconds

        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")

        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._last_success_at: float | None = None

    def is_allowed(self, action: Action) -> bool:
        """Check whether an action is allowed at the current time."""

        if action not in MODEL_CHANGING_ACTIONS:
            return True

        if self._last_success_at is None:
            return True

        elapsed = self.clock() - self._last_success_at
        return elapsed >= self.cooldown_seconds

    def record_success(self, action: Action) -> None:
        """Start the cooldown only after a successful model-changing action."""

        if action in MODEL_CHANGING_ACTIONS:
            self._last_success_at = self.clock()
