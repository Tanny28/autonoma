
from autonoma.core.taxonomy import Action
from autonoma.remediation.rate_limiter import RateLimiter


class FakeClock:
    """A controllable clock for testing cooldowns."""

    def __init__(self):
        self.current_time = 100.0

    def __call__(self):
        return self.current_time

    def advance(self, seconds):
        self.current_time += seconds


def test_first_model_changing_action_is_allowed():
    limiter = RateLimiter(cooldown_seconds=300)

    assert limiter.is_allowed(Action.RETRAIN)


def test_successful_action_starts_cooldown():
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=300, clock=clock)

    limiter.record_success(Action.RETRAIN)

    assert not limiter.is_allowed(Action.RETRAIN)


def test_action_allowed_after_cooldown_expires():
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=300, clock=clock)

    limiter.record_success(Action.RETRAIN)
    clock.advance(300)

    assert limiter.is_allowed(Action.RETRAIN)


def test_failed_or_unrecorded_action_does_not_start_cooldown():
    limiter = RateLimiter(cooldown_seconds=300)

    # Checking permission alone must not start the cooldown.
    assert limiter.is_allowed(Action.RETRAIN)
    assert limiter.is_allowed(Action.RETRAIN)


def test_non_model_changing_actions_are_not_rate_limited():
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=300, clock=clock)

    limiter.record_success(Action.RETRAIN)

    assert limiter.is_allowed(Action.SUPPRESS)
    assert limiter.is_allowed(Action.ESCALATE)


def test_rollback_and_recalibration_are_rate_limited_too():
    clock = FakeClock()
    limiter = RateLimiter(cooldown_seconds=300, clock=clock)

    limiter.record_success(Action.RETRAIN)

    assert not limiter.is_allowed(Action.ROLLBACK)
    assert not limiter.is_allowed(Action.RECALIBRATE)


def test_negative_cooldown_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        RateLimiter(cooldown_seconds=-1)
