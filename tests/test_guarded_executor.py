
from sqlalchemy import select

from autonoma.agent.arms import Decision
from autonoma.core.taxonomy import Action
from autonoma.remediation.audit import AuditLogger, decision_log
from autonoma.remediation.executor import Executor, Status
from autonoma.remediation.guarded_executor import GuardedExecutor
from autonoma.remediation.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self):
        self.current_time = 100.0

    def __call__(self):
        return self.current_time

    def advance(self, seconds):
        self.current_time += seconds


class FakeRegistry:
    def __init__(self):
        self.current = "model-v1"
        self.previous = None

    def champion(self):
        return self.current

    def promote(self, model, justification):
        self.previous = self.current
        self.current = model
        return "2"

    def rollback(self, justification):
        if self.previous is not None:
            self.current, self.previous = self.previous, self.current
        return "1"


def make_decision(action):
    return Decision(
        drift_detected=True,
        action=action,
        justification="Test decision",
    )


def make_guarded_executor(
    clock,
    candidate="better",
    audit_logger=None,
    **executor_options,
):
    scores = {
        "model-v1": 0.80,
        "better": 0.90,
        "better2": 0.95,
        "worse": 0.70,
    }

    executor = Executor(
        registry=FakeRegistry(),
        retrain=lambda: candidate,
        recalibrate=lambda champion: candidate,
        score=scores.__getitem__,
        **executor_options,
    )

    limiter = RateLimiter(
        cooldown_seconds=300,
        clock=clock,
    )

    return GuardedExecutor(executor, limiter, audit_logger)


def test_successful_action_starts_cooldown():
    clock = FakeClock()
    guarded = make_guarded_executor(clock)

    first = guarded.execute(make_decision(Action.RETRAIN))
    second = guarded.execute(make_decision(Action.RETRAIN))

    assert first.status is Status.APPLIED
    assert second.status is Status.RATE_LIMITED


def test_action_allowed_after_cooldown():
    clock = FakeClock()
    guarded = make_guarded_executor(clock)

    first = guarded.execute(make_decision(Action.RETRAIN))
    clock.advance(300)

    guarded.executor.retrain = lambda: "better2"
    second = guarded.execute(make_decision(Action.RETRAIN))

    assert first.status is Status.APPLIED
    assert second.status is Status.APPLIED


def test_rejected_candidate_does_not_start_cooldown():
    clock = FakeClock()
    guarded = make_guarded_executor(clock, candidate="worse")

    first = guarded.execute(make_decision(Action.RETRAIN))
    second = guarded.execute(make_decision(Action.RETRAIN))

    assert first.status is Status.REJECTED
    assert second.status is Status.REJECTED


def test_suppress_is_not_blocked_by_cooldown():
    clock = FakeClock()
    guarded = make_guarded_executor(clock)

    guarded.execute(make_decision(Action.RETRAIN))
    result = guarded.execute(make_decision(Action.SUPPRESS))

    assert result.status is Status.APPLIED


def test_approval_pending_does_not_start_cooldown():
    clock = FakeClock()
    guarded = make_guarded_executor(
        clock,
        needs_approval=frozenset({Action.RETRAIN}),
        approve=lambda decision: False,
    )

    first = guarded.execute(make_decision(Action.RETRAIN))
    second = guarded.execute(make_decision(Action.RETRAIN))

    assert first.status is Status.AWAITING_APPROVAL
    assert second.status is Status.AWAITING_APPROVAL


def test_audit_logger_records_success_and_rate_limited_outcomes(tmp_path):
    clock = FakeClock()
    audit_logger = AuditLogger(
        database_url=f"sqlite:///{tmp_path / 'audit.db'}"
    )
    guarded = make_guarded_executor(clock, audit_logger=audit_logger)

    first = guarded.execute(
        make_decision(Action.RETRAIN),
        episode_id="episode-001",
    )
    second = guarded.execute(
        make_decision(Action.RETRAIN),
        episode_id="episode-001",
    )

    assert first.status is Status.APPLIED
    assert second.status is Status.RATE_LIMITED

    with audit_logger.engine.connect() as connection:
        records = connection.execute(
            select(decision_log).order_by(decision_log.c.id)
        ).mappings().all()

    assert len(records) == 2
    assert records[0]["episode"] == "episode-001"
    assert records[0]["outcome"].startswith("applied:")
    assert records[1]["outcome"].startswith("rate_limited:")
    assert records[0]["justification"] == "Test decision"
