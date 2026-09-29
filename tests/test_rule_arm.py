import pytest

from autonoma.agent.rules import RuleArm, integrity_faults
from autonoma.core.bundle import (
    DistributionalSignals,
    GroundTruth,
    IntegritySignals,
    LabelledBundle,
    PerformanceSignals,
    SignalBundle,
)
from autonoma.core.taxonomy import CORRECT_ACTION, Action, Cause
from autonoma.evaluation.metrics import correct_action_rate, harmful_action_rate, run_arm


def bundle(
    episode: str,
    i: int,
    psi: float = 0.02,
    acc_delta: float = 0.0,
    integrity: IntegritySignals | None = None,
) -> SignalBundle:
    return SignalBundle(
        dataset="elec2",
        episode_id=episode,
        window_start=i * 100,
        window_end=i * 100 + 99,
        distributional=DistributionalSignals(psi={"nswprice": psi}),
        performance=PerformanceSignals(
            labels_available=100, accuracy=0.8 + acc_delta, accuracy_delta=acc_delta
        ),
        integrity=integrity or IntegritySignals(),
    )


def episode(name: str, cause: Cause, windows: list[dict]) -> list[LabelledBundle]:
    truth = GroundTruth(episode_id=name, cause=cause, onset_index=100, severity=2, seed=0)
    return [LabelledBundle(bundle=bundle(name, i, **w), truth=truth) for i, w in enumerate(windows)]


QUIET = {}
EPISODES = {
    Cause.SUDDEN_COVARIATE: [QUIET, QUIET, {"psi": 0.6, "acc_delta": -0.1}],
    Cause.GRADUAL_COVARIATE: [
        QUIET,
        {"psi": 0.15},
        {"psi": 0.25},
        {"psi": 0.35, "acc_delta": -0.08},
    ],
    Cause.CONCEPT: [QUIET, {"acc_delta": -0.15}],
    Cause.PIPELINE_FAULT: [
        QUIET,
        {
            "psi": 0.4,
            "acc_delta": -0.1,
            "integrity": IntegritySignals(null_rate={"nswdemand": 0.3}),
        },
    ],
    Cause.SEASONAL: [QUIET, {"psi": 0.3}, {"psi": 0.5}, {"psi": 0.3}, QUIET],
}


def first_decision(cause: Cause):
    arm = RuleArm()
    for row in episode(cause.value, cause, EPISODES[cause]):
        decision = arm(row.bundle)
        if decision.action is not None:
            return decision
    return None


@pytest.mark.parametrize("cause", list(Cause))
def test_rules_diagnose_each_class(cause):
    decision = first_decision(cause)
    assert decision is not None
    assert decision.cause is cause
    assert decision.action is CORRECT_ACTION[cause]
    assert decision.justification


def test_rules_score_perfectly_on_clean_synthetic_set():
    rows = [row for cause, w in EPISODES.items() for row in episode(cause.value, cause, w)]
    results = run_arm(RuleArm(), rows)
    assert harmful_action_rate(results) == 0.0
    assert correct_action_rate(results) == 1.0


def test_quiet_window_takes_no_action():
    decision = RuleArm()(bundle("e", 0))
    assert not decision.drift_detected
    assert decision.action is None


def test_benign_drift_is_watched_before_acting():
    arm = RuleArm()
    decisions = [arm(bundle("e", i, psi=p)) for i, p in enumerate([0.02, 0.5, 0.5, 0.5])]
    assert [d.action for d in decisions] == [None, None, None, Action.RETRAIN]
    assert decisions[-1].cause is Cause.SUDDEN_COVARIATE


def test_history_is_kept_per_episode():
    arm = RuleArm()
    arm(bundle("a", 0, psi=0.5))
    decision = arm(bundle("b", 0, psi=0.3))
    assert decision.action is None


@pytest.mark.parametrize(
    "integrity",
    [
        IntegritySignals(null_rate={"x": 0.2}),
        IntegritySignals(range_violation_rate={"x": 0.5}),
        IntegritySignals(stale_run_length={"x": 90}),
        IntegritySignals(cardinality_ratio={"x": 0.05}),
        IntegritySignals(schema_order_changed=True),
    ],
)
def test_each_integrity_rule_fires(integrity):
    assert integrity_faults(bundle("e", 0, integrity=integrity))


def test_healthy_integrity_has_no_faults():
    healthy = IntegritySignals(
        null_rate={"x": 0.01},
        range_violation_rate={"x": 0.02},
        stale_run_length={"x": 3},
        cardinality_ratio={"x": 0.9},
    )
    assert integrity_faults(bundle("e", 0, integrity=healthy)) == []
