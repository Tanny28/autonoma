import pytest

from autonoma.agent.arms import Decision, always_retrain
from autonoma.agent.rules import RuleArm
from autonoma.core.bundle import IntegritySignals, LabelledBundle
from autonoma.core.taxonomy import CORRECT_ACTION, Action, Cause
from autonoma.evaluation.metrics import (
    ablate,
    action_cost,
    brier_score,
    detection_latency,
    evaluate,
    faithfulness,
    mttr,
    recurrence_improvement,
    reliability_bins,
    run_arm,
)
from tests.test_rule_arm import EPISODES, episode


def scenario_set() -> list[LabelledBundle]:
    return [row for cause, w in EPISODES.items() for row in episode(cause.value, cause, w)]


def null_rate_arm(bundle):
    """Rolls back on nulls and says so: a faithful explanation."""
    if bundle.integrity.null_rate.get("nswdemand", 0) > 0.05:
        return Decision(
            drift_detected=True,
            action=Action.ROLLBACK,
            cause=Cause.PIPELINE_FAULT,
            confidence=0.8,
            decisive_signal="integrity.null_rate",
        )
    return Decision(drift_detected=False, action=None)


def stubborn_arm(bundle):
    """Always rolls back but claims nulls drove it: an unfaithful explanation."""
    return Decision(
        drift_detected=True,
        action=Action.ROLLBACK,
        cause=Cause.PIPELINE_FAULT,
        confidence=1.0,
        decisive_signal="integrity.null_rate",
    )


def test_every_arm_gets_every_metric():
    report = evaluate({"always": lambda: always_retrain, "rules": RuleArm}, scenario_set())
    assert set(report) == {"always", "rules"}
    assert report["always"]["harmful_action_rate"] == 1.0
    assert report["always"]["diagnostic_accuracy"] is None  # never diagnoses
    assert report["rules"]["diagnostic_accuracy"] == 1.0
    assert report["rules"]["cost_weighted_error"] == 0.0
    assert report["rules"]["faithfulness"] is None  # rules name no decisive signal


def test_cost_ranks_harm_above_a_miss_above_escalation():
    assert action_cost(Cause.SEASONAL, Action.SUPPRESS) == 0
    harmful = action_cost(Cause.SEASONAL, Action.RETRAIN)
    missed = action_cost(Cause.CONCEPT, None)
    escalated = action_cost(Cause.CONCEPT, Action.ESCALATE)
    assert harmful > missed > escalated > 0


def test_latency_and_mttr_count_records_from_onset():
    results = run_arm(RuleArm(), scenario_set())
    concept = next(r for r in results if r.truth.cause is Cause.CONCEPT)
    assert concept.first_detect_window == 199  # window 1 ends at record 199, onset is 100
    assert detection_latency([concept]) == mttr([concept]) == 99


def test_brier_and_reliability_from_confidence():
    results = run_arm(stubborn_arm, scenario_set())
    # Confidence 1.0 everywhere, right on 1 of 5 causes.
    assert brier_score(results) == pytest.approx(4 / 5)
    assert reliability_bins(results) == [
        {"bin": "0.8-1.0", "mean_confidence": 1.0, "accuracy": 0.2, "count": 5}
    ]


def test_faithfulness_separates_honest_from_stubborn_explanations():
    rows = scenario_set()
    honest = faithfulness(lambda: null_rate_arm, rows, run_arm(null_rate_arm, rows))
    stubborn = faithfulness(lambda: stubborn_arm, rows, run_arm(stubborn_arm, rows))
    assert honest == 1.0
    assert stubborn == 0.0


def test_ablate_blanks_only_the_named_signal():
    b = scenario_set()[-1].bundle.model_copy(
        update={"integrity": IntegritySignals(null_rate={"x": 0.5}, stale_run_length={"x": 9})}
    )
    out = ablate(b, "integrity.null_rate")
    assert out.integrity.null_rate == {} and out.integrity.stale_run_length == {"x": 9}


class LearningArm:
    """Escalates until told the cause of an episode, then handles that cause correctly.

    Peeks at the episode id, which the test names after the cause; real arms cannot.
    """

    def __init__(self):
        self.known = set()

    def __call__(self, bundle):
        cause = Cause(bundle.episode_id.split("#")[0])
        action = CORRECT_ACTION[cause] if cause in self.known else Action.ESCALATE
        return Decision(drift_detected=True, action=action)

    def learn(self, bundle, decision, confirmed):
        self.known.add(confirmed)


def test_learn_hook_feeds_recurrence_improvement():
    rows = [
        row
        for n in (1, 2)
        for cause in (Cause.CONCEPT, Cause.SEASONAL)
        for row in episode(f"{cause.value}#{n}", cause, [{}])
    ]
    arm = LearningArm()
    results = run_arm(arm, rows)
    assert arm.known == {Cause.CONCEPT, Cause.SEASONAL}
    assert recurrence_improvement(results) == 1.0  # 0% right first time, 100% on repeats
