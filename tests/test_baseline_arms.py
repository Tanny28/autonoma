from autonoma.agent.arms import alert_only, always_retrain, drift_detected
from autonoma.core.bundle import (
    DistributionalSignals,
    GroundTruth,
    IntegritySignals,
    LabelledBundle,
    PerformanceSignals,
    SignalBundle,
)
from autonoma.core.taxonomy import Action, Cause
from autonoma.evaluation.metrics import (
    correct_action_rate,
    detection_rate,
    harmful_action_rate,
    run_arm,
)


def window(episode: str, start: int, drifting: bool) -> SignalBundle:
    return SignalBundle(
        dataset="elec2",
        episode_id=episode,
        window_start=start,
        window_end=start + 99,
        distributional=DistributionalSignals(psi={"nswprice": 0.35 if drifting else 0.02}),
        performance=PerformanceSignals(labels_available=0),
        integrity=IntegritySignals(),
    )


def episode(name: str, cause: Cause, drift_from: int, n: int = 4) -> list[LabelledBundle]:
    truth = GroundTruth(
        episode_id=name, cause=cause, onset_index=drift_from * 100, severity=2, seed=0
    )
    return [
        LabelledBundle(bundle=window(name, i * 100, drifting=i >= drift_from), truth=truth)
        for i in range(n)
    ]


def scenario_set() -> list[LabelledBundle]:
    return [
        *episode("sudden", Cause.SUDDEN_COVARIATE, drift_from=2),
        *episode("concept", Cause.CONCEPT, drift_from=1),
        *episode("pipeline", Cause.PIPELINE_FAULT, drift_from=2),
        *episode("seasonal", Cause.SEASONAL, drift_from=1),
    ]


def test_quiet_window_is_not_drift():
    assert not drift_detected(window("e", 0, drifting=False))
    assert drift_detected(window("e", 0, drifting=True))


def test_performance_drop_alone_counts_as_drift():
    bundle = window("e", 0, drifting=False).model_copy(
        update={
            "performance": PerformanceSignals(
                labels_available=100, accuracy=0.7, accuracy_delta=-0.2
            )
        }
    )
    assert drift_detected(bundle)


def test_alert_only_never_acts():
    decision = alert_only(window("e", 0, drifting=True))
    assert decision.drift_detected
    assert decision.action is None


def test_always_retrain_retrains_on_drift_only():
    assert always_retrain(window("e", 0, drifting=True)).action is Action.RETRAIN
    assert always_retrain(window("e", 0, drifting=False)).action is None


def test_first_action_ends_the_episode():
    results = run_arm(always_retrain, episode("sudden", Cause.SUDDEN_COVARIATE, drift_from=2))
    assert len(results) == 1
    assert results[0].first_action_window == 299


def test_always_retrain_is_harmful_on_every_no_retrain_episode():
    results = run_arm(always_retrain, scenario_set())
    assert harmful_action_rate(results) == 1.0
    assert correct_action_rate(results) == 0.5


def test_alert_only_is_never_harmful_but_never_correct():
    results = run_arm(alert_only, scenario_set())
    assert harmful_action_rate(results) == 0.0
    assert correct_action_rate(results) == 0.0
    assert detection_rate(results) == 1.0


def test_harmful_rate_undefined_without_no_retrain_episodes():
    rows = episode("sudden", Cause.SUDDEN_COVARIATE, drift_from=1)
    assert harmful_action_rate(run_arm(always_retrain, rows)) is None
