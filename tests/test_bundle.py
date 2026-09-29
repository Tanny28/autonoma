import pytest
from pydantic import ValidationError

from autonoma.core.bundle import (
    DistributionalSignals,
    GroundTruth,
    IntegritySignals,
    LabelledBundle,
    PerformanceSignals,
    SignalBundle,
    read_jsonl,
    write_jsonl,
)
from autonoma.core.taxonomy import Cause


def make_bundle(episode_id: str = "elec2-pipeline-s2-seed1", **overrides) -> SignalBundle:
    fields = {
        "dataset": "elec2",
        "episode_id": episode_id,
        "window_start": 1000,
        "window_end": 1499,
        "distributional": DistributionalSignals(
            ks_statistic={"nswprice": 0.31},
            ks_pvalue={"nswprice": 0.001},
            psi={"nswprice": 0.27},
            adwin_drift=True,
        ),
        "performance": PerformanceSignals(
            labels_available=400, accuracy=0.71, f1=0.68, accuracy_delta=-0.12, f1_delta=-0.1
        ),
        "integrity": IntegritySignals(
            null_rate={"nswdemand": 0.35},
            stale_run_length={"vicprice": 480},
            schema_order_changed=False,
        ),
    }
    fields.update(overrides)
    return SignalBundle(**fields)


def make_truth(episode_id: str = "elec2-pipeline-s2-seed1") -> GroundTruth:
    return GroundTruth(
        episode_id=episode_id, cause=Cause.PIPELINE_FAULT, onset_index=1100, severity=2, seed=1
    )


def test_arms_cannot_see_the_true_cause():
    assert "cause" not in SignalBundle.model_fields
    assert "truth" not in SignalBundle.model_fields


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_bundle(true_cause="seasonal_variation")


def test_bundles_are_immutable():
    bundle = make_bundle()
    with pytest.raises(ValidationError):
        bundle.window_end = 2000


def test_performance_metrics_require_labels():
    PerformanceSignals(labels_available=0)
    with pytest.raises(ValidationError):
        PerformanceSignals(labels_available=0, accuracy=0.9)


def test_window_must_be_ordered():
    with pytest.raises(ValidationError):
        make_bundle(window_start=500, window_end=100)


def test_bundle_and_truth_must_match_episode():
    with pytest.raises(ValidationError):
        LabelledBundle(bundle=make_bundle("a"), truth=make_truth("b"))


def test_scenario_set_round_trips_through_jsonl(tmp_path):
    rows = [
        LabelledBundle(bundle=make_bundle(f"ep{i}"), truth=make_truth(f"ep{i}")) for i in range(3)
    ]
    path = tmp_path / "scenarios" / "elec2.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
