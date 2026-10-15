import math

import pytest

from autonoma.core.bundle import GroundTruth
from autonoma.core.taxonomy import Cause
from autonoma.injection import injector as inj
from autonoma.injection.injector import InjectionResult


def _records(values, label=0):
    return [
        {"features": {"x": float(v), "y": float(v) * 10}, "record_index": i, "label": label}
        for i, v in enumerate(values)
    ]


def test_injection_result_stores_cause_and_onset():
    result = InjectionResult(records=[], cause=Cause.SUDDEN_COVARIATE, onset_index=10)

    assert result.cause is Cause.SUDDEN_COVARIATE
    assert result.onset_index == 10


def test_sudden_covariate_shifts_from_onset_by_severity_times_pre_onset_std():
    # Pre-onset values [1,2,3,4]: mean 2.5, population std sqrt(1.25).
    # Post-onset values are huge, so a whole-dataset std would give a very different shift.
    records = _records([1, 2, 3, 4, 1000, 1000])
    expected_shift = 2 * math.sqrt(1.25)

    result = inj.sudden_covariate(records, feature="x", onset_index=4, severity=2)

    assert result.cause is Cause.SUDDEN_COVARIATE
    assert result.onset_index == 4
    assert result.records[3]["features"]["x"] == 4.0
    assert result.records[4]["features"]["x"] == pytest.approx(1000 + expected_shift)
    assert result.records[5]["features"]["x"] == pytest.approx(1000 + expected_shift)


def test_injectors_do_not_mutate_input():
    records = _records([1, 2, 3, 4, 5, 6])
    snapshot = [{**r, "features": dict(r["features"])} for r in records]

    inj.sudden_covariate(records, "x", onset_index=3, severity=1)
    inj.concept_drift(records, onset_index=2, severity=3, seed=1)

    assert records == snapshot


def test_sudden_covariate_rejects_bad_severity():
    with pytest.raises(ValueError):
        inj.sudden_covariate(_records([1, 2, 3, 4]), "x", onset_index=2, severity=4)


def test_sudden_covariate_rejects_constant_pre_onset_feature():
    with pytest.raises(ValueError):
        inj.sudden_covariate(_records([5, 5, 5, 9]), "x", onset_index=3, severity=1)


def test_gradual_covariate_ramps_up_to_the_sudden_shift():
    records = _records([1, 2, 3, 4] + [4.0] * 300)
    sudden = inj.sudden_covariate(records, "x", onset_index=4, severity=1)
    gradual = inj.gradual_covariate(records, "x", onset_index=4, severity=1)

    shifts = [
        g["features"]["x"] - r["features"]["x"]
        for g, r in zip(gradual.records[4:], records[4:], strict=True)
    ]

    assert gradual.cause is Cause.GRADUAL_COVARIATE
    assert shifts[0] < shifts[10] < shifts[-1]  # grows over time
    assert shifts[-1] == pytest.approx(
        sudden.records[-1]["features"]["x"] - 4.0
    )  # same final shift


def test_concept_drift_flips_labels_only_after_onset_and_keeps_features():
    records = _records([1, 2, 3, 4] * 25, label=0)

    result = inj.concept_drift(records, onset_index=50, severity=3, seed=7)

    assert result.cause is Cause.CONCEPT
    assert all(r["label"] == 0 for r in result.records[:50])
    assert any(r["label"] == 1 for r in result.records[50:])
    assert [r["features"] for r in result.records] == [r["features"] for r in records]


def test_same_seed_gives_same_output():
    records = _records([1, 2, 3, 4] * 25)

    first = inj.concept_drift(records, onset_index=10, severity=2, seed=42)
    second = inj.concept_drift(records, onset_index=10, severity=2, seed=42)

    assert first.records == second.records


def test_pipeline_fault_null_replaces_values_after_onset():
    records = _records([1, 2, 3, 4] * 25)

    result = inj.pipeline_fault(records, "x", onset_index=20, severity=3, seed=0, kind="null")

    assert result.cause is Cause.PIPELINE_FAULT
    assert all(r["features"]["x"] is not None for r in result.records[:20])
    assert any(r["features"]["x"] is None for r in result.records[20:])


def test_pipeline_fault_scale_multiplies_values():
    records = _records([1, 2, 3, 4, 5])

    result = inj.pipeline_fault(records, "x", onset_index=3, severity=1, kind="scale")

    assert [r["features"]["x"] for r in result.records] == [1.0, 2.0, 3.0, 400.0, 500.0]


def test_pipeline_fault_freeze_holds_value_at_onset():
    records = _records([1, 2, 3, 4, 5, 6])

    result = inj.pipeline_fault(records, "x", onset_index=2, severity=1, kind="freeze")

    assert [r["features"]["x"] for r in result.records] == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]


def test_pipeline_fault_swap_exchanges_two_columns():
    records = _records([1, 2, 3])

    result = inj.pipeline_fault(records, "x", onset_index=1, severity=1, kind="swap", swap_with="y")

    assert result.records[0]["features"]["x"] == 1.0
    assert result.records[1]["features"]["x"] == 20.0
    assert result.records[1]["features"]["y"] == 2.0


def test_pipeline_fault_swap_requires_swap_with():
    with pytest.raises(ValueError):
        inj.pipeline_fault(_records([1, 2, 3]), "x", onset_index=1, severity=1, kind="swap")


def test_seasonal_wave_returns_to_normal_after_one_period():
    records = _records([1, 2, 3, 4] + [4.0] * 200)

    result = inj.seasonal(records, "x", onset_index=4, severity=2)

    assert result.cause is Cause.SEASONAL
    peak = result.records[4 + inj.SEASON_PERIOD // 4]["features"]["x"] - 4.0
    assert peak > 0
    assert result.records[4 + inj.SEASON_PERIOD]["features"]["x"] == pytest.approx(4.0)
    assert all(r["features"]["x"] == 4.0 for r in result.records[4 + inj.SEASON_PERIOD :])


def test_ground_truth_answer_key_uses_shared_class():
    result = inj.seasonal(_records([1, 2, 3, 4] + [4.0] * 200), "x", onset_index=4, severity=2)

    truth = inj.ground_truth(result, episode_id="ep-1", severity=2, seed=0)

    assert isinstance(truth, GroundTruth)
    assert truth == GroundTruth(
        episode_id="ep-1", cause=Cause.SEASONAL, onset_index=4, severity=2, seed=0
    )
