from autonoma.core.taxonomy import Cause
from autonoma.injection.injector import InjectionResult


def test_injection_result_stores_cause_and_onset():
    result = InjectionResult(
        records=[],
        cause=Cause.SUDDEN_COVARIATE,
        onset_index=10,
    )

    assert result.cause is Cause.SUDDEN_COVARIATE
    assert result.onset_index == 10

def test_sudden_covariate_changes_numeric_feature():
    from autonoma.injection.injector import sudden_covariate

    records = [
        {"features": {"x": 1.0}, "record_index": 0},
        {"features": {"x": 2.0}, "record_index": 1},
        {"features": {"x": 3.0}, "record_index": 2},
    ]

    result = sudden_covariate(
        records,
        feature="x",
        onset_index=1,
        k=1.0,
    )

    assert result.cause is Cause.SUDDEN_COVARIATE
    assert result.onset_index == 1
    assert result.records[0]["features"]["x"] == 1.0
    assert result.records[1]["features"]["x"] != 2.0
    assert result.records[2]["features"]["x"] != 3.0