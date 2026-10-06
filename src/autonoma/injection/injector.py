from dataclasses import dataclass
from typing import Any

from autonoma.core.taxonomy import Cause

@dataclass
class InjectionResult:
    records: list[dict[str, Any]]
    cause: Cause
    onset_index: int

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

def sudden_covariate(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    k: float,
) -> InjectionResult:
    values = [
        record["features"][feature]
        for record in records
        if isinstance(record["features"].get(feature), (int, float))
    ]

    if not values:
        raise ValueError(f"No numeric values found for feature: {feature}")

    mean = sum(values) / len(values)

    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5

    displacement = k * std

    modified_records = []

    for index, record in enumerate(records):
        new_record = {
            **record,
            "features": dict(record["features"]),
        }

        if index >= onset_index:
            value = new_record["features"].get(feature)

            if isinstance(value, (int, float)):
                new_record["features"][feature] = value + displacement

        modified_records.append(new_record)

    return InjectionResult(
        records=modified_records,
        cause=Cause.SUDDEN_COVARIATE,
        onset_index=onset_index,
    )