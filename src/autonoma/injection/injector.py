"""Degradation injectors for the O1 replay layer.

Each function takes a clean stream of records, applies ONE of the five
degradation classes from `core/taxonomy.py` from `onset_index` onwards, and
returns an `InjectionResult`. Inputs are never mutated.

- `severity` (1, 2 or 3) scales how big the change is.
- `seed` makes every random choice repeatable: same inputs + same seed = same output.
- Scale-based injectors measure the feature's std on the records BEFORE the onset
  only, so the injected shift is relative to normal behaviour, not to the fault.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from autonoma.core.bundle import GroundTruth
from autonoma.core.taxonomy import Cause

#: Severity -> shift size in pre-onset standard deviations (sudden, gradual, seasonal).
SHIFT_BY_SEVERITY = {1: 1.0, 2: 2.0, 3: 3.0}
#: Severity -> records taken to ramp up to the full shift (gradual covariate drift).
RAMP_BY_SEVERITY = {1: 200, 2: 100, 3: 50}
#: Severity -> fraction of post-onset labels flipped (concept drift).
FLIP_RATE_BY_SEVERITY = {1: 0.2, 2: 0.4, 3: 0.6}
#: Severity -> fraction of post-onset values replaced by null (pipeline fault, kind "null").
NULL_RATE_BY_SEVERITY = {1: 0.1, 2: 0.3, 3: 0.5}
#: Length of one seasonal wave, in records. The wave returns to normal after one period.
SEASON_PERIOD = 100
#: The pipeline faults `pipeline_fault` can inject.
PIPELINE_FAULTS = ("null", "scale", "freeze", "swap")


@dataclass
class InjectionResult:
    records: list[dict[str, Any]]
    cause: Cause
    onset_index: int


def sudden_covariate(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    severity: int,
    seed: int = 0,
) -> InjectionResult:
    """Shift `feature` by a fixed amount from `onset_index` onwards."""
    displacement = _shift_size(records, feature, onset_index, severity)
    return InjectionResult(
        records=_apply_shift(records, feature, onset_index, lambda _index: displacement),
        cause=Cause.SUDDEN_COVARIATE,
        onset_index=onset_index,
    )


def gradual_covariate(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    severity: int,
    seed: int = 0,
) -> InjectionResult:
    """Same final shift as `sudden_covariate`, but ramped up linearly over N records."""
    full_shift = _shift_size(records, feature, onset_index, severity)
    ramp = RAMP_BY_SEVERITY[_check_severity(severity)]

    def shift(index: int) -> float:
        return full_shift * min(1.0, (index - onset_index + 1) / ramp)

    return InjectionResult(
        records=_apply_shift(records, feature, onset_index, shift),
        cause=Cause.GRADUAL_COVARIATE,
        onset_index=onset_index,
    )


def concept_drift(
    records: list[dict[str, Any]],
    onset_index: int,
    severity: int,
    seed: int = 0,
    label_key: str = "label",
) -> InjectionResult:
    """Flip the label on a random share of post-onset records. Features are untouched.

    Assumes a binary 0/1 label.
    """
    rate = FLIP_RATE_BY_SEVERITY[_check_severity(severity)]
    rng = random.Random(seed)
    modified = _copy_records(records)

    for index, record in enumerate(modified):
        if index >= onset_index and rng.random() < rate:
            record[label_key] = 1 - record[label_key]

    return InjectionResult(
        records=modified,
        cause=Cause.CONCEPT,
        onset_index=onset_index,
    )


def pipeline_fault(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    severity: int,
    seed: int = 0,
    *,
    kind: str | None = None,
    swap_with: str | None = None,
) -> InjectionResult:
    """Break the data pipeline from `onset_index` onwards.

    `kind` is one of PIPELINE_FAULTS. If omitted, the seed picks one.
    - "null":   replace some values with None (rate depends on severity)
    - "scale":  multiply values by 10**(severity+1), e.g. a units bug
    - "freeze": hold the feature at its value at onset (a stuck constant)
    - "swap":   swap `feature` with `swap_with` (a column-order bug)
    """
    sev = _check_severity(severity)
    rng = random.Random(seed)

    if kind is None:
        choices = [k for k in PIPELINE_FAULTS if k != "swap" or swap_with is not None]
        kind = rng.choice(choices)
    if kind not in PIPELINE_FAULTS:
        raise ValueError(f"Unknown pipeline fault: {kind}")
    if kind == "swap" and swap_with is None:
        raise ValueError("kind='swap' needs swap_with=<other feature name>")

    modified = _copy_records(records)
    after_onset = [record for index, record in enumerate(modified) if index >= onset_index]

    if kind == "null":
        rate = NULL_RATE_BY_SEVERITY[sev]
        for record in after_onset:
            if rng.random() < rate:
                record["features"][feature] = None

    elif kind == "scale":
        factor = 10 ** (sev + 1)
        for record in after_onset:
            value = record["features"].get(feature)
            if isinstance(value, (int, float)):
                record["features"][feature] = value * factor

    elif kind == "freeze":
        stuck_value = modified[onset_index]["features"].get(feature)
        for record in after_onset:
            record["features"][feature] = stuck_value

    else:  # swap
        for record in after_onset:
            features = record["features"]
            features[feature], features[swap_with] = features.get(swap_with), features.get(feature)

    return InjectionResult(
        records=modified,
        cause=Cause.PIPELINE_FAULT,
        onset_index=onset_index,
    )


def seasonal(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    severity: int,
    seed: int = 0,
) -> InjectionResult:
    """Add a sine wave for one period from onset, then return to normal.

    Deterministic, so `seed` is accepted only to keep the signature uniform.
    """
    amplitude = _shift_size(records, feature, onset_index, severity)

    def shift(index: int) -> float:
        elapsed = index - onset_index
        if elapsed >= SEASON_PERIOD:
            return 0.0
        return amplitude * math.sin(2 * math.pi * elapsed / SEASON_PERIOD)

    return InjectionResult(
        records=_apply_shift(records, feature, onset_index, shift),
        cause=Cause.SEASONAL,
        onset_index=onset_index,
    )


def ground_truth(
    result: InjectionResult,
    episode_id: str,
    severity: int,
    seed: int,
) -> GroundTruth:
    """Build the answer key for an injected episode, using the shared GroundTruth class."""
    return GroundTruth(
        episode_id=episode_id,
        cause=result.cause,
        onset_index=result.onset_index,
        severity=severity,
        seed=seed,
    )


def _check_severity(severity: int) -> int:
    if severity not in SHIFT_BY_SEVERITY:
        raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
    return severity


def _copy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy each record and its features dict so the caller's input is never changed."""
    return [{**record, "features": dict(record["features"])} for record in records]


def _pre_onset_std(records: list[dict[str, Any]], feature: str, onset_index: int) -> float:
    values = [
        record["features"][feature]
        for record in records[:onset_index]
        if isinstance(record["features"].get(feature), (int, float))
    ]
    if len(values) < 2:
        raise ValueError(f"Need at least 2 numeric pre-onset values for feature: {feature}")

    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std == 0:
        raise ValueError(f"Feature is constant before onset, cannot scale a shift: {feature}")
    return std


def _shift_size(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    severity: int,
) -> float:
    std = _pre_onset_std(records, feature, onset_index)
    return SHIFT_BY_SEVERITY[_check_severity(severity)] * std


def _apply_shift(
    records: list[dict[str, Any]],
    feature: str,
    onset_index: int,
    shift: Callable[[int], float],
) -> list[dict[str, Any]]:
    """Add shift(index) to each numeric value of `feature` at or after onset_index."""
    modified = _copy_records(records)

    for index, record in enumerate(modified):
        value = record["features"].get(feature)
        if index >= onset_index and isinstance(value, (int, float)):
            record["features"][feature] = value + shift(index)

    return modified
