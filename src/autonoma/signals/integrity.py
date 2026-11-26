"""Integrity signals: is the data itself broken?

These checks look at the raw values in a monitoring window compared with a
reference window (the training window). They are what separates a pipeline
fault from genuine drift: a pipeline fault can leave the distribution nearly
unchanged while corrupting the data outright.

Each record is a dict with a "features" mapping, as produced by the replay
layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autonoma.core.bundle import IntegritySignals


def _values(records: Sequence[Mapping[str, Any]], feature: str) -> list[Any]:
    return [record["features"].get(feature) for record in records]


def _null_rate(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v is None) / len(values)


def _cardinality_ratio(reference: Sequence[Any], current: Sequence[Any]) -> float:
    """Distinct non-null values in the window divided by those in the reference."""
    ref_distinct = len({v for v in reference if v is not None})
    if ref_distinct == 0:
        return 0.0
    cur_distinct = len({v for v in current if v is not None})
    return cur_distinct / ref_distinct


def _range_violation_rate(reference: Sequence[Any], current: Sequence[Any]) -> float:
    """Fraction of current numeric values outside the reference min-max."""
    ref_nums = [v for v in reference if isinstance(v, (int, float))]
    cur_nums = [v for v in current if isinstance(v, (int, float))]
    if not ref_nums or not cur_nums:
        return 0.0
    low, high = min(ref_nums), max(ref_nums)
    outside = sum(1 for v in cur_nums if v < low or v > high)
    return outside / len(cur_nums)


def _longest_run(values: Sequence[Any]) -> int:
    """Longest run of identical consecutive non-null values (a stuck feature)."""
    best = 0
    run = 0
    previous: Any = object()
    for value in values:
        if value is None:
            run = 0
            previous = object()
            continue
        run = run + 1 if value == previous else 1
        previous = value
        best = max(best, run)
    return best


def _schema_order_changed(
    reference: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]
) -> bool:
    if not reference or not current:
        return False
    return list(reference[0]["features"].keys()) != list(current[0]["features"].keys())


def integrity_signals(
    reference: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    features: Sequence[str],
) -> IntegritySignals:
    """Compute the integrity family for one monitoring window.

    `reference` is the training-window data, `current` the monitoring window,
    and `features` the feature names to check.
    """
    null_rate: dict[str, float] = {}
    cardinality_ratio: dict[str, float] = {}
    range_violation_rate: dict[str, float] = {}
    stale_run_length: dict[str, int] = {}

    for feature in features:
        ref_values = _values(reference, feature)
        cur_values = _values(current, feature)
        null_rate[feature] = _null_rate(cur_values)
        cardinality_ratio[feature] = _cardinality_ratio(ref_values, cur_values)
        range_violation_rate[feature] = _range_violation_rate(ref_values, cur_values)
        stale_run_length[feature] = _longest_run(cur_values)

    return IntegritySignals(
        null_rate=null_rate,
        cardinality_ratio=cardinality_ratio,
        range_violation_rate=range_violation_rate,
        stale_run_length=stale_run_length,
        schema_order_changed=_schema_order_changed(reference, current),
    )
