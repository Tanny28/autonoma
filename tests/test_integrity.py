"""Tests for the integrity signal family.

Each signal must fire on its matching pipeline fault and stay quiet on
covariate drift (issue #22, "Done when").
"""

import random

from autonoma.core.bundle import IntegritySignals
from autonoma.injection.injector import pipeline_fault, sudden_covariate
from autonoma.signals.integrity import integrity_signals

FEATURES = ["x", "y"]
ONSET = 150


def _clean_stream(n: int = 300, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    return [
        {"features": {"x": rng.gauss(0.0, 1.0), "y": rng.gauss(5.0, 2.0)}, "label": 0}
        for _ in range(n)
    ]


def test_identical_windows_give_quiet_signals():
    ref = _clean_stream(seed=1)
    result = integrity_signals(ref, ref, FEATURES)
    assert isinstance(result, IntegritySignals)
    assert all(v == 0.0 for v in result.null_rate.values())
    assert all(v == 0.0 for v in result.range_violation_rate.values())
    assert all(abs(v - 1.0) < 1e-9 for v in result.cardinality_ratio.values())
    assert result.schema_order_changed is False


def test_null_rate_fires_on_null_fault():
    ref = _clean_stream(seed=1)
    cur = pipeline_fault(ref, "x", ONSET, severity=3, kind="null").records
    result = integrity_signals(ref, cur, FEATURES)
    # Only the post-onset half is hit, so the window-wide rate is about half the fault rate.
    assert result.null_rate["x"] > 0.1
    assert result.null_rate["y"] == 0.0


def test_range_violation_fires_on_scale_fault():
    ref = _clean_stream(seed=1)
    cur = pipeline_fault(ref, "x", ONSET, severity=3, kind="scale").records
    result = integrity_signals(ref, cur, FEATURES)
    # Half the window is scaled by 10**4; nearly all of those values fall outside the range.
    assert result.range_violation_rate["x"] > 0.4
    assert result.range_violation_rate["y"] == 0.0


def test_staleness_fires_on_freeze_fault():
    ref = _clean_stream(seed=1)
    cur = pipeline_fault(ref, "x", ONSET, severity=2, kind="freeze").records
    result = integrity_signals(ref, cur, FEATURES)
    # Frozen from onset to the end of the 300-record window.
    assert result.stale_run_length["x"] >= 300 - ONSET
    assert result.stale_run_length["y"] < 5


def test_schema_order_fires_on_reordered_columns():
    # The injector's "swap" fault swaps values, not column order, so this
    # case reorders the columns directly.
    ref = _clean_stream(seed=1)
    cur = [
        {"features": {"y": r["features"]["y"], "x": r["features"]["x"]}, "label": 0} for r in ref
    ]
    result = integrity_signals(ref, cur, FEATURES)
    assert result.schema_order_changed is True


def test_cardinality_fires_when_values_collapse():
    # No injected fault changes cardinality, so build the collapse by hand.
    ref = _clean_stream(seed=1)
    cur = [
        {"features": {"x": round(r["features"]["x"] % 3, 0), "y": r["features"]["y"]}, "label": 0}
        for r in ref
    ]
    result = integrity_signals(ref, cur, FEATURES)
    assert result.cardinality_ratio["x"] < 0.05
    assert abs(result.cardinality_ratio["y"] - 1.0) < 1e-9


def test_integrity_stays_quiet_on_covariate_drift():
    ref = _clean_stream(seed=1)
    cur = sudden_covariate(ref, "x", ONSET, severity=1).records
    result = integrity_signals(ref, cur, FEATURES)
    assert result.null_rate["x"] == 0.0
    assert result.range_violation_rate["x"] < 0.10  # the rule arm's fault threshold
    assert result.stale_run_length["x"] < 5
    assert abs(result.cardinality_ratio["x"] - 1.0) < 0.05
    assert result.schema_order_changed is False


def test_range_violation_fires_on_a_tenfold_unit_error():
    rng = random.Random(0)
    ref = [{"features": {"x": rng.uniform(0.1, 0.8)}} for _ in range(300)]
    cur = [{"features": {"x": r["features"]["x"] * 10}} for r in ref]
    assert integrity_signals(ref, cur, ["x"]).range_violation_rate["x"] > 0.8
