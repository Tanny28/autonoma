import numpy as np

from autonoma.core.bundle import DistributionalSignals
from autonoma.signals.distributional import (
    adwin_detected,
    chi2_pvalue,
    distributional_signals,
    ks_test,
    psi,
)


def _records(values, category=None):
    return [
        {"features": {"x": float(v), "c": category[i] if category else "a"}}
        for i, v in enumerate(values)
    ]


def _normal(seed, n=500, mean=0.0):
    return np.random.default_rng(seed).normal(mean, 1.0, n).tolist()


def test_identical_windows_show_no_drift():
    ref = _records(_normal(1))
    stat, p = ks_test(ref, ref, "x")
    assert stat == 0.0
    assert p == 1.0
    assert abs(psi(ref, ref, "x")) < 1e-9


def test_shifted_window_is_detected_by_ks_and_psi():
    ref = _records(_normal(1))
    cur = _records(_normal(2, mean=2.0))
    stat, p = ks_test(ref, cur, "x")
    assert stat > 0.5
    assert p < 0.001
    assert psi(ref, cur, "x") > 0.25


def test_same_distribution_shows_small_drift():
    ref = _records(_normal(1))
    cur = _records(_normal(3))
    stat, p = ks_test(ref, cur, "x")
    assert stat < 0.15
    assert p > 0.01


def test_psi_handles_constant_reference():
    ref = _records([5.0] * 100)
    cur = _records(_normal(4))
    assert psi(ref, cur, "x") == 0.0


def test_chi2_detects_changed_category_shares():
    ref = _records([0] * 300, category=["a"] * 150 + ["b"] * 150)
    cur = _records([0] * 300, category=["a"] * 280 + ["b"] * 20)
    assert chi2_pvalue(ref, cur, "c") < 0.001


def test_chi2_same_shares_is_not_significant():
    ref = _records([0] * 200, category=["a"] * 100 + ["b"] * 100)
    cur = _records([0] * 200, category=["a"] * 100 + ["b"] * 100)
    assert chi2_pvalue(ref, cur, "c") > 0.5


def test_adwin_detects_a_change_in_prediction_stream():
    stable = [0.1] * 300
    changed = [0.1] * 300 + [0.9] * 300
    assert adwin_detected(stable) is False
    assert adwin_detected(changed) is True


def test_bundle_collects_all_signals():
    ref = _records(_normal(1))
    cur = _records(_normal(2, mean=2.0))
    signals = distributional_signals(
        ref,
        cur,
        numeric_features=["x"],
        categorical_features=["c"],
        prediction_stream=[0.1] * 300 + [0.9] * 300,
    )
    assert isinstance(signals, DistributionalSignals)
    assert set(signals.ks_statistic) == {"x"}
    assert set(signals.ks_pvalue) == {"x"}
    assert set(signals.psi) == {"x"}
    assert set(signals.chi2_pvalue) == {"c"}
    assert signals.adwin_drift is True
