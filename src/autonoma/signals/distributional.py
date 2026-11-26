"""Distributional signals (O1): has the input moved away from the reference window?

Computed from two windows of records: a reference window (normal behaviour) and a
current window (the one being monitored). Returns a `DistributionalSignals` bundle.
This module never sees the true cause, so it imports nothing from the taxonomy
or from `GroundTruth`.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from river.drift import ADWIN
from scipy import stats

from autonoma.core.bundle import DistributionalSignals

PSI_BINS = 10
_EPS = 1e-6


def _numeric(records: Sequence[dict[str, Any]], feature: str) -> list[float]:
    return [
        float(r["features"][feature])
        for r in records
        if isinstance(r["features"].get(feature), (int, float))
    ]


def _categorical(records: Sequence[dict[str, Any]], feature: str) -> list[Any]:
    return [r["features"][feature] for r in records if r["features"].get(feature) is not None]


def ks_test(
    reference: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
    feature: str,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value for a numeric feature."""
    result = stats.ks_2samp(_numeric(reference, feature), _numeric(current, feature))
    return float(result.statistic), float(result.pvalue)


def psi(
    reference: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
    feature: str,
    bins: int = PSI_BINS,
) -> float:
    """Population Stability Index for a numeric feature.

    Bin edges come from reference quantiles, so each reference bin holds about the
    same share of records. Current values outside the reference range fall into the
    first or last bin. A constant reference feature gives 0.0.
    """
    ref = np.asarray(_numeric(reference, feature))
    cur = np.asarray(_numeric(current, feature))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0

    cur = np.clip(cur, edges[0], edges[-1])
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_share = np.clip(ref_share, _EPS, None)
    cur_share = np.clip(cur_share, _EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def chi2_pvalue(
    reference: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
    feature: str,
) -> float:
    """Chi-square p-value for a categorical feature: do the category shares differ?"""
    ref_values = _categorical(reference, feature)
    cur_values = _categorical(current, feature)
    categories = sorted({*ref_values, *cur_values}, key=str)
    if len(categories) < 2:
        return 1.0

    table = np.array(
        [
            [ref_values.count(c) for c in categories],
            [cur_values.count(c) for c in categories],
        ]
    )
    return float(stats.chi2_contingency(table)[1])


def adwin_detected(stream: Sequence[float]) -> bool:
    """True if ADWIN detects a change anywhere in the stream (for example, predictions)."""
    detector = ADWIN()
    detected = False
    for value in stream:
        detector.update(float(value))
        detected = detected or bool(detector.drift_detected)
    return detected


def distributional_signals(
    reference: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    prediction_stream: Sequence[float] = (),
) -> DistributionalSignals:
    """Compute all distributional signals for one monitoring window.

    A feature with no usable values (for example, entirely null) is left out: NaN does
    not survive the JSON round trip, and the integrity family reports the nulls.
    """
    ks_stat: dict[str, float] = {}
    ks_p: dict[str, float] = {}
    psi_by_feature: dict[str, float] = {}
    for feature in numeric_features:
        if not _numeric(reference, feature) or not _numeric(current, feature):
            continue
        ks_stat[feature], ks_p[feature] = ks_test(reference, current, feature)
        psi_by_feature[feature] = psi(reference, current, feature)

    chi2_by_feature = {
        feature: chi2_pvalue(reference, current, feature)
        for feature in categorical_features
        if _categorical(reference, feature) and _categorical(current, feature)
    }

    return DistributionalSignals(
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        psi=psi_by_feature,
        chi2_pvalue=chi2_by_feature,
        adwin_drift=adwin_detected(prediction_stream),
    )
