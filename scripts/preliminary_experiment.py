"""Preliminary experiment: arms 1-3 on injected degradation episodes in Elec2.

Each episode takes a 3,500-record segment of the live stream (after the
training split), injects one degradation class at record 1,000, and cuts the
segment into 500-record windows. Every window becomes a SignalBundle computed
against the 2,000 clean records just before the segment. Arms see only the
bundles; the injected cause is kept as ground truth for scoring.

Two setups are reported:
  default     off-the-shelf thresholds (PSI 0.2, KS/chi-square p < 0.01,
              5-point accuracy drop) on all features
  calibrated  thresholds set from clean windows in an earlier calibration range
              (never the test episodes); calendar fields (day, period) are not
              monitored as distributions; KS is disabled because at these sample
              sizes it flags every clean window

Run:  PYTHONPATH=src python scripts/preliminary_experiment.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from autonoma.agent import arms as arms_mod
from autonoma.agent import rules as rules_mod
from autonoma.agent.arms import alert_only, always_retrain
from autonoma.agent.rules import RuleArm
from autonoma.core.bundle import (
    DistributionalSignals,
    GroundTruth,
    IntegritySignals,
    PerformanceSignals,
    SignalBundle,
)
from autonoma.core.taxonomy import CORRECT_ACTION, NO_RETRAIN_CAUSES, Action, Cause

OUT = Path("results/preliminary")
TRAIN_FRACTION = 0.30
VIC_LIVE_FROM = 17_424  # vicprice/vicdemand/transfer are constant before this row
CALIB_END = 25_000  # clean calibration windows come from before this row; episodes after
REF_LEN, SEG_LEN, WIN, ONSET = 2_000, 3_500, 500, 1_000
RAMP, SEASON = 1_500, 1_500
SEVERITY_K = {1: 0.5, 2: 1.0, 3: 2.0}
SEEDS = range(5)

NUMERIC = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
FEATURES = ["day", *NUMERIC]
TARGET_FEATURE = "nswdemand"

SETUP = {"monitor": NUMERIC, "chi2": True}


def load() -> tuple[pd.DataFrame, np.ndarray]:
    d = fetch_openml(data_id=151, as_frame=True)
    x = d.data[FEATURES].copy()
    x["day"] = x["day"].astype(int)
    y = (d.target == "UP").astype(int).to_numpy()
    return x, y


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def longest_run(v: np.ndarray) -> int:
    best = run = 1
    for a, b in zip(v[:-1], v[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def inject(seg, y, ref, cause: Cause, sev: int, seed: int):
    seg, y = seg.copy(), y.copy()
    rng = np.random.default_rng(seed)
    k = SEVERITY_K[sev]
    std = ref[TARGET_FEATURE].std()
    t = np.arange(len(seg)) - ONSET
    col = seg.columns.get_loc(TARGET_FEATURE)
    if cause is Cause.SUDDEN_COVARIATE:
        seg.iloc[ONSET:, col] += k * std
    elif cause is Cause.GRADUAL_COVARIATE:
        seg.iloc[:, col] += k * std * np.clip(t / RAMP, 0, 1)
    elif cause is Cause.CONCEPT:
        region = (seg["nswprice"] > ref["nswprice"].median()).to_numpy()
        flip = region & (t >= 0) & (rng.random(len(seg)) < 0.3 * sev)
        y[flip] = 1 - y[flip]
    elif cause is Cause.PIPELINE_FAULT:
        kind = seed % 3
        if kind == 0:  # nulls from a failed join
            hit = (t >= 0) & (rng.random(len(seg)) < 0.1 * sev)
            seg.loc[hit, TARGET_FEATURE] = np.nan
        elif kind == 1:  # unit conversion error
            seg.iloc[ONSET:, col] *= 10.0**sev
        else:  # feature frozen at a stale value
            seg.iloc[ONSET:, col] = seg.iloc[ONSET - 1, col]
    elif cause is Cause.SEASONAL:
        wave = np.where((t >= 0) & (t < SEASON), np.sin(np.pi * t / SEASON), 0.0)
        seg.iloc[:, col] += k * std * wave
    return seg, y


def bundle(eid, start, w, ref, ref_acc, ref_f1, yw, pw) -> SignalBundle:
    ks_s, ks_p, ps, nulls, rng_v, stale, card = {}, {}, {}, {}, {}, {}, {}
    for f in SETUP["monitor"]:
        r, c = ref[f].to_numpy(), w[f].to_numpy()
        cc = c[~np.isnan(c)]
        if len(cc):
            res = ks_2samp(r, cc)
            ks_s[f], ks_p[f] = float(res.statistic), float(res.pvalue)
        ps[f] = psi(r, c)
        nulls[f] = float(np.isnan(c).mean())
        rng_v[f] = float(((cc < r.min()) | (cc > r.max())).mean()) if len(cc) else 0.0
        stale[f] = longest_run(c)
        card[f] = len(np.unique(cc)) / max(len(np.unique(r[-len(c) :])), 1)
    chi = {}
    if SETUP["chi2"]:
        table = pd.crosstab(np.r_[np.zeros(len(ref)), np.ones(len(w))], np.r_[ref["day"], w["day"]])
        chi = {"day": float(chi2_contingency(table)[1])}
    acc, f1 = accuracy_score(yw, pw), f1_score(yw, pw, zero_division=0)
    return SignalBundle(
        dataset="elec2",
        episode_id=eid,
        window_start=start,
        window_end=start + len(w) - 1,
        distributional=DistributionalSignals(
            ks_statistic=ks_s, ks_pvalue=ks_p, psi=ps, chi2_pvalue=chi
        ),
        performance=PerformanceSignals(
            labels_available=len(yw),
            accuracy=acc,
            f1=f1,
            accuracy_delta=acc - ref_acc,
            f1_delta=f1 - ref_f1,
        ),
        integrity=IntegritySignals(
            null_rate=nulls,
            range_violation_rate=rng_v,
            stale_run_length=stale,
            cardinality_ratio=card,
        ),
    )


def reference_stats(x, y, model, s0):
    ref = x.iloc[s0 - REF_LEN : s0]
    rp = model.predict(ref)
    yr = y[s0 - REF_LEN : s0]
    return ref, accuracy_score(yr, rp), f1_score(yr, rp, zero_division=0)


def build_episodes(x, y, model):
    lo, hi = CALIB_END + REF_LEN, len(x) - SEG_LEN
    episodes = []
    for ci, cause in enumerate(Cause):
        for sev in SEVERITY_K:
            for seed in SEEDS:
                s0 = int(np.random.default_rng(1000 * ci + 10 * sev + seed).integers(lo, hi))
                ref, ref_acc, ref_f1 = reference_stats(x, y, model, s0)
                seg, ys = inject(
                    x.iloc[s0 : s0 + SEG_LEN], y[s0 : s0 + SEG_LEN], ref, cause, sev, seed
                )
                pred = model.predict(seg)
                eid = f"{cause.value}-s{sev}-seed{seed}"
                bundles = [
                    bundle(
                        eid,
                        s0 + i,
                        seg.iloc[i : i + WIN],
                        ref,
                        ref_acc,
                        ref_f1,
                        ys[i : i + WIN],
                        pred[i : i + WIN],
                    )
                    for i in range(0, SEG_LEN, WIN)
                ]
                truth = GroundTruth(
                    episode_id=eid, cause=cause, onset_index=s0 + ONSET, severity=sev, seed=seed
                )
                episodes.append((truth, bundles))
    return episodes


def clean_windows(x, y, model, n=200):
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        s0 = int(rng.integers(VIC_LIVE_FROM + REF_LEN, CALIB_END - WIN))
        ref, ref_acc, ref_f1 = reference_stats(x, y, model, s0)
        w = x.iloc[s0 : s0 + WIN]
        out.append(bundle("calib", s0, w, ref, ref_acc, ref_f1, y[s0 : s0 + WIN], model.predict(w)))
    return out


def false_alarm_rates(windows):
    d = [b.distributional for b in windows]
    return {
        "KS p<0.01 (any feature)": float(np.mean([min(v.ks_pvalue.values()) < 0.01 for v in d])),
        "PSI>0.2 (any feature)": float(np.mean([max(v.psi.values()) > 0.2 for v in d])),
        "Chi-square p<0.01 (day)": float(np.mean([v.chi2_pvalue.get("day", 1) < 0.01 for v in d])),
        "Accuracy drop >5 points": float(
            np.mean([b.performance.accuracy_delta < -0.05 for b in windows])
        ),
        "Any integrity rule": float(
            np.mean([bool(rules_mod.integrity_faults(b)) for b in windows])
        ),
    }


def score(episodes):
    makers = {
        "Alert-only": lambda: alert_only,
        "Always-retrain": lambda: always_retrain,
        "Threshold rules": RuleArm,
    }
    rows, recs = [], []
    for name, make in makers.items():
        arm = make()
        mine = []
        for truth, bundles in episodes:
            act = cause_hat = when = None
            detected = False
            for b in bundles:
                dec = arm(b)
                detected |= dec.drift_detected
                if dec.action is not None:
                    act, cause_hat, when = dec.action, dec.cause, b.window_end
                    break
            mine.append(
                {
                    "arm": name,
                    "episode": truth.episode_id,
                    "true_cause": truth.cause.value,
                    "severity": truth.severity,
                    "action": act.value if act else None,
                    "diagnosed": cause_hat.value if cause_hat else None,
                    "detected": detected,
                    "acted_before_onset": when is not None and when < truth.onset_index,
                }
            )
        df = pd.DataFrame(mine)
        no_rt = df[df.true_cause.isin([c.value for c in NO_RETRAIN_CAUSES])]
        rows.append(
            {
                "Arm": name,
                "Harmful action rate": float((no_rt.action == Action.RETRAIN.value).mean()),
                "Correct action rate": float(
                    np.mean(
                        [
                            a == CORRECT_ACTION[Cause(c)].value
                            for a, c in zip(df.action, df.true_cause, strict=True)
                        ]
                    )
                ),
                "Diagnostic accuracy": float((df.diagnosed == df.true_cause).mean())
                if df.diagnosed.notna().any()
                else np.nan,
                "Acted before fault": float(df.acted_before_onset.mean()),
            }
        )
        recs += mine
    return pd.DataFrame(rows), pd.DataFrame(recs)


def main() -> None:
    x, y = load()
    n_train = int(len(x) * TRAIN_FRACTION)
    model = HistGradientBoostingClassifier(random_state=42).fit(x[:n_train], y[:n_train])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    report = {
        "dataset": "Elec2 (OpenML 151)",
        "rows": len(x),
        "train_rows": n_train,
        "model": "HistGradientBoostingClassifier",
        "features": FEATURES,
    }

    # Setup A: off-the-shelf thresholds
    report["default_false_alarms_on_clean_windows"] = false_alarm_rates(clean_windows(x, y, model))
    episodes = build_episodes(x, y, model)
    report["episodes"] = len(episodes)
    summary_a, recs_a = score(episodes)

    # Setup B: thresholds calibrated on clean history
    SETUP.update(monitor=["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"], chi2=False)
    calib = clean_windows(x, y, model)
    psi_thr = float(np.quantile([max(b.distributional.psi.values()) for b in calib], 0.95))
    acc_thr = float(np.quantile([b.performance.accuracy_delta for b in calib], 0.05))
    arms_mod.PSI_DRIFT = psi_thr
    arms_mod.KS_PVALUE_DRIFT = 0.0
    arms_mod.ACCURACY_DROP = rules_mod.ACCURACY_DROP = acc_thr
    report["calibrated_thresholds"] = {"psi": psi_thr, "accuracy_drop": acc_thr}
    report["calibrated_false_alarms_on_clean_windows"] = {
        "drift_detected": float(np.mean([arms_mod.drift_detected(b) for b in calib])),
        "any_integrity_rule": float(np.mean([bool(rules_mod.integrity_faults(b)) for b in calib])),
    }
    summary_b, recs_b = score(build_episodes(x, y, model))

    summary = pd.concat([summary_a.assign(Setup="default"), summary_b.assign(Setup="calibrated")])
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat([recs_a.assign(setup="default"), recs_b.assign(setup="calibrated")]).to_csv(
        OUT / "episodes.csv", index=False
    )
    rules = recs_b[recs_b.arm == "Threshold rules"]
    confusion = pd.crosstab(rules.true_cause, rules.diagnosed.fillna("no action"))
    confusion.to_csv(OUT / "rules_confusion_calibrated.csv")
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(summary.round(3).to_string(index=False))
    print("\nRule arm (calibrated): true cause vs diagnosis\n", confusion.to_string())


if __name__ == "__main__":
    main()
