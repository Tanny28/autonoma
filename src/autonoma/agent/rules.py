"""Arm 3: hand-written threshold rules over the signal bundle.

This is the ablation that isolates the LLM's value (H1). If this arm matches
the agent, the honest conclusion is that LLM reasoning is unnecessary here.
So the rules should be as good as a careful engineer can reasonably make them.

Rules, checked in order:

1. Any integrity fault           -> upstream pipeline fault -> rollback
2. No drift at all               -> no action
3. Accuracy drops, inputs stable -> concept drift           -> retrain
4. Inputs drift, accuracy drops  -> sudden / gradual drift  -> retrain
5. Inputs drift, accuracy holds  -> wait and watch:
     drift receding              -> seasonal variation      -> suppress
     drift persists              -> sudden / gradual drift  -> retrain

Sudden and gradual drift are told apart by how fast PSI rose across the
episode's windows.

The arm keeps per-episode history of earlier bundles, because a single
window cannot separate seasonal variation from covariate drift. It sees
nothing except bundles, the same inputs every arm gets.

The thresholds below are starting values. Tune them on a training split of
the scenario set (#24), never on the evaluation split.
"""

from dataclasses import dataclass, field

from autonoma.agent.arms import ACCURACY_DROP, Decision, drift_detected
from autonoma.core.bundle import SignalBundle
from autonoma.core.taxonomy import Action, Cause

NULL_RATE_FAULT = 0.05
RANGE_VIOLATION_FAULT = 0.10
STALE_WINDOW_FRACTION = 0.5
CARDINALITY_COLLAPSE = 0.10

SUDDEN_JUMP = 0.2
RECEDING_FRACTION = 0.7
PERSIST_WINDOWS = 3


@dataclass
class _EpisodeHistory:
    psi: list[float] = field(default_factory=list)
    benign_drift_windows: int = 0


def _max_psi(bundle: SignalBundle) -> float:
    return max(bundle.distributional.psi.values(), default=0.0)


def _distribution_drifted(bundle: SignalBundle) -> bool:
    no_perf = bundle.model_copy(
        update={"performance": bundle.performance.model_copy(update={"accuracy_delta": None})}
    )
    return drift_detected(no_perf)


def _accuracy_dropped(bundle: SignalBundle) -> bool:
    delta = bundle.performance.accuracy_delta
    return delta is not None and delta < ACCURACY_DROP


def integrity_faults(bundle: SignalBundle) -> list[str]:
    """Human-readable list of integrity rules the bundle breaks."""
    i = bundle.integrity
    window_len = bundle.window_end - bundle.window_start + 1
    faults = [f"{f}: null rate {v:.0%}" for f, v in i.null_rate.items() if v > NULL_RATE_FAULT]
    faults += [
        f"{f}: {v:.0%} of values outside reference range"
        for f, v in i.range_violation_rate.items()
        if v > RANGE_VIOLATION_FAULT
    ]
    faults += [
        f"{f}: same value for {n} consecutive records"
        for f, n in i.stale_run_length.items()
        if n > STALE_WINDOW_FRACTION * window_len
    ]
    faults += [
        f"{f}: cardinality collapsed to {v:.0%} of reference"
        for f, v in i.cardinality_ratio.items()
        if v < CARDINALITY_COLLAPSE
    ]
    if i.schema_order_changed:
        faults.append("column order differs from reference schema")
    return faults


class RuleArm:
    """Stateful rule policy. Use one instance per evaluation run."""

    def __init__(self) -> None:
        self._history: dict[str, _EpisodeHistory] = {}

    def __call__(self, bundle: SignalBundle) -> Decision:
        history = self._history.setdefault(bundle.episode_id, _EpisodeHistory())
        psi_now = _max_psi(bundle)
        previous = history.psi[:]
        history.psi.append(psi_now)

        faults = integrity_faults(bundle)
        if faults:
            return Decision(
                drift_detected=True,
                action=Action.ROLLBACK,
                cause=Cause.PIPELINE_FAULT,
                confidence=0.9,
                justification="Integrity fault: " + "; ".join(faults),
            )

        if not drift_detected(bundle):
            history.benign_drift_windows = 0
            return Decision(drift_detected=False, action=None)

        if not _distribution_drifted(bundle):
            return Decision(
                drift_detected=True,
                action=Action.RETRAIN,
                cause=Cause.CONCEPT,
                confidence=0.7,
                justification="Accuracy dropped while input distributions stayed stable.",
            )

        covariate = self._covariate_cause(psi_now, previous)
        if _accuracy_dropped(bundle):
            return Decision(
                drift_detected=True,
                action=Action.RETRAIN,
                cause=covariate,
                confidence=0.8,
                justification=f"Inputs drifted (max PSI {psi_now:.2f}) and accuracy dropped.",
            )

        peak = max(previous, default=0.0)
        if peak > 0 and psi_now < RECEDING_FRACTION * peak:
            return Decision(
                drift_detected=True,
                action=Action.SUPPRESS,
                cause=Cause.SEASONAL,
                confidence=0.6,
                justification=(
                    f"Drift receding (PSI {psi_now:.2f} from peak {peak:.2f}) "
                    "with accuracy intact; treating as seasonal."
                ),
            )

        history.benign_drift_windows += 1
        if history.benign_drift_windows >= PERSIST_WINDOWS:
            return Decision(
                drift_detected=True,
                action=Action.RETRAIN,
                cause=covariate,
                confidence=0.6,
                justification=(
                    f"Inputs drifted for {PERSIST_WINDOWS} windows without receding; "
                    "treating as covariate drift."
                ),
            )

        return Decision(
            drift_detected=True,
            action=None,
            justification="Inputs drifted but accuracy holds; watching for recovery.",
        )

    @staticmethod
    def _covariate_cause(psi_now: float, previous: list[float]) -> Cause:
        series = [0.0, *previous, psi_now]
        biggest_jump = max(b - a for a, b in zip(series, series[1:], strict=False))
        return Cause.SUDDEN_COVARIATE if biggest_jump > SUDDEN_JUMP else Cause.GRADUAL_COVARIATE
