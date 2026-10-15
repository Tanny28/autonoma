"""Evaluation arms: policies that map a SignalBundle to a Decision.

Every arm has the same signature, `(SignalBundle) -> Decision`, and sees only
the bundle. Arms 1-2 live here; the rule arm (#30), classifier arm (#31) and
LLM agent (#27) follow the same signature so the evaluation can run them
side by side.
"""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from autonoma.core.bundle import SignalBundle
from autonoma.core.taxonomy import Action, Cause

# Standard industry thresholds, the kind of defaults monitoring tools ship with.
# The baselines use them as-is: tuning them would stop the arms representing
# current practice.
PSI_DRIFT = 0.2
KS_PVALUE_DRIFT = 0.01
ACCURACY_DROP = -0.05


class Decision(BaseModel):
    """What an arm decided for one window."""

    model_config = ConfigDict(frozen=True)

    drift_detected: bool
    action: Action | None = Field(description="None means no remediation was taken.")
    cause: Cause | None = Field(
        default=None, description="Diagnosed cause; None for arms that do not diagnose."
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""


Arm = Callable[[SignalBundle], Decision]


def drift_detected(bundle: SignalBundle) -> bool:
    """Conventional drift check on distributional and performance signals.

    Deliberately ignores the integrity family: existing monitoring tools don't
    distinguish a broken pipeline from a real distribution shift, and that gap
    is what this project measures.
    """
    d, p = bundle.distributional, bundle.performance
    return (
        d.adwin_drift
        or any(v > PSI_DRIFT for v in d.psi.values())
        or any(v < KS_PVALUE_DRIFT for v in d.ks_pvalue.values())
        or any(v < KS_PVALUE_DRIFT for v in d.chi2_pvalue.values())
        or (p.accuracy_delta is not None and p.accuracy_delta < ACCURACY_DROP)
    )


def alert_only(bundle: SignalBundle) -> Decision:
    """Arm 1: detect and alert, never act. Represents Evidently / Arize practice."""
    detected = drift_detected(bundle)
    return Decision(
        drift_detected=detected,
        action=None,
        justification="drift alert raised" if detected else "",
    )


def always_retrain(bundle: SignalBundle) -> Decision:
    """Arm 2: retrain on any drift signal. Represents published auto-retrain pipelines."""
    detected = drift_detected(bundle)
    return Decision(
        drift_detected=detected,
        action=Action.RETRAIN if detected else None,
        justification="drift detected, retraining" if detected else "",
    )
