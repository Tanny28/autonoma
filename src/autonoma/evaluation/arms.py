from dataclasses import dataclass

from autonoma.core.taxonomy import Action, Cause, CORRECT_ACTION


@dataclass(frozen=True)
class ArmDecision:
    arm: str
    action: Action
    harmful: bool


def arm_1_alert_only(cause: Cause) -> ArmDecision:
    """A1: detect the issue but take no remediation action."""
    return ArmDecision(
        arm="A1",
        action=Action.SUPPRESS,
        harmful=False,
    )


def arm_2_always_retrain(cause: Cause) -> ArmDecision:
    """A2: retrain whenever a degradation signal is detected."""
    action = Action.RETRAIN

    return ArmDecision(
        arm="A2",
        action=action,
        harmful=action is Action.RETRAIN and cause in {
            Cause.PIPELINE_FAULT,
            Cause.SEASONAL,
        },
    )


def arm_3_threshold_rules(cause: Cause) -> ArmDecision:
    """A3: use the reference rule-based remediation policy."""
    action = CORRECT_ACTION[cause]

    return ArmDecision(
        arm="A3",
        action=action,
        harmful=action is Action.RETRAIN and cause in {
            Cause.PIPELINE_FAULT,
            Cause.SEASONAL,
        },
    )
def arm_4_trained_classifier(cause: Cause) -> ArmDecision:
    """A4: conventional trained-classifier baseline.

    The classifier predicts the cause, then the reference remediation
    policy is applied to that predicted cause.
    """

    # Simulated classifier prediction.
    # In the evaluation pipeline, this value will come from
    # the trained classifier rather than the rule-based policy.
    predicted_cause = cause

    action = CORRECT_ACTION[predicted_cause]

    return ArmDecision(
        arm="A4",
        action=action,
        harmful=(
            action is Action.RETRAIN
            and predicted_cause in {
                Cause.PIPELINE_FAULT,
                Cause.SEASONAL,
            }
        ),
    )