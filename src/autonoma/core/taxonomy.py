"""The five degradation classes and the five remediation actions.

This module is the single source of truth for the project's taxonomy. The
injection layer (O1), the diagnostic agent (O2), the remediation executor (O3)
and the evaluation harness (O4) all import from here so that a class or action
is spelled exactly one way across the codebase.

The action space is deliberately closed: the agent selects from `Action`, it
cannot invent new actions. Widening it is a design change, not an
implementation detail (see project context, "Things to Never Do").
"""

from enum import StrEnum


class Cause(StrEnum):
    """Root cause of an observed model degradation episode."""

    SUDDEN_COVARIATE = "sudden_covariate_drift"
    GRADUAL_COVARIATE = "gradual_covariate_drift"
    CONCEPT = "concept_drift"
    PIPELINE_FAULT = "upstream_pipeline_fault"
    SEASONAL = "seasonal_variation"


class Action(StrEnum):
    """The closed remediation action space."""

    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    RECALIBRATE = "recalibrate"
    SUPPRESS = "suppress"
    ESCALATE = "escalate"


#: The remediation each cause *should* receive. This is the reference policy
#: every evaluation arm is scored against, and it encodes the project's central
#: argument: two of five causes must not trigger a retrain.
CORRECT_ACTION: dict[Cause, Action] = {
    Cause.SUDDEN_COVARIATE: Action.RETRAIN,
    Cause.GRADUAL_COVARIATE: Action.RETRAIN,
    Cause.CONCEPT: Action.RETRAIN,
    Cause.PIPELINE_FAULT: Action.ROLLBACK,
    Cause.SEASONAL: Action.SUPPRESS,
}

#: Causes for which retraining is actively harmful: it either bakes corrupted
#: upstream data into the model, or discards a valid model in response to a
#: fluctuation that would have reverted on its own. Retrains fired on these
#: causes are counted by the harmful-action-rate metric (the headline result).
NO_RETRAIN_CAUSES: frozenset[Cause] = frozenset({Cause.PIPELINE_FAULT, Cause.SEASONAL})


def is_harmful(cause: Cause, action: Action) -> bool:
    """True if `action` is a retrain on a cause where retraining does damage."""
    return action is Action.RETRAIN and cause in NO_RETRAIN_CAUSES
