"""O3 — carry out an arm's decision behind guardrails (#29, executor logic).

- Validation gate: a retrained or recalibrated candidate replaces the champion
  only if it scores higher on held-out data. Otherwise the champion stays.
- Reversible: promotions go through the MLflow registry, which keeps the old
  champion under the alias "previous" so rollback can restore it.
- Approval: actions listed in `needs_approval` wait for a human `approve` call.

Rate limiting and the audit log are Namrata's half of #29 and wrap around
`Executor.execute`; every `Outcome` carries the decision's justification for them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from autonoma.agent.arms import Decision
from autonoma.core.taxonomy import Action


class Status(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"  # candidate failed the validation gate
    AWAITING_APPROVAL = "awaiting_approval"
    ESCALATED = "escalated"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class Outcome:
    action: Action | None
    status: Status
    detail: str
    justification: str


class Registry(Protocol):
    def champion(self) -> Any: ...
    def promote(self, model: Any, justification: str) -> str: ...
    def rollback(self, justification: str) -> str: ...


class Executor:
    def __init__(
        self,
        registry: Registry,
        retrain: Callable[[], Any],
        recalibrate: Callable[[Any], Any],
        score: Callable[[Any], float],
        needs_approval: frozenset[Action] = frozenset(),
        approve: Callable[[Decision], bool] | None = None,
    ) -> None:
        """`retrain()` fits a candidate on recent data, `recalibrate(champion)` returns
        a recalibrated copy, `score(model)` evaluates on held-out data, higher is better."""
        self.registry = registry
        self.retrain = retrain
        self.recalibrate = recalibrate
        self.score = score
        self.needs_approval = needs_approval
        self.approve = approve

    def execute(self, decision: Decision) -> Outcome:
        action = decision.action

        def done(status: Status, detail: str) -> Outcome:
            return Outcome(action, status, detail, decision.justification)

        if action is None:
            return done(Status.NO_ACTION, "nothing to do")
        if action in self.needs_approval and not (self.approve and self.approve(decision)):
            return done(Status.AWAITING_APPROVAL, f"{action} needs human approval")

        if action is Action.SUPPRESS:
            return done(Status.APPLIED, "alert suppressed, champion kept")
        if action is Action.ESCALATE:
            return done(Status.ESCALATED, "handed to a human")
        if action is Action.ROLLBACK:
            version = self.registry.rollback(decision.justification)
            return done(Status.APPLIED, f"rolled back to version {version}")

        champion = self.registry.champion()
        candidate = self.retrain() if action is Action.RETRAIN else self.recalibrate(champion)
        new, old = self.score(candidate), self.score(champion)
        if new <= old:
            return done(Status.REJECTED, f"candidate {new:.4f} did not beat champion {old:.4f}")
        version = self.registry.promote(candidate, decision.justification)
        return done(Status.APPLIED, f"promoted version {version}: {new:.4f} vs {old:.4f}")


class MlflowRegistry:
    """Champion/previous aliases on one registered MLflow model."""

    def __init__(self, model_name: str = "AutonomaElec2") -> None:
        from mlflow import MlflowClient

        self.name = model_name
        self.client = MlflowClient()

    def champion(self) -> Any:
        import mlflow.sklearn

        return mlflow.sklearn.load_model(f"models:/{self.name}@champion")

    def _alias(self, alias: str) -> str:
        return str(self.client.get_model_version_by_alias(self.name, alias).version)

    def promote(self, model: Any, justification: str) -> str:
        import mlflow
        import mlflow.sklearn

        old = self._alias("champion")
        with mlflow.start_run(run_name="autonoma-remediation"):
            info = mlflow.sklearn.log_model(model, name="model", registered_model_name=self.name)
        version = str(info.registered_model_version)
        self.client.set_model_version_tag(self.name, version, "justification", justification)
        self.client.set_registered_model_alias(self.name, "previous", old)
        self.client.set_registered_model_alias(self.name, "champion", version)
        return version

    def rollback(self, justification: str) -> str:
        from mlflow.exceptions import MlflowException

        current = self._alias("champion")
        try:
            previous = self._alias("previous")
        except MlflowException:
            return current  # nothing older to go back to: the champion is the known-good model
        self.client.set_registered_model_alias(self.name, "champion", previous)
        self.client.set_registered_model_alias(self.name, "previous", current)
        self.client.set_model_version_tag(self.name, previous, "rollback_reason", justification)
        return previous
