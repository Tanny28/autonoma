import pytest

from autonoma.agent.arms import Decision
from autonoma.core.taxonomy import Action
from autonoma.remediation.executor import Executor, MlflowRegistry, Status


class FakeRegistry:
    def __init__(self):
        self.versions, self.champ, self.prev = ["model-v1"], 1, None

    def champion(self):
        return self.versions[self.champ - 1]

    def promote(self, model, justification):
        self.versions.append(model)
        self.prev, self.champ = self.champ, len(self.versions)
        return str(self.champ)

    def rollback(self, justification):
        if self.prev:
            self.champ, self.prev = self.prev, self.champ
        return str(self.champ)


SCORES = {"model-v1": 0.80, "better": 0.85, "worse": 0.70, "recalibrated": 0.81}


def executor(candidate="better", **kw):
    return Executor(
        FakeRegistry(),
        retrain=lambda: candidate,
        recalibrate=lambda champion: "recalibrated",
        score=SCORES.__getitem__,
        **kw,
    )


def decide(action):
    return Decision(drift_detected=True, action=action, justification="because")


def test_better_candidate_replaces_champion():
    ex = executor("better")
    out = ex.execute(decide(Action.RETRAIN))
    assert out.status is Status.APPLIED and ex.registry.champion() == "better"
    assert out.justification == "because"


def test_worse_candidate_is_rejected_and_champion_kept():
    ex = executor("worse")
    assert ex.execute(decide(Action.RETRAIN)).status is Status.REJECTED
    assert ex.registry.champion() == "model-v1"


def test_recalibration_goes_through_the_same_gate():
    ex = executor()
    assert ex.execute(decide(Action.RECALIBRATE)).status is Status.APPLIED
    assert ex.registry.champion() == "recalibrated"


def test_rollback_restores_the_previous_champion():
    ex = executor("better")
    ex.execute(decide(Action.RETRAIN))
    assert ex.execute(decide(Action.ROLLBACK)).status is Status.APPLIED
    assert ex.registry.champion() == "model-v1"


@pytest.mark.parametrize(
    ("action", "status"),
    [
        (None, Status.NO_ACTION),
        (Action.SUPPRESS, Status.APPLIED),
        (Action.ESCALATE, Status.ESCALATED),
    ],
)
def test_actions_that_leave_the_model_alone(action, status):
    ex = executor()
    assert ex.execute(decide(action)).status is status
    assert ex.registry.champion() == "model-v1"


def test_high_impact_action_waits_for_approval():
    ex = executor(needs_approval=frozenset({Action.RETRAIN}))
    assert ex.execute(decide(Action.RETRAIN)).status is Status.AWAITING_APPROVAL
    assert ex.registry.champion() == "model-v1"
    ex.approve = lambda decision: True
    assert ex.execute(decide(Action.RETRAIN)).status is Status.APPLIED


def test_mlflow_registry_promotes_and_rolls_back(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from sklearn.dummy import DummyClassifier

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    first = DummyClassifier(strategy="constant", constant=0).fit([[0], [1]], [0, 1])
    second = DummyClassifier(strategy="constant", constant=1).fit([[0], [1]], [0, 1])
    with mlflow.start_run():
        mlflow.sklearn.log_model(first, name="model", registered_model_name="M")
    registry = MlflowRegistry("M")
    registry.client.set_registered_model_alias("M", "champion", "1")

    assert registry.rollback("nothing older") == "1"
    assert registry.promote(second, "drift") == "2"
    assert registry.champion().predict([[0]])[0] == 1
    assert registry.rollback("bad data") == "1"
    assert registry.champion().predict([[0]])[0] == 0
