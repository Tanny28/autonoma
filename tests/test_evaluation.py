from autonoma.core.taxonomy import Action, Cause, CORRECT_ACTION
from autonoma.evaluation.arms import (
    arm_1_alert_only,
    arm_2_always_retrain,
    arm_3_threshold_rules,
)


def test_arm_1_alert_only_takes_no_action():
    for cause in Cause:
        result = arm_1_alert_only(cause)

        assert result.arm == "A1"
        assert result.action is Action.SUPPRESS
        assert result.harmful is False


def test_arm_2_always_retrain_is_harmful_for_no_retrain_causes():
    pipeline_result = arm_2_always_retrain(Cause.PIPELINE_FAULT)
    seasonal_result = arm_2_always_retrain(Cause.SEASONAL)

    assert pipeline_result.action is Action.RETRAIN
    assert pipeline_result.harmful is True

    assert seasonal_result.action is Action.RETRAIN
    assert seasonal_result.harmful is True


def test_arm_2_retrain_is_not_harmful_for_retrain_causes():
    for cause in (
        Cause.SUDDEN_COVARIATE,
        Cause.GRADUAL_COVARIATE,
        Cause.CONCEPT,
    ):
        result = arm_2_always_retrain(cause)

        assert result.action is Action.RETRAIN
        assert result.harmful is False


def test_arm_3_follows_correct_action_policy():
    assert (
        arm_3_threshold_rules(Cause.SUDDEN_COVARIATE).action
        is Action.RETRAIN
    )

    assert (
        arm_3_threshold_rules(Cause.GRADUAL_COVARIATE).action
        is Action.RETRAIN
    )

    assert (
        arm_3_threshold_rules(Cause.CONCEPT).action
        is Action.RETRAIN
    )

    assert (
        arm_3_threshold_rules(Cause.PIPELINE_FAULT).action
        is Action.ROLLBACK
    )

    assert (
        arm_3_threshold_rules(Cause.SEASONAL).action
        is Action.SUPPRESS
    )


def test_arm_3_does_not_make_harmful_retrain_decisions():
    for cause in Cause:
        result = arm_3_threshold_rules(cause)

        assert result.harmful is False

def test_arm_4_trained_classifier_returns_a4():
    from autonoma.evaluation.arms import arm_4_trained_classifier

    result = arm_4_trained_classifier(Cause.SUDDEN_COVARIATE)

    assert result.arm == "A4"
    assert result.action == CORRECT_ACTION[Cause.SUDDEN_COVARIATE]
    assert result.harmful is False


def test_arm_4_does_not_retrain_on_no_retrain_causes():
    from autonoma.evaluation.arms import arm_4_trained_classifier

    pipeline_result = arm_4_trained_classifier(Cause.PIPELINE_FAULT)
    seasonal_result = arm_4_trained_classifier(Cause.SEASONAL)

    assert pipeline_result.action == Action.ROLLBACK
    assert seasonal_result.action == Action.SUPPRESS
    assert pipeline_result.harmful is False
    assert seasonal_result.harmful is False