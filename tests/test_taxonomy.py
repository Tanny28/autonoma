from autonoma.core.taxonomy import (
    CORRECT_ACTION,
    NO_RETRAIN_CAUSES,
    Action,
    Cause,
    is_harmful,
)


def test_taxonomy_is_five_by_five():
    assert len(Cause) == 5
    assert len(Action) == 5


def test_every_cause_has_a_reference_action():
    assert set(CORRECT_ACTION) == set(Cause)


def test_no_retrain_causes_are_not_mapped_to_retrain():
    for cause in NO_RETRAIN_CAUSES:
        assert CORRECT_ACTION[cause] is not Action.RETRAIN


def test_harmful_action_is_retrain_on_a_no_retrain_cause():
    assert is_harmful(Cause.PIPELINE_FAULT, Action.RETRAIN)
    assert is_harmful(Cause.SEASONAL, Action.RETRAIN)
    assert not is_harmful(Cause.CONCEPT, Action.RETRAIN)
    assert not is_harmful(Cause.PIPELINE_FAULT, Action.ROLLBACK)
