import json

from autonoma.agent.llm import SYSTEM_PROMPT, GroqComplete, LLMAgent, build_prompt, parse
from autonoma.agent.memory import IncidentMemory
from autonoma.core.bundle import (
    DistributionalSignals,
    IntegritySignals,
    PerformanceSignals,
    SignalBundle,
)
from autonoma.core.taxonomy import Action, Cause


def bundle(episode="upstream_pipeline_fault-s2-seed0", i=0, null_rate=0.0, dataset="elec2"):
    return SignalBundle(
        dataset=dataset,
        episode_id=episode,
        window_start=i * 500,
        window_end=i * 500 + 499,
        distributional=DistributionalSignals(psi={"nswdemand": 0.05}),
        performance=PerformanceSignals(labels_available=500, accuracy=0.8, accuracy_delta=-0.01),
        integrity=IntegritySignals(null_rate={"nswdemand": null_rate}),
    )


def reply(**fields):
    base = {"degraded": True, "confidence": 0.9, "justification": "nulls"}
    return json.dumps(base | fields)


class FakeLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.text


def test_valid_reply_becomes_a_decision():
    d = parse(
        reply(
            cause="upstream_pipeline_fault",
            action="rollback",
            decisive_signal="integrity.null_rate",
        )
    )
    assert (d.cause, d.action, d.confidence) == (Cause.PIPELINE_FAULT, Action.ROLLBACK, 0.9)
    assert d.decisive_signal == "integrity.null_rate"


def test_action_outside_the_closed_space_escalates():
    d = parse(reply(cause="concept_drift", action="delete_the_model"))
    assert d.action is Action.ESCALATE and d.confidence == 0.0


def test_non_json_reply_escalates():
    assert parse("I think it is drift").action is Action.ESCALATE


def test_healthy_reply_takes_no_action():
    d = parse(json.dumps({"degraded": False}))
    assert not d.drift_detected and d.action is None


def test_null_action_means_keep_watching():
    d = parse(reply(cause=None, action=None))
    assert d.drift_detected and d.action is None


def test_unknown_decisive_signal_is_dropped():
    assert parse(reply(cause=None, action=None, decisive_signal="vibes")).decisive_signal is None


def test_confidence_is_clamped():
    assert parse(reply(cause=None, action=None, confidence=3)).confidence == 1.0


def test_prompt_hides_the_episode_id():
    prompt = build_prompt(bundle(), [], [])
    assert "upstream_pipeline_fault" not in prompt
    assert "upstream_pipeline_fault" in SYSTEM_PROMPT  # the cause list itself is fine


def test_agent_sends_earlier_windows_as_trend():
    llm = FakeLLM(reply(cause=None, action=None))
    agent = LLMAgent(llm)
    agent(bundle(i=0))
    agent(bundle(i=1))
    agent(bundle(episode="other", i=0))
    assert "none" in llm.prompts[0].split("Earlier windows")[1]
    assert '"window_start": 0' in llm.prompts[1].split("Earlier windows")[1]
    assert "none" in llm.prompts[2].split("Earlier windows")[1]


def test_memory_round_trip_and_nearest_first(tmp_path):
    memory = IncidentMemory(f"sqlite:///{tmp_path / 'm.db'}")
    rollback = parse(reply(cause="upstream_pipeline_fault", action="rollback"))
    retrain = parse(reply(cause="sudden_covariate_drift", action="retrain"))
    memory.record(bundle(null_rate=0.3), rollback, Cause.PIPELINE_FAULT)
    memory.record(bundle(null_rate=0.0), retrain, Cause.SEASONAL)
    memory.record(bundle(null_rate=0.3, dataset="airlines"), rollback, Cause.PIPELINE_FAULT)

    past = memory.recall(bundle(null_rate=0.25), k=5)
    assert len(past) == 2  # other datasets are not recalled
    assert past[0]["outcome"] == "correct"
    assert past[1]["outcome"] == "harmful"


def test_agent_with_memory_learns_and_recalls(tmp_path):
    memory = IncidentMemory(f"sqlite:///{tmp_path / 'm.db'}")
    llm = FakeLLM(reply(cause="upstream_pipeline_fault", action="rollback"))
    agent = LLMAgent(llm, memory)
    b = bundle(null_rate=0.3)
    agent.learn(b, agent(b), Cause.PIPELINE_FAULT)
    agent(bundle(episode="next", null_rate=0.3))
    assert "Similar past incidents" not in llm.prompts[0]
    assert '"confirmed_cause": "upstream_pipeline_fault"' in llm.prompts[1]


def test_groq_responses_are_cached_and_counted(tmp_path):
    client = GroqComplete(cache_dir=tmp_path)
    calls = []

    def fake_call(system, user):
        calls.append(user)
        return {"text": "{}", "prompt_tokens": 100, "completion_tokens": 20}

    client._call = fake_call
    assert client("s", "u") == client("s", "u") == "{}"
    assert len(calls) == 1 and client.api_calls == 1 and client.cache_hits == 1
    assert (client.prompt_tokens, client.completion_tokens) == (200, 40)
