"""Arms 5 and 6: the LLM diagnostic agent, without and with incident memory (#27).

One LLM call per monitoring window. The prompt carries the current signal
bundle, a short trend of the episode's earlier windows and, for arm 6, the
most similar past incidents. The model must answer in JSON with a cause and
an action from core/taxonomy.py. Anything outside that closed space becomes
an escalation, so the agent can never invent an action.

The episode id is never shown to the model: in the scenario set it spells
out the injected cause.

`GroqComplete` caches every response on disk, keyed by the exact prompt, so a
rerun is free and gives identical decisions. It also counts tokens, which the
evaluation reports as inference cost.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from autonoma.agent.arms import Decision
from autonoma.agent.memory import IncidentMemory, signature
from autonoma.agent.rules import integrity_faults
from autonoma.core.bundle import SignalBundle
from autonoma.core.taxonomy import Action, Cause

#: (system prompt, user prompt) -> response text
Complete = Callable[[str, str], str]

DEFAULT_MODEL = "llama-3.3-70b-versatile"
TREND_WINDOWS = 6

#: Signals the agent may name as decisive. Each one has an empty default, so
#: the faithfulness check can blank it out.
DECISIVE_SIGNALS = frozenset(
    f"{family}.{name}"
    for family, names in {
        "distributional": ["ks_statistic", "ks_pvalue", "psi", "chi2_pvalue", "adwin_drift"],
        "performance": ["accuracy", "f1", "accuracy_delta", "f1_delta"],
        "integrity": [
            "null_rate",
            "cardinality_ratio",
            "range_violation_rate",
            "stale_run_length",
            "schema_order_changed",
        ],
    }.items()
    for name in names
)

SYSTEM_PROMPT = f"""You diagnose why a deployed ML model is degrading.

You get monitoring signals for the current window, the trend over earlier
windows of the same stream, and sometimes similar past incidents. Decide
whether something is wrong, what the root cause is, and which action to take.

Causes and the action each one needs:
- {Cause.SUDDEN_COVARIATE}: inputs shift abruptly and stay shifted. Action: {Action.RETRAIN}.
- {Cause.GRADUAL_COVARIATE}: inputs drift slowly over several windows. Action: {Action.RETRAIN}.
- {Cause.CONCEPT}: accuracy drops while input distributions look stable. Action: {Action.RETRAIN}.
- {Cause.PIPELINE_FAULT}: the data itself is broken (nulls, values far outside the
  reference range, a frozen feature, collapsed cardinality, reordered columns).
  Retraining would learn the broken data. Action: {Action.ROLLBACK}.
- {Cause.SEASONAL}: inputs move but accuracy holds and the shift recedes on its own.
  Retraining throws away a good model. Action: {Action.SUPPRESS}.
If you cannot tell, use action {Action.ESCALATE} so a human decides.
If inputs moved but it is too early to tell seasonal from covariate drift,
set "action" to null to keep watching.

Answer with one JSON object and nothing else:
{{"degraded": true or false,
 "cause": one of {[c.value for c in Cause]} or null,
 "action": one of {[a.value for a in Action]} or null,
 "confidence": number from 0 to 1,
 "decisive_signal": the one signal that drove the decision, as "family.field",
   e.g. "integrity.null_rate" or "performance.accuracy_delta", or null,
 "justification": one or two sentences}}"""


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def window_summary(bundle: SignalBundle) -> dict:
    """One line of the trend: the headline numbers of an earlier window."""
    return {
        "window_start": bundle.window_start,
        **_rounded(signature(bundle)),
        "integrity_faults": len(integrity_faults(bundle)),
    }


def build_prompt(bundle: SignalBundle, trend: list[dict], past: list[dict]) -> str:
    current = bundle.model_dump(exclude={"episode_id"})
    parts = [
        "Current window:\n" + json.dumps(_rounded(current)),
        "Earlier windows, oldest first:\n" + (json.dumps(trend) if trend else "none"),
    ]
    if past:
        parts.append("Similar past incidents and how they turned out:\n" + json.dumps(past))
    return "\n\n".join(parts)


def parse(text: str) -> Decision:
    """Turn the model's reply into a Decision; malformed replies escalate."""
    try:
        data = json.loads(text)
        if not data.get("degraded"):
            return Decision(drift_detected=False, action=None, justification="no degradation")
        cause = Cause(data["cause"]) if data.get("cause") else None
        action = Action(data["action"]) if data.get("action") else None
        confidence = min(max(float(data.get("confidence", 0.5)), 0.0), 1.0)
        decisive = data.get("decisive_signal")
        return Decision(
            drift_detected=True,
            action=action,
            cause=cause,
            confidence=confidence,
            justification=str(data.get("justification", "")),
            decisive_signal=decisive if decisive in DECISIVE_SIGNALS else None,
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return Decision(
            drift_detected=True,
            action=Action.ESCALATE,
            confidence=0.0,
            justification=f"Unusable LLM reply, escalating: {text[:200]}",
        )


class LLMAgent:
    """Arm 5 without `memory`, arm 6 with it. Use one instance per evaluation run."""

    def __init__(self, complete: Complete, memory: IncidentMemory | None = None) -> None:
        self.complete = complete
        self.memory = memory
        self._trend: dict[str, list[dict]] = {}

    def __call__(self, bundle: SignalBundle) -> Decision:
        trend = self._trend.setdefault(bundle.episode_id, [])
        past = self.memory.recall(bundle) if self.memory else []
        decision = parse(
            self.complete(SYSTEM_PROMPT, build_prompt(bundle, trend[-TREND_WINDOWS:], past))
        )
        trend.append(window_summary(bundle))
        return decision

    def learn(self, bundle: SignalBundle, decision: Decision, confirmed: Cause) -> None:
        """Called by the evaluation once an episode closes and its true cause is known."""
        if self.memory:
            self.memory.record(bundle, decision, confirmed)


class GroqComplete:
    """Groq chat completion with a prompt-keyed disk cache and token accounting."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Path = Path("results/llm_cache"),
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._client = None
        self.api_calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, system: str, user: str) -> str:
        key = hashlib.sha256(json.dumps([self.model, system, user]).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = self._call(system, user)
            self.api_calls += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")
        # Count cached calls too: the reported cost is what one run costs, cache or not.
        self.prompt_tokens += record["prompt_tokens"]
        self.completion_tokens += record["completion_tokens"]
        return record["text"]

    def _call(self, system: str, user: str) -> dict:
        if self._client is None:
            from groq import Groq  # optional dependency: pip install -e ".[agent]"

            self._client = Groq(api_key=self.api_key)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return {
            "text": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
