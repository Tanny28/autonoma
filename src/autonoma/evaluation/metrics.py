"""Episode-level scoring for evaluation arms (#32).

An episode is one injected scenario, seen as a sequence of windows. An arm's
response to the episode is the first action it takes, in window order: once a
model is retrained or rolled back, the episode is over. An arm that never acts
has response None.

Latencies are in records, measured from the injected onset to the end of the
window where the arm detected or acted.

Performance retention, one of the nine planned metrics, is not here: it needs
the executor to apply each action and the replay to score the model that
results. Bundles alone cannot measure it.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from autonoma.agent.arms import Arm, Decision
from autonoma.core.bundle import GroundTruth, LabelledBundle, SignalBundle
from autonoma.core.taxonomy import CORRECT_ACTION, NO_RETRAIN_CAUSES, Action, Cause, is_harmful


@dataclass(frozen=True)
class EpisodeResult:
    truth: GroundTruth
    action: Action | None
    detected: bool
    first_action_window: int | None
    cause: Cause | None = None
    confidence: float | None = None
    first_detect_window: int | None = None
    decision: Decision | None = field(default=None, compare=False)
    decision_index: int | None = None
    windows_seen: int = 0
    seconds: float = 0.0


def episodes(rows: list[LabelledBundle]) -> list[list[LabelledBundle]]:
    """Group rows by episode, keeping first-seen episode order and sorting windows."""
    grouped: dict[str, list[LabelledBundle]] = defaultdict(list)
    for row in rows:
        grouped[row.truth.episode_id].append(row)
    return [sorted(w, key=lambda r: r.bundle.window_start) for w in grouped.values()]


def run_arm(arm: Arm, rows: list[LabelledBundle]) -> list[EpisodeResult]:
    """Feed each episode's windows to `arm` in order and record its first action.

    If the arm has a `learn(bundle, decision, confirmed_cause)` method (the
    agent with memory), it is told the true cause after each episode it acted
    on, so later episodes can benefit.
    """
    learn = getattr(arm, "learn", None)
    results = []
    for windows in episodes(rows):
        truth = windows[0].truth
        detected_at = acted = None
        decision = None
        start = time.perf_counter()
        for i, row in enumerate(windows):
            d = arm(row.bundle)
            if d.drift_detected and detected_at is None:
                detected_at = row.bundle.window_end
            if d.action is not None:
                decision, acted = d, i
                break
        seconds = time.perf_counter() - start
        if decision is not None and learn:
            learn(windows[acted].bundle, decision, truth.cause)
        results.append(
            EpisodeResult(
                truth=truth,
                action=decision.action if decision else None,
                detected=detected_at is not None,
                first_action_window=windows[acted].bundle.window_end if decision else None,
                cause=decision.cause if decision else None,
                confidence=decision.confidence if decision else None,
                first_detect_window=detected_at,
                decision=decision,
                decision_index=acted,
                windows_seen=len(windows) if acted is None else acted + 1,
                seconds=seconds,
            )
        )
    return results


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def harmful_action_rate(results: list[EpisodeResult]) -> float | None:
    """Share of no-retrain episodes (pipeline fault, seasonal) that the arm retrained.

    The headline metric. None if the scenario set has no such episodes.
    """
    relevant = [r for r in results if r.truth.cause in NO_RETRAIN_CAUSES]
    if not relevant:
        return None
    return sum(r.action is Action.RETRAIN for r in relevant) / len(relevant)


def correct_action_rate(results: list[EpisodeResult]) -> float:
    """Share of episodes where the arm's action matches the reference policy."""
    if not results:
        return 0.0
    return sum(r.action is CORRECT_ACTION[r.truth.cause] for r in results) / len(results)


def detection_rate(results: list[EpisodeResult]) -> float:
    """Share of episodes in which the arm flagged drift at least once."""
    if not results:
        return 0.0
    return sum(r.detected for r in results) / len(results)


def diagnostic_accuracy(results: list[EpisodeResult]) -> float | None:
    """Share of episodes diagnosed with the true cause. None for arms that never diagnose."""
    if not any(r.cause for r in results):
        return None
    return sum(r.cause is r.truth.cause for r in results) / len(results)


# ponytail: placeholder weights, agree them with the team before the final run.
COST_HARMFUL = 5.0  # retrain on a pipeline fault or seasonal swing
COST_MISSED = 3.0  # degradation left untreated
COST_WRONG = 2.0  # any other wrong action
COST_ESCALATE = 1.0  # a human has to look at it


def action_cost(cause: Cause, action: Action | None) -> float:
    if action is CORRECT_ACTION[cause]:
        return 0.0
    if action is None:
        return COST_MISSED
    if is_harmful(cause, action):
        return COST_HARMFUL
    if action is Action.ESCALATE:
        return COST_ESCALATE
    return COST_WRONG


def cost_weighted_error(results: list[EpisodeResult]) -> float | None:
    return _mean([action_cost(r.truth.cause, r.action) for r in results])


def _calibrated(results: list[EpisodeResult]) -> list[tuple[float, bool]]:
    return [
        (r.confidence, r.cause is r.truth.cause)
        for r in results
        if r.cause is not None and r.confidence is not None
    ]


def brier_score(results: list[EpisodeResult]) -> float | None:
    """Mean squared gap between stated confidence and whether the diagnosis was right."""
    return _mean([(c - right) ** 2 for c, right in _calibrated(results)])


def reliability_bins(results: list[EpisodeResult], bins: int = 5) -> list[dict]:
    """Data for a reliability diagram: per confidence bin, mean confidence vs accuracy."""
    grouped: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, right in _calibrated(results):
        grouped[min(int(c * bins), bins - 1)].append((c, right))
    return [
        {
            "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
            "mean_confidence": _mean([c for c, _ in grouped[b]]),
            "accuracy": _mean([float(r) for _, r in grouped[b]]),
            "count": len(grouped[b]),
        }
        for b in sorted(grouped)
    ]


def detection_latency(results: list[EpisodeResult]) -> float | None:
    """Mean records from onset to first detection, over episodes detected after onset."""
    return _mean(
        [
            r.first_detect_window - r.truth.onset_index
            for r in results
            if r.first_detect_window is not None and r.first_detect_window >= r.truth.onset_index
        ]
    )


def acted_before_onset_rate(results: list[EpisodeResult]) -> float | None:
    """Share of episodes where the arm acted before the fault existed: a false alarm."""
    return _mean(
        [
            float(r.first_action_window is not None and r.first_action_window < r.truth.onset_index)
            for r in results
        ]
    )


def mttr(results: list[EpisodeResult]) -> float | None:
    """Mean records from onset to the correct action, over correctly handled episodes."""
    return _mean(
        [
            r.first_action_window - r.truth.onset_index
            for r in results
            if r.action is CORRECT_ACTION[r.truth.cause]
            and r.first_action_window is not None
            and r.first_action_window >= r.truth.onset_index
        ]
    )


def recurrence_improvement(results: list[EpisodeResult]) -> float | None:
    """Correct-action rate on repeat occurrences of a cause minus that on first occurrences.

    Results must be in run order. Positive means the arm got better at causes it
    had seen before, which is what H2 predicts for the agent with memory.
    """
    seen: set[Cause] = set()
    first, repeat = [], []
    for r in results:
        (repeat if r.truth.cause in seen else first).append(
            float(r.action is CORRECT_ACTION[r.truth.cause])
        )
        seen.add(r.truth.cause)
    if not first or not repeat:
        return None
    return _mean(repeat) - _mean(first)


def seconds_per_decision(results: list[EpisodeResult]) -> float | None:
    windows = sum(r.windows_seen for r in results)
    return sum(r.seconds for r in results) / windows if windows else None


def ablate(bundle: SignalBundle, signal: str) -> SignalBundle:
    """Blank one signal, e.g. "integrity.null_rate", back to its empty default."""
    family, name = signal.split(".")
    group = getattr(bundle, family)
    empty = type(group).model_fields[name].get_default(call_default_factory=True)
    return bundle.model_copy(update={family: group.model_copy(update={name: empty})})


def faithfulness(
    make_arm: Callable[[], Arm], rows: list[LabelledBundle], results: list[EpisodeResult]
) -> float | None:
    """Share of explained decisions that change when the claimed decisive signal is removed.

    Each episode is replayed on a fresh arm up to the deciding window, which is
    fed with the named signal blanked out. If the decision stays the same, the
    explanation did not describe what actually drove it. Higher is better.
    None if the arm never names a decisive signal.
    """
    by_id = {w[0].truth.episode_id: w for w in episodes(rows)}
    changed = []
    for r in results:
        if r.decision is None or r.decision.decisive_signal is None:
            continue
        arm = make_arm()
        windows = by_id[r.truth.episode_id]
        for row in windows[: r.decision_index]:
            arm(row.bundle)
        again = arm(ablate(windows[r.decision_index].bundle, r.decision.decisive_signal))
        changed.append(float((again.cause, again.action) != (r.cause, r.action)))
    return _mean(changed)


def summarize(results: list[EpisodeResult]) -> dict[str, float | None]:
    """Every metric for one arm, keyed by its report name."""
    return {
        "harmful_action_rate": harmful_action_rate(results),
        "correct_action_rate": correct_action_rate(results),
        "diagnostic_accuracy": diagnostic_accuracy(results),
        "cost_weighted_error": cost_weighted_error(results),
        "brier_score": brier_score(results),
        "detection_rate": detection_rate(results),
        "detection_latency": detection_latency(results),
        "acted_before_onset": acted_before_onset_rate(results),
        "mttr": mttr(results),
        "recurrence_improvement": recurrence_improvement(results),
        "seconds_per_decision": seconds_per_decision(results),
    }


def evaluate(
    arms: dict[str, Callable[[], Arm]], rows: list[LabelledBundle]
) -> dict[str, dict[str, float | None]]:
    """Run every arm on the identical scenario set. Arms are factories: each run starts fresh."""
    report = {}
    for name, make in arms.items():
        results = run_arm(make(), rows)
        report[name] = summarize(results) | {"faithfulness": faithfulness(make, rows, results)}
    return report
