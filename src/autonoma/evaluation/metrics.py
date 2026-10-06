"""Episode-level scoring for evaluation arms.

An episode is one injected scenario, seen as a sequence of windows. An arm's
response to the episode is the first action it takes, in window order: once a
model is retrained or rolled back, the episode is over. An arm that never acts
has response None.
"""

from collections import defaultdict
from dataclasses import dataclass

from autonoma.agent.arms import Arm
from autonoma.core.bundle import GroundTruth, LabelledBundle
from autonoma.core.taxonomy import CORRECT_ACTION, NO_RETRAIN_CAUSES, Action


@dataclass(frozen=True)
class EpisodeResult:
    truth: GroundTruth
    action: Action | None
    detected: bool
    first_action_window: int | None


def run_arm(arm: Arm, rows: list[LabelledBundle]) -> list[EpisodeResult]:
    """Feed each episode's windows to `arm` in order and record its first action."""
    episodes: dict[str, list[LabelledBundle]] = defaultdict(list)
    for row in rows:
        episodes[row.truth.episode_id].append(row)

    results = []
    for windows in episodes.values():
        windows.sort(key=lambda r: r.bundle.window_start)
        detected, action, at = False, None, None
        for row in windows:
            decision = arm(row.bundle)
            detected = detected or decision.drift_detected
            if decision.action is not None:
                action, at = decision.action, row.bundle.window_end
                break
        results.append(EpisodeResult(windows[0].truth, action, detected, at))
    return results


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
