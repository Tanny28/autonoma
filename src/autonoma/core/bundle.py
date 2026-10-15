"""The signal bundle: the one interface every layer shares.

Signal extractors (Pranav, Dushyant) produce a `SignalBundle` for each
monitoring window. Every evaluation arm consumes the identical bundle: the
alert-only and always-retrain baselines, the threshold rules, the trained
classifier and the LLM agent. If an arm reads anything that is not in the
bundle, the comparison between arms is invalid.

Per-feature signals are dicts keyed by feature name, so the same schema works
for Elec2, Airlines and UCI Credit without changes.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autonoma.core.taxonomy import Cause


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DistributionalSignals(_Frozen):
    """Has the input distribution moved away from the reference window?"""

    ks_statistic: dict[str, float] = Field(
        default_factory=dict, description="Two-sample KS statistic per numeric feature."
    )
    ks_pvalue: dict[str, float] = Field(
        default_factory=dict, description="KS p-value per numeric feature."
    )
    psi: dict[str, float] = Field(
        default_factory=dict, description="Population Stability Index per feature."
    )
    chi2_pvalue: dict[str, float] = Field(
        default_factory=dict, description="Chi-square p-value per categorical feature."
    )
    adwin_drift: bool = Field(
        default=False, description="ADWIN detected a change in the prediction stream."
    )


class PerformanceSignals(_Frozen):
    """Is the model getting worse? Only available once delayed labels arrive."""

    labels_available: int = Field(
        ge=0, description="Labelled records in the window; 0 means no performance signal yet."
    )
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    accuracy_delta: float | None = Field(
        default=None, description="Window accuracy minus reference accuracy."
    )
    f1_delta: float | None = Field(default=None, description="Window F1 minus reference F1.")

    @model_validator(mode="after")
    def _no_metrics_without_labels(self):
        if self.labels_available == 0 and any(
            v is not None for v in (self.accuracy, self.f1, self.accuracy_delta, self.f1_delta)
        ):
            raise ValueError("performance metrics require labels_available > 0")
        return self


class IntegritySignals(_Frozen):
    """Is the data itself broken? Separates pipeline faults from genuine drift."""

    null_rate: dict[str, float] = Field(
        default_factory=dict, description="Fraction of nulls per feature, 0–1."
    )
    cardinality_ratio: dict[str, float] = Field(
        default_factory=dict,
        description="Distinct values in window / distinct values in reference, per feature.",
    )
    range_violation_rate: dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of values outside the reference min–max, per feature. "
        "Catches unit-scaling faults.",
    )
    stale_run_length: dict[str, int] = Field(
        default_factory=dict,
        description="Longest run of identical consecutive values per feature. "
        "Catches a feature frozen at a stale constant.",
    )
    schema_order_changed: bool = Field(
        default=False, description="Column order differs from the reference schema."
    )


class SignalBundle(_Frozen):
    """Everything an evaluation arm is allowed to see about one monitoring window."""

    dataset: str = Field(description='Dataset name, e.g. "elec2".')
    episode_id: str = Field(description="Identifies one injected scenario run.")
    window_start: int = Field(ge=0, description="First record_index in the window.")
    window_end: int = Field(description="Last record_index in the window (inclusive).")

    distributional: DistributionalSignals
    performance: PerformanceSignals
    integrity: IntegritySignals

    @model_validator(mode="after")
    def _window_ordered(self):
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        return self


class GroundTruth(_Frozen):
    """What the injector actually did. Written by the injector, read only by evaluation.

    Kept separate from `SignalBundle` on purpose: no arm may see the true cause.
    """

    episode_id: str
    cause: Cause
    onset_index: int = Field(ge=0)
    severity: int = Field(ge=1, le=3)
    seed: int


class LabelledBundle(_Frozen):
    """A bundle paired with its ground truth: one row of the scenario set."""

    bundle: SignalBundle
    truth: GroundTruth

    @model_validator(mode="after")
    def _same_episode(self):
        if self.bundle.episode_id != self.truth.episode_id:
            raise ValueError("bundle and truth belong to different episodes")
        return self


def write_jsonl(rows: list[LabelledBundle], path: Path) -> None:
    """Save a scenario set, one labelled bundle per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[LabelledBundle]:
    """Load a scenario set written by `write_jsonl`."""
    with path.open(encoding="utf-8") as f:
        return [LabelledBundle.model_validate_json(line) for line in f if line.strip()]
