from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One replayed record from the evaluation stream."""

    features: dict[str, float | int | str | None]
    record_index: int | None = Field(
        default=None,
        description="Position in the replay stream, used to align predictions "
        "with the injector's ground-truth onset index.",
    )


class PredictResponse(BaseModel):
    prediction: int
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str
