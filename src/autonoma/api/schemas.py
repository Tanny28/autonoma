from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str
    note: str | None = None
