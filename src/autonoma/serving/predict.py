import mlflow.sklearn
import pandas as pd
from fastapi import APIRouter, Request

from autonoma.serving.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])

MODEL_NAME = "AutonomaElec2"
MODEL_ALIAS = "champion"


def load_model():
    """Load the champion Elec2 model from the MLflow registry."""
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    http_request: Request,
) -> PredictResponse:
    model = getattr(http_request.app.state, "model", None)

    if model is None:
        model = load_model()
        http_request.app.state.model = model

    features = pd.DataFrame([request.features])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=f"{MODEL_NAME}@{MODEL_ALIAS}",
    )
