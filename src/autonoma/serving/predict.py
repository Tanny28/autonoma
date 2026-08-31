from fastapi import APIRouter

from autonoma.serving.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])

# ponytail: constant-response stub so the replay harness (O1) can be built and
# tested against a live endpoint before the baseline model exists. Replaced by
# the MLflow-loaded Elec2 classifier; the response contract does not change.
_STUB_VERSION = "stub-v0"


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction=1, confidence=0.5, model_version=_STUB_VERSION)
