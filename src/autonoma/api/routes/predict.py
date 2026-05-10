from fastapi import APIRouter

from autonoma.api.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(
        prediction="up",
        confidence=0.5,
        model_version="stub-v0",
        note="Stub endpoint — real model arrives in Week 2",
    )
