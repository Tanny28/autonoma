from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0", "service": "autonoma"}


@router.get("/health/ready")
async def health_ready() -> dict:
    return {"status": "ready"}
