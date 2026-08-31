from fastapi import APIRouter

from autonoma import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__, "service": "autonoma"}


@router.get("/health/ready")
async def health_ready() -> dict:
    return {"status": "ready"}
