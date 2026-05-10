import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from autonoma.api.routes import health, predict
from autonoma.core.config import settings
from autonoma.core.logging import setup_logging

REQUEST_COUNT = Counter(
    "autonoma_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "autonoma_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

app = FastAPI(
    title="AUTONOMA",
    version="0.1.0",
    description="Self-healing MLOps platform with an autonomous AI agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(latency)

    return response


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(health.router)
app.include_router(predict.router)


@app.on_event("startup")
async def startup():
    setup_logging(log_level=settings.log_level, app_env=settings.app_env)
    logger.info("AUTONOMA starting up")


@app.on_event("shutdown")
async def shutdown():
    logger.info("AUTONOMA shutting down")
