import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from autonoma import __version__
from autonoma.core.config import settings
from autonoma.core.logging import setup_logging
from autonoma.serving import health, predict

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(log_level=settings.log_level, app_env=settings.app_env)
    logger.info("AUTONOMA serving layer starting up")
    yield
    logger.info("AUTONOMA serving layer shutting down")


app = FastAPI(
    title="AUTONOMA",
    version=__version__,
    description="Serving layer for the AUTONOMA self-healing MLOps platform",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    # Label on the route template, not the raw path: the replay harness sends
    # tens of thousands of requests and raw paths would blow up Prometheus
    # cardinality. Unmatched paths collapse to a single bucket.
    route = request.scope.get("route")
    endpoint = getattr(route, "path", "unmatched")

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
