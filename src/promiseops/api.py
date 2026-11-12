"""PromiseOps serving API: /predict, /healthz, /readyz, /metrics."""

from __future__ import annotations

import logging
import math
import secrets
import time
import uuid
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from promiseops.contracts import PredictionRequest, PredictionResponse
from promiseops.data import (
    FEATURE_COLUMNS,
    ConflictError,
    build_features,
    get_connection,
    get_or_save_prediction,
    init_db,
)
from promiseops.model_pipeline.bundle import load_bundle
from promiseops.settings import Settings

logger = logging.getLogger("promiseops.api")

API_KEY_HEADER = "X-API-Key"
UNMATCHED_ROUTE = "unmatched"


class OrderNotFoundError(LookupError):
    """The order does not exist or is not released yet."""


class UnauthorizedError(Exception):
    """Missing or wrong API key."""


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def create_app(
    settings: Settings | None = None,
    *,
    bundle_loader=load_bundle,
    connection_factory=None,
) -> FastAPI:
    registry = CollectorRegistry()
    requests_total = Counter(
        "promiseops_http_requests_total", "HTTP requests", ["route", "status"], registry=registry
    )
    latency = Histogram(
        "promiseops_http_request_seconds", "HTTP latency", ["route"], registry=registry
    )
    missing_features = Counter(
        "promiseops_feature_missing_total", "Null feature values at serving", ["feature"], registry=registry
    )
    state: dict = {"settings": None, "model": None, "release": None, "schema_ready": False}

    def open_connection():
        if connection_factory is not None:
            return connection_factory()
        return get_connection(state["settings"].database_url)

    def ensure_schema() -> bool:
        if state["schema_ready"]:
            return True
        try:
            with closing(open_connection()) as conn:
                init_db(conn)
            state["schema_ready"] = True
        except Exception as error:  # noqa: BLE001 - never leak DB details
            logger.error("schema init failed: %s", type(error).__name__)
        return state["schema_ready"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["settings"] = settings if settings is not None else Settings.from_env()
        release_path = Path(state["settings"].release_path)
        release_dir = release_path if release_path.is_dir() else release_path.parent
        try:
            state["model"], state["release"] = bundle_loader(release_dir)
        except Exception as error:  # noqa: BLE001 - service stays up, /predict returns 503
            state["model"] = state["release"] = None
            logger.error("model bundle not loaded: %s", type(error).__name__)
        ensure_schema()
        yield

    app = FastAPI(title="PromiseOps", lifespan=lifespan)

    def require_key(request: Request) -> None:
        supplied = request.headers.get(API_KEY_HEADER, "")
        expected = state["settings"].api_key
        if not secrets.compare_digest(supplied.encode(), expected.encode()):
            raise UnauthorizedError

    @app.exception_handler(UnauthorizedError)
    async def _unauthorized(_request: Request, _exc: Exception):
        return _error(401, "unauthorized")

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, _exc: Exception):
        return _error(422, "invalid request")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:  # noqa: BLE001
            logger.error("unhandled error: %s", type(error).__name__)
            response = _error(500, "internal error")
        route = getattr(request.scope.get("route"), "path", UNMATCHED_ROUTE)
        requests_total.labels(route=route, status=str(response.status_code)).inc()
        latency.labels(route=route).observe(time.perf_counter() - started)
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if state["model"] is None:
            return _error(503, "not ready")
        if not ensure_schema():
            return _error(503, "not ready")
        try:
            with closing(open_connection()) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:  # noqa: BLE001
            return _error(503, "not ready")
        return {"status": "ready"}

    @app.get("/metrics", dependencies=[Depends(require_key)])
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    def score(conn, order_id: str) -> dict:
        model, release = state["model"], state["release"]
        as_of = datetime.now(UTC)
        frame = build_features(conn, [order_id], as_of=as_of)
        if frame.empty:
            raise OrderNotFoundError(order_id)
        features = frame.loc[:, list(FEATURE_COLUMNS)]
        for name in FEATURE_COLUMNS:
            if features[name].isna().any():
                missing_features.labels(feature=name).inc()
        column = list(model.classes_).index(1)
        probability = float(model.predict_proba(features)[0][column])
        if not math.isfinite(probability):
            raise ValueError("non-finite probability")
        return {
            "risk_probability": min(1.0, max(0.0, probability)),
            "model_version": str(release["model_version"]),
        }

    @app.post("/predict", dependencies=[Depends(require_key)])
    async def predict(request: Request):
        try:
            body = PredictionRequest.model_validate(await request.json())
        except (ValidationError, ValueError):
            return _error(422, "invalid request")
        if state["model"] is None:
            return _error(503, "model unavailable")
        request_id = body.request_id or uuid.uuid4().hex
        return await _run_predict(body.order_id, request_id)

    async def _run_predict(order_id: str, request_id: str):
        def work():
            with closing(open_connection()) as conn:
                return get_or_save_prediction(
                    conn,
                    request_id=request_id,
                    order_id=order_id,
                    compute_fn=lambda: score(conn, order_id),
                )

        try:
            result = await run_in_threadpool(work)
        except ConflictError:
            return _error(409, "request_id already used for another order")
        except OrderNotFoundError:
            return _error(404, "order not found")
        except Exception as error:  # noqa: BLE001
            logger.error("prediction failed: %s", type(error).__name__)
            return _error(500, "internal error")
        response = PredictionResponse(**result)
        logger.info("prediction request_id=%s model_version=%s", request_id, response.model_version)
        return response.model_dump()

    return app