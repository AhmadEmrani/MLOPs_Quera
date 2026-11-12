"""Prometheus metrics with bounded labels and the HTTP observing middleware."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from promiseops.api.errors import error_response

logger = logging.getLogger("promiseops.api")

UNMATCHED_ROUTE = "unmatched"


class Metrics:
    """Per-app registry. Labels: route template, status, fixed feature names only."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "promiseops_http_requests_total", "HTTP requests", ["route", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "promiseops_http_request_seconds", "HTTP latency", ["route"], registry=self.registry
        )
        self.missing_features = Counter(
            "promiseops_feature_missing_total",
            "Null feature values at serving",
            ["feature"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


def install_metrics_middleware(app: FastAPI, metrics: Metrics) -> None:
    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:  # noqa: BLE001 - never leak internals
            logger.error("unhandled error: %s", type(error).__name__)
            response = error_response(500, "internal error")
        route = getattr(request.scope.get("route"), "path", UNMATCHED_ROUTE)
        metrics.requests_total.labels(route=route, status=str(response.status_code)).inc()
        metrics.latency.labels(route=route).observe(time.perf_counter() - started)
        return response
