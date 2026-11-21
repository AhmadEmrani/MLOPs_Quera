"""Prometheus instrumentation; bounded labels and per-process observations."""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Mapping

from fastapi import FastAPI, Request
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info, generate_latest

from promiseops.api.errors import error_response
from promiseops.data import FEATURE_COLUMNS

logger = logging.getLogger("promiseops.api")
UNMATCHED_ROUTE = "unmatched"
_VERSION = re.compile(r"sha256:[0-9a-f]{64}")
# Only these immutable, aggregate fields may be exported from the bundle.
REFERENCE_FIELDS = {
    "recall_at_10pct": "Recall at ten percent on the recorded validation cohort",
    "precision_at_10pct": "Precision at ten percent on the recorded validation cohort",
    "average_precision": "Average precision (PR-AUC) on the recorded validation cohort",
    "brier": "Brier score on the recorded validation cohort",
    "n": "Size of the recorded validation cohort",
    "late_count": "Number of late orders in the recorded validation cohort",
    "prevalence": "Late-order prevalence on the recorded validation cohort",
    "k": "Review-list size on the recorded validation cohort",
}


class Metrics:
    """One registry per API process. No request IDs, order IDs or raw feature values."""

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
            "Feature rows with a null value for the named feature at serving",
            ["feature"], registry=self.registry,
        )
        # prometheus_client creates labelled children lazily. Materialize exactly
        # the fixed contract feature set even if no null has ever been seen.
        for name in FEATURE_COLUMNS:
            self.missing_features.labels(feature=name)
        self.feature_rows = Counter(
            "promiseops_feature_rows_total", "Feature rows inspected for nulls", registry=self.registry
        )
        self.inferences = Counter(
            "promiseops_model_inferences_total", "Successful fresh model computations (not unique orders)",
            registry=self.registry,
        )
        self.inference_errors = Counter(
            "promiseops_model_inference_errors_total", "Model computations that raised or returned invalid risk",
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "promiseops_model_inference_seconds", "Model computation latency, including failed attempts",
            registry=self.registry,
        )
        self.risk = Histogram(
            "promiseops_prediction_risk", "Risk probabilities of successful fresh computations",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "promiseops_model_loaded", "One if the validated model is loaded in this process",
            registry=self.registry,
        )
        self.model = Info(
            "promiseops_model", "Immutable validated model artifact identity (one value per process)",
            ["model_version"], registry=self.registry,
        )
        self.reference_available = Gauge(
            "promiseops_validation_reference_available", "One if required validation reference metrics exist",
            registry=self.registry,
        )
        self.reference = {
            name: Gauge(f"promiseops_validation_{name}", help_text, registry=self.registry)
            for name, help_text in REFERENCE_FIELDS.items()
        }
        self.set_release(None)

    def set_release(self, release: Mapping | None) -> None:
        """Expose only validated release identity and recorded aggregate metrics.

        Validation reference metrics are NOT delayed-label production metrics.
        Missing/invalid reference data is NaN rather than a fabricated zero.
        """
        self.model_loaded.set(int(release is not None))
        self.model.clear()  # Replace old metadata, including on startup failure.
        version = release.get("model_version") if release else None
        # The only extra identity dimension is an immutable artifact digest,
        # never an arbitrary version from a request or replayed prediction.
        if isinstance(version, str) and _VERSION.fullmatch(version):
            self.model.labels(model_version=version).info({})
        raw = release.get("metrics", {}) if release else {}
        values = raw if isinstance(raw, Mapping) else {}
        valid = set()
        for name, gauge in self.reference.items():
            value = values.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                if name in {"n", "late_count", "k"}:
                    good = value >= 0 and value == int(value)
                    if name in {"n", "k"}:
                        good = good and value > 0
                else:
                    good = 0 <= value <= 1
                if good:
                    gauge.set(value)
                    valid.add(name)
                    continue
            gauge.set(float("nan"))
        required = {"recall_at_10pct", "precision_at_10pct", "average_precision", "brier", "n", "prevalence"}
        self.reference_available.set(int(required.issubset(valid)))

    def observe_features(self, missing: Mapping[str, int], rows: int) -> None:
        """Use the same denominator for every feature; never create arbitrary labels."""
        if rows < 0 or set(missing) != set(FEATURE_COLUMNS):
            raise ValueError("feature observation must match the fixed contract")
        if any(count < 0 or count > rows for count in missing.values()):
            raise ValueError("invalid missing-feature count")
        self.feature_rows.inc(rows)
        for name in FEATURE_COLUMNS:
            count = missing[name]
            if count:
                self.missing_features.labels(feature=name).inc(count)

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
