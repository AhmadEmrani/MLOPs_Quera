"""Prediction use case: features -> model -> idempotent persistence."""

from __future__ import annotations

import logging
import math
import time
from contextlib import closing
from datetime import UTC, datetime

from promiseops.api.errors import OrderNotFoundError
from promiseops.api.state import AppState
from promiseops.data import FEATURE_COLUMNS, build_features, get_or_save_prediction
from promiseops.feature_lineage import observation

logger = logging.getLogger("promiseops.api")


def score(state: AppState, conn, order_id: str) -> dict:
    model, release = state.model, state.release
    # Preserve the same point-in-time cutoff for extraction and evidence.
    # AppState.startup already initializes schema: never run DDL per request.
    feature_as_of = datetime.now(UTC)
    frame = build_features(conn, [order_id], as_of=feature_as_of, ensure_schema=False)
    if frame.empty:
        raise OrderNotFoundError(order_id)
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    state.metrics.observe_features(
        {name: int(features[name].isna().sum()) for name in FEATURE_COLUMNS},
        len(features),
    )
    started = time.perf_counter()
    try:
        column = list(model.classes_).index(1)
        probability = float(model.predict_proba(features)[0][column])
        if not math.isfinite(probability):
            raise ValueError("non-finite probability")
    except Exception:
        state.metrics.inference_errors.inc()
        raise
    finally:
        state.metrics.inference_latency.observe(time.perf_counter() - started)
    # Preserve the API's existing clipping behavior and observe the served risk.
    probability = min(1.0, max(0.0, probability))
    state.metrics.inferences.inc()
    state.metrics.risk.observe(probability)
    return {
        "risk_probability": probability,
        "model_version": str(release["model_version"]),
        "_feature_observation": observation(
            features, feature_as_of=feature_as_of, release=release
        ),
    }


def predict_order(state: AppState, *, order_id: str, request_id: str) -> dict:
    """Blocking; call from a worker thread. Raises ConflictError / OrderNotFoundError."""
    with closing(state.open_connection()) as conn:
        return get_or_save_prediction(
            conn,
            request_id=request_id,
            order_id=order_id,
            compute_fn=lambda: score(state, conn, order_id),
        )
