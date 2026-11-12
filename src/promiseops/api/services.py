"""Prediction use case: features -> model -> idempotent persistence."""

from __future__ import annotations

import logging
import math
from contextlib import closing
from datetime import UTC, datetime

from promiseops.api.errors import OrderNotFoundError
from promiseops.api.state import AppState
from promiseops.data import FEATURE_COLUMNS, build_features, get_or_save_prediction

logger = logging.getLogger("promiseops.api")


def score(state: AppState, conn, order_id: str) -> dict:
    model, release = state.model, state.release
    frame = build_features(conn, [order_id], as_of=datetime.now(UTC))
    if frame.empty:
        raise OrderNotFoundError(order_id)
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    for name in FEATURE_COLUMNS:
        if features[name].isna().any():
            state.metrics.missing_features.labels(feature=name).inc()
    column = list(model.classes_).index(1)
    probability = float(model.predict_proba(features)[0][column])
    if not math.isfinite(probability):
        raise ValueError("non-finite probability")
    return {
        "risk_probability": min(1.0, max(0.0, probability)),
        "model_version": str(release["model_version"]),
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
