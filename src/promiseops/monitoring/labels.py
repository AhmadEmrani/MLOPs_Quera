"""Delayed-label evaluation, strictly using outcomes released at the requested as_of."""
from __future__ import annotations
from datetime import timedelta
from promiseops.metrics import evaluate
from promiseops.monitoring.contracts import (
    Prediction, instant, iso, unique_predictions,
    MIN_MATURE_LABELS, PREDICTION_RETENTION_DAYS,
)


def evaluate_mature(rows: list[Prediction], *, as_of: object,
                    min_labels: int = MIN_MATURE_LABELS) -> dict:
    if min_labels < 1:
        raise ValueError("min_labels must be positive")
    t = instant(as_of)
    oldest = t - timedelta(days=PREDICTION_RETENTION_DAYS)
    available = [p for p in unique_predictions(rows, t)
                 if oldest <= p.scored_as_of <= t]
    # Any accidentally supplied future label is not usable in a past replay.
    mature = [p for p in available if p.label is not None
              and p.label_available_at is not None
              and instant(p.label_available_at) <= t]
    classes = sorted(set(p.label for p in mature))
    result = {"as_of": iso(t), "lookback_days": PREDICTION_RETENTION_DAYS,
              "n_unique_predictions": len(available), "n_mature": len(mature),
              "classes": classes, "sufficient": len(mature) >= min_labels and classes == [0, 1],
              "metrics": None}
    if result["sufficient"]:
        result["metrics"] = evaluate(
            [p.risk_probability for p in mature],
            [p.label for p in mature],
            [p.order_id for p in mature],
        )
    else:
        result["reason"] = "insufficient_mature_labels" if len(mature) < min_labels else "single_class_labels"
    return result
