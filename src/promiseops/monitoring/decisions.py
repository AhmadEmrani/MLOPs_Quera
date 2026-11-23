"""Contract thresholds. This module recommends review; it NEVER runs a worker."""
from __future__ import annotations
from math import isfinite


def decide(drift: dict, mature: dict, validation_reference: dict | None) -> dict:
    if not mature.get("sufficient"):
        return {"status": "insufficient_labels", "retraining_recommended": False,
                "reasons": [mature.get("reason", "insufficient_mature_labels")],
                "manual_review": bool(drift.get("persistent_drift")),
                "automatic_training_permitted": False}
    reasons: list[str] = []
    if drift.get("persistent_drift") and drift.get("sufficient_windows"):
        reasons.append("persistent_feature_drift")
    metrics = mature["metrics"]
    if metrics["recall_at_10pct"] < 0.10:
        reasons.append("recall_below_random_review_baseline")
    if validation_reference is not None:
        keys = ("recall_at_10pct", "average_precision", "brier")
        if not all(key in validation_reference and
                   isinstance(validation_reference[key], (float, int)) and
                   isfinite(float(validation_reference[key])) for key in keys):
            raise ValueError("invalid validation reference metrics")
        if metrics["recall_at_10pct"] < 0.8 * validation_reference["recall_at_10pct"]:
            reasons.append("recall_degraded_vs_validation")
        if metrics["average_precision"] < validation_reference["average_precision"] - 0.05:
            reasons.append("average_precision_degraded_vs_validation")
        if metrics["brier"] > validation_reference["brier"] + 0.02:
            reasons.append("brier_degraded_vs_validation")
    return {"status": "review_required" if reasons else "healthy",
            "retraining_recommended": bool(reasons), "reasons": reasons,
            "manual_review": bool(reasons),
            "automatic_training_permitted": False}
