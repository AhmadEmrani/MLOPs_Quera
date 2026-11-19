"""Validation metrics and recall-first candidate quality gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from promiseops.metrics import cohort_hash, evaluate


# Diagnostic tolerances only. These do NOT veto a candidate that satisfies
# the primary recall@10% gate. The model-selection code evaluates AP/Brier as
# tie-breakers on a common validation cohort.
MAX_AP_DROP = 0.02
MAX_BRIER_WORSENING = 0.005
_EPSILON = 1e-12


@dataclass(frozen=True)
class CandidateEvaluation:
    """Metrics and cohort identity for one candidate on one validation set."""

    model_name: str
    order_ids: tuple[str, ...]
    cohort_hash: str
    metrics: dict[str, float | int]


def evaluate_candidates(
    scores_by_name: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    order_ids: Sequence[str],
) -> dict[str, CandidateEvaluation]:
    """Evaluate each score vector against the same labels and order IDs."""
    if not scores_by_name:
        raise ValueError("at least one candidate score vector is required")
    if len(order_ids) == 0:
        raise ValueError("validation cohort must not be empty")
    if len(labels) != len(order_ids):
        raise ValueError("validation labels and order IDs must have equal length")

    ids = tuple(str(order_id) for order_id in order_ids)
    identity = cohort_hash(ids)
    results: dict[str, CandidateEvaluation] = {}
    for model_name, scores in scores_by_name.items():
        if not model_name:
            raise ValueError("candidate names must be non-empty")
        values = list(scores)
        if len(values) != len(ids):
            raise ValueError(f"candidate {model_name!r} score count does not match validation cohort")
        results[model_name] = CandidateEvaluation(
            model_name=model_name,
            order_ids=ids,
            cohort_hash=identity,
            metrics=evaluate(values, labels, ids),
        )
    return results


def compare_candidate(
    candidate: CandidateEvaluation,
    reference: CandidateEvaluation,
) -> dict[str, Any]:
    """Apply a recall-first gate against a baseline or incumbent.

    Recall@10% is the only metric that can veto a candidate here. AP drops
    greater than 0.02 and Brier worsening greater than 0.005 are recorded as
    warnings, not rejections. Ranking against champion/challenger remains in
    model.py, comparing all models on the *same* validation cohort using
    recall first, average precision second and Brier third.

    A passing gate does NOT by itself authorize setting an MLflow alias,
    deploying a model, or promoting a champion.
    """
    if set(candidate.order_ids) != set(reference.order_ids):
        raise ValueError("candidate and reference must use identical validation order IDs")
    if (
        candidate.cohort_hash != reference.cohort_hash
        or cohort_hash(candidate.order_ids) != candidate.cohort_hash
        or cohort_hash(reference.order_ids) != reference.cohort_hash
    ):
        raise ValueError("candidate and reference cohort hashes do not match")

    candidate_metrics = candidate.metrics
    reference_metrics = reference.metrics
    required = {"recall_at_10pct", "average_precision", "brier"}
    if not required.issubset(candidate_metrics) or not required.issubset(reference_metrics):
        raise ValueError("candidate and reference must include recall, average precision, and Brier")
    if any(
        not math.isfinite(float(metrics[name]))
        for metrics in (candidate_metrics, reference_metrics)
        for name in required
    ):
        raise ValueError("candidate and reference metrics must be finite")

    recall_change = (
        float(candidate_metrics["recall_at_10pct"])
        - float(reference_metrics["recall_at_10pct"])
    )
    ap_change = (
        float(candidate_metrics["average_precision"])
        - float(reference_metrics["average_precision"])
    )
    brier_change = (
        float(candidate_metrics["brier"])
        - float(reference_metrics["brier"])
    )

    # Hard gate: never accept lower recall@10%, allowing only negligible
    # floating-point noise in otherwise equal metrics.
    reasons: list[str] = []
    if recall_change < -_EPSILON:
        reasons.append("recall_at_10pct_decreased")

    # AP and Brier remain visible for review/calibration work, but they must
    # not silently take priority over the project's primary recall metric.
    warnings: list[str] = []
    if ap_change < -MAX_AP_DROP - _EPSILON:
        warnings.append("average_precision_dropped_by_more_than_0.02")
    if brier_change > MAX_BRIER_WORSENING + _EPSILON:
        warnings.append("brier_worsened_by_more_than_0.005")

    return {
        "promote": not reasons,  # Existing model.py interface: gate passed.
        "reasons": reasons,      # Hard failures only.
        "warnings": warnings,   # Diagnostic issues, never automatic vetoes.
        "gate_policy": "recall_at_10pct_non_decreasing",
        "comparison_deltas": {
            "recall_at_10pct": recall_change,
            "average_precision": ap_change,
            "brier": brier_change,
        },
        "candidate": candidate.model_name,
        "reference": reference.model_name,
        "cohort_hash": candidate.cohort_hash,
        "cohort_size": len(candidate.order_ids),
        "candidate_metrics": dict(candidate_metrics),
        "reference_metrics": dict(reference_metrics),
    }
