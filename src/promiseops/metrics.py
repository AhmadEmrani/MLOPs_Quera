"""Deterministic business metrics shared by training, batch scoring and monitoring.

Every definition here is taken from ``docs/CONTRACTS.md`` so that training, the batch
priority list and delayed-label monitoring rank orders identically:

* ``late = local_date(delivered_at) > promised_date`` in ``America/Sao_Paulo``.
* ``K = max(1, ceil(0.1 * N))``.
* Ranking is risk descending, ties broken by ascending ``order_id``.
* ``recall@10%`` = late orders in top K / all late orders.
* ``precision@10%`` = late orders in top K / K.
* PR-AUC uses average precision; Brier is the mean squared probability error.
* Numeric drift uses PSI, categorical drift uses total variation distance, with
  ``0.2`` as the review threshold.

This module is intentionally dependency-free (standard library only). The serving
image installs ``requirements.lock``, which does not contain scikit-learn/MLflow, so
metric code must stay importable there. Do not add third-party imports.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from datetime import date, datetime
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

# Matches ``promiseops.contracts.LOCAL_TZ``; duplicated so this module has no imports
# outside the standard library.
LOCAL_TZ = ZoneInfo("America/Sao_Paulo")

#: Share of the cohort selected for human review.
REVIEW_FRACTION = 0.1

#: Recall of a random 10% review list; the contract's retraining floor.
RANDOM_REVIEW_RECALL = 0.1

#: PSI / TVD value at which monitoring asks for a human review.
DRIFT_REVIEW_THRESHOLD = 0.2


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
def is_late(delivered_at: datetime, promised_date: date) -> bool:
    """Return the contract's late label for one delivered order.

    The comparison is between *local calendar dates* in ``America/Sao_Paulo``. A
    delivery later on the promised day is on time; only the next local date is late.
    """
    if delivered_at.tzinfo is None:
        raise ValueError("delivered_at must be timezone-aware")
    return delivered_at.astimezone(LOCAL_TZ).date() > promised_date


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------
def cutoff(n: int) -> int:
    """Return ``K = max(1, ceil(0.1 * N))`` using integer arithmetic.

    ``math.ceil(0.1 * 30)`` is ``4`` because of binary floating point; the contract
    means exactly one tenth rounded up, so compute it on integers.
    """
    if n < 1:
        raise ValueError("cohort size must be at least 1")
    return max(1, (n + 9) // 10)


def _check_scores(scores: Sequence[float]) -> list[float]:
    values = [float(value) for value in scores]
    if not values:
        raise ValueError("cohort is empty")
    for value in values:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("risk probabilities must be finite and within [0, 1]")
    return values


def _check_labels(labels: Sequence[int], n: int) -> list[int]:
    values = [int(value) for value in labels]
    if len(values) != n:
        raise ValueError("scores and labels must have equal length")
    if any(value not in (0, 1) for value in values):
        raise ValueError("labels must be 0 or 1")
    return values


def _check_ids(order_ids: Sequence[str], n: int) -> list[str]:
    values = [str(value) for value in order_ids]
    if len(values) != n:
        raise ValueError("scores and order_ids must have equal length")
    if len(set(values)) != n:
        raise ValueError("order_ids must be unique within a cohort")
    return values


def _rank(scores: Sequence[float], order_ids: Sequence[str]) -> list[int]:
    """Indices sorted by risk descending, ties by ascending order ID."""
    return sorted(range(len(scores)), key=lambda index: (-scores[index], order_ids[index]))


def top_k(scores: Sequence[float], order_ids: Sequence[str], k: int | None = None) -> list[tuple[str, float]]:
    """Return the reviewed priority list as ``(order_id, risk)`` pairs."""
    values = _check_scores(scores)
    ids = _check_ids(order_ids, len(values))
    size = cutoff(len(values)) if k is None else k
    if size < 1:
        raise ValueError("k must be at least 1")
    return [(ids[index], values[index]) for index in _rank(values, ids)[:size]]


# --------------------------------------------------------------------------------------
# Decision metrics
# --------------------------------------------------------------------------------------
def recall_at_k(scores: Sequence[float], labels: Sequence[int], order_ids: Sequence[str]) -> float:
    """Share of all late orders that appear in the top-``K`` review list."""
    values = _check_scores(scores)
    truth = _check_labels(labels, len(values))
    ids = _check_ids(order_ids, len(values))
    late_total = sum(truth)
    if late_total == 0:
        raise ValueError("cohort has no late orders; recall is undefined")
    selected = _rank(values, ids)[: cutoff(len(values))]
    return sum(truth[index] for index in selected) / late_total


def precision_at_k(scores: Sequence[float], labels: Sequence[int], order_ids: Sequence[str]) -> float:
    """Share of the top-``K`` review list that is actually late."""
    values = _check_scores(scores)
    truth = _check_labels(labels, len(values))
    ids = _check_ids(order_ids, len(values))
    size = cutoff(len(values))
    selected = _rank(values, ids)[:size]
    return sum(truth[index] for index in selected) / size


def average_precision(scores: Sequence[float], labels: Sequence[int], order_ids: Sequence[str]) -> float:
    """Average precision (the contract's PR-AUC)."""
    values = _check_scores(scores)
    truth = _check_labels(labels, len(values))
    ids = _check_ids(order_ids, len(values))
    late_total = sum(truth)
    if late_total == 0:
        raise ValueError("cohort has no late orders; average precision is undefined")
    hits = 0
    total = 0.0
    for position, index in enumerate(_rank(values, ids), start=1):
        if truth[index]:
            hits += 1
            total += hits / position
    return total / late_total


def brier(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between risk probability and the 0/1 outcome."""
    values = _check_scores(scores)
    truth = _check_labels(labels, len(values))
    return sum((probability - outcome) ** 2 for probability, outcome in zip(values, truth)) / len(values)


def prevalence(labels: Sequence[int]) -> float:
    """Share of late orders in the cohort."""
    truth = [int(value) for value in labels]
    if not truth:
        raise ValueError("cohort is empty")
    if any(value not in (0, 1) for value in truth):
        raise ValueError("labels must be 0 or 1")
    return sum(truth) / len(truth)


def constant_scores(labels: Sequence[int]) -> list[float]:
    """Constant train-prevalence baseline used as the first candidate's reference."""
    return [prevalence(labels)] * len(labels)


def evaluate(scores: Sequence[float], labels: Sequence[int], order_ids: Sequence[str]) -> dict[str, float | int]:
    """Return the full metric set reported for a cohort, including N and prevalence."""
    values = _check_scores(scores)
    truth = _check_labels(labels, len(values))
    ids = _check_ids(order_ids, len(values))
    return {
        "n": len(values),
        "late_count": sum(truth),
        "prevalence": prevalence(truth),
        "k": cutoff(len(values)),
        "recall_at_10pct": recall_at_k(values, truth, ids),
        "precision_at_10pct": precision_at_k(values, truth, ids),
        "average_precision": average_precision(values, truth, ids),
        "brier": brier(values, truth),
    }


# --------------------------------------------------------------------------------------
# Cohort identity
# --------------------------------------------------------------------------------------
def cohort_hash(order_ids: Iterable[str]) -> str:
    """Stable hash of a cohort's order IDs, used to prove identical evaluation sets."""
    ids = [str(value) for value in order_ids]
    if not ids:
        raise ValueError("cohort is empty")
    if len(set(ids)) != len(ids):
        raise ValueError("order_ids must be unique within a cohort")
    payload = json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------------------
def _quantile_edges(reference: Sequence[float], bins: int) -> list[float]:
    """Interior cut points of ``bins`` equal-frequency buckets of the reference."""
    ordered = sorted(reference)
    last = len(ordered) - 1
    cuts = []
    for step in range(1, bins):
        position = step / bins * last
        lower = math.floor(position)
        upper = math.ceil(position)
        cuts.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    return sorted(set(cuts))


def _bucket_proportions(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bisect.bisect_left(edges, value)] += 1
    return [count / len(values) for count in counts]


def psi(reference: Sequence[float], current: Sequence[float], *, bins: int = 10, epsilon: float = 1e-6) -> float:
    """Population stability index between a reference and a current numeric sample."""
    expected_values = [float(value) for value in reference]
    actual_values = [float(value) for value in current]
    if not expected_values or not actual_values:
        raise ValueError("both samples are required")
    edges = _quantile_edges(expected_values, bins)
    expected = _bucket_proportions(expected_values, edges)
    actual = _bucket_proportions(actual_values, edges)
    total = 0.0
    for expected_share, actual_share in zip(expected, actual):
        expected_share = max(expected_share, epsilon)
        actual_share = max(actual_share, epsilon)
        total += (actual_share - expected_share) * math.log(actual_share / expected_share)
    return total


def tvd(reference: Iterable[str], current: Iterable[str]) -> float:
    """Total variation distance between two categorical samples, in ``[0, 1]``."""
    expected = [str(value) for value in reference]
    actual = [str(value) for value in current]
    if not expected or not actual:
        raise ValueError("both samples are required")
    expected_counts: dict[str, int] = {}
    actual_counts: dict[str, int] = {}
    for value in expected:
        expected_counts[value] = expected_counts.get(value, 0) + 1
    for value in actual:
        actual_counts[value] = actual_counts.get(value, 0) + 1
    keys = set(expected_counts) | set(actual_counts)
    distance = sum(
        abs(expected_counts.get(key, 0) / len(expected) - actual_counts.get(key, 0) / len(actual)) for key in keys
    )
    return 0.5 * distance


def needs_review(score: float) -> bool:
    """Whether a PSI/TVD value crosses the contract's review threshold."""
    return score >= DRIFT_REVIEW_THRESHOLD
