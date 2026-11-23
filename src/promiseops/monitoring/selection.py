"""Deterministic label-review priority, not artificial labels or train-set mutation."""
from __future__ import annotations
from hashlib import sha256
from hmac import new as hmac_new
from math import isfinite
from typing import Any
from promiseops.monitoring.contracts import Prediction, instant, unique_predictions


def _distance(a: dict[str, Any], b: dict[str, Any], scales: dict[str, tuple[float, float]]) -> float:
    score = 0.0
    count = 0
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if x is None or y is None:
            d = 0.0 if x is y else 1.0
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)) and not isinstance(y, bool):
            if not isfinite(float(x)) or not isfinite(float(y)):
                raise ValueError("invalid numeric candidate feature")
            lo, hi = scales[key]
            d = abs(float(x) - float(y)) / max(hi - lo, 1e-9)
        else:
            d = float(x != y)
        score += d
        count += 1
    return score / count if count else 0.0


def select_for_review(rows: list[Prediction], *, as_of: object, limit: int = 20,
                      salt: str = "") -> dict:
    """Select unlabeled uncertain and diverse points, exposing *only* hashes.

    Hashes are safe from accidental direct-ID leakage but NOT anonymity against
    dictionary attack. Never export without a team-secret salt in a real CI job.
    """
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be in 1..1000")
    if not salt:
        raise ValueError("a private HMAC-like salt is required for exportable identities")
    eligible = [p for p in unique_predictions(rows, instant(as_of)) if p.label is None]
    if not eligible:
        return {"n_unlabeled": 0, "selected_count": 0, "review": []}
    # Materialize scalar ranges once. Missing values are handled distinctly.
    scales: dict[str, tuple[float, float]] = {}
    for key in sorted(set.intersection(*(set(p.features) for p in eligible))):
        nums = [float(p.features[key]) for p in eligible
                if isinstance(p.features[key], (int, float)) and not isinstance(p.features[key], bool)]
        if nums:
            scales[key] = (min(nums), max(nums))
    # First choose most uncertain, thereafter encourage feature diversity.
    selected: list[Prediction] = []
    remaining = {p.order_id: p for p in eligible}
    while remaining and len(selected) < limit:
        def criterion(p: Prediction) -> tuple[float, float, str]:
            uncertainty = 1.0 - 2.0 * abs(p.risk_probability - .5)
            diversity = min((_distance(p.features, q.features, scales) for q in selected), default=1.0)
            return (0.75 * uncertainty + 0.25 * diversity, uncertainty, p.order_id)
        best = sorted(remaining.values(), key=lambda p: (-criterion(p)[0], -criterion(p)[1], p.order_id))[0]
        selected.append(best)
        del remaining[best.order_id]
    return {"n_unlabeled": len(eligible), "selected_count": len(selected),
            "review": [{"order_hash": hmac_new(salt.encode(), p.order_id.encode(), sha256).hexdigest(),
                        "uncertainty": round(1 - 2 * abs(p.risk_probability - .5), 6)}
                       for p in selected],
            "note": "Selection prioritizes review only. No labels are requested or fabricated."}
