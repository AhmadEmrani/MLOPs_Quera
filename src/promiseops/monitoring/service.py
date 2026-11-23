"""Pure monitoring orchestration, safe to test without DB, MLflow or Kubernetes."""
from __future__ import annotations
from typing import Any
from promiseops.monitoring.contracts import Prediction, instant, iso
from promiseops.monitoring.decisions import decide
from promiseops.monitoring.drift import drift_report, reference_contract
from promiseops.monitoring.labels import evaluate_mature
from promiseops.monitoring.selection import select_for_review


def build_report(predictions: list[Prediction], released_outcomes: dict[str, tuple[int, object]] | None,
                 reference: dict, *, as_of: object,
                 validation_reference: dict | None = None,
                 review_salt: str | None = None, review_limit: int = 20,
                 min_window_orders: int = 100, min_mature_labels: int = 100) -> dict[str, Any]:
    """Calculate observational evidence; calling this never creates a training job.

    `released_outcomes` may only contain labels already proven mature by the
    repository at as_of. Passing future/unverified labels is a caller error.
    """
    t = instant(as_of)
    contract, _ = reference_contract(reference)
    if released_outcomes is not None:
        # External outcomes must carry proof of publication, not just a label.
        # Mapping: order_id -> (late: 0|1, publication_time: offset-aware datetime).
        amended: list[Prediction] = []
        for p in predictions:
            supplied = released_outcomes.get(p.order_id)
            if supplied is None:
                if p.label is not None:
                    raise ValueError("unverified label is present")
                amended.append(p)
                continue
            if not isinstance(supplied, (tuple, list)) or len(supplied) != 2:
                raise ValueError("released outcomes require label and publication time")
            label, published_at = supplied
            if type(label) is not int or label not in (0, 1):
                raise ValueError("invalid mature label")
            # Future labels cannot appear in this report, even if known today.
            is_released = instant(published_at) <= t
            amended.append(Prediction(order_id=p.order_id, model_version=p.model_version,
                    risk_probability=p.risk_probability, scored_as_of=p.scored_as_of,
                    feature_contract_sha256=p.feature_contract_sha256, features=p.features,
                    label=label if is_released else None, batch_id=p.batch_id,
                    label_available_at=instant(published_at) if is_released else None))
        predictions = amended
    elif any(p.label is not None and
             (p.label_available_at is None or instant(p.label_available_at) > t)
             for p in predictions):
        raise ValueError("future or unverified labels are present")
    models = {p.model_version for p in predictions}
    if len(models) > 1:
        raise ValueError("cannot mix model versions in one monitoring report")
    if any(p.feature_contract_sha256 != contract for p in predictions):
        raise ValueError("prediction features and baseline contract differ")
    drift = drift_report(predictions, reference, as_of=t, min_orders=min_window_orders)
    mature = evaluate_mature(predictions, as_of=t, min_labels=min_mature_labels)
    decision = decide(drift, mature, validation_reference)
    result = {"schema_version": 1, "as_of": iso(t),
              "model_version": next(iter(models), None), "feature_contract_sha256": contract,
              "drift": drift, "delayed_labels": mature, "decision": decision}
    if review_salt is not None:
        result["active_learning"] = select_for_review(predictions, as_of=t,
                                                       limit=review_limit, salt=review_salt)
    return result
