"""Training orchestration for PromiseOps candidate models."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sklearn.base import BaseEstimator

from promiseops.data import build_features
from promiseops.metrics import cohort_hash
from promiseops.model_pipeline.bundle import save_bundle
from promiseops.model_pipeline.candidates import build_candidates, fit_candidates, score_candidates
from promiseops.model_pipeline.evaluation import (
    CandidateEvaluation,
    compare_candidate as apply_quality_gates,
    evaluate_candidates,
)
from promiseops.model_pipeline.split import TemporalSplit, select_temporal_split
from promiseops.model_pipeline.tracking import TrackingResult, log_run
from promiseops.model_pipeline.validation import FEATURE_COLUMNS, LabeledFeatureRow, validate_feature_rows


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_BASELINE_NAME = "prevalence_baseline"


@dataclass(frozen=True)
class TrainingResult:
    """Fitted candidates, validation results, gates, and selected candidate."""

    candidates: dict[str, BaseEstimator]
    evaluations: dict[str, CandidateEvaluation]
    decisions: dict[str, dict[str, Any]]
    tracking: dict[str, TrackingResult]
    bundles: dict[str, dict[str, Any]]
    selected_model_name: str
    selected_model: BaseEstimator
    split: TemporalSplit


def _instant(value: datetime | str, *, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _cohort_order_ids(connection, *, validation_end: datetime, as_of: datetime) -> list[str]:
    statement = text(
        """
        SELECT orders.order_id
        FROM orders
        JOIN outcomes ON outcomes.order_id = orders.order_id
        WHERE orders.approved_at <= :validation_end
          AND outcomes.available_at <= :as_of
        ORDER BY orders.approved_at, orders.order_id
        """
    )
    result = connection.execute(statement, {"validation_end": validation_end, "as_of": as_of})
    return [str(order_id) for order_id in result.scalars().all()]


def _schema_identity() -> str:
    schema = {
        "features": list(FEATURE_COLUMNS),
        "target": "late",
        "approval_time": "approved_at",
        "label_availability_time": "label_available_at",
    }
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dataset_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("order_id", kind="mergesort")
    payload = ordered.to_json(orient="records", date_format="iso", double_precision=15)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_params(model: BaseEstimator) -> dict[str, Any]:
    params = model.get_params(deep=True)
    return {
        name: value
        for name, value in params.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _select_candidate(
    evaluations: dict[str, CandidateEvaluation],
    decisions: dict[str, dict[str, Any]],
) -> str:
    passing = [
        name
        for name in ("logistic_regression", "random_forest")
        if decisions[name]["promote"]
    ]
    if not passing:
        return _BASELINE_NAME
    return max(
        passing,
        key=lambda name: (
            evaluations[name].metrics["recall_at_10pct"],
            evaluations[name].metrics["average_precision"],
            -evaluations[name].metrics["brier"],
            name == "logistic_regression",
        ),
    )


def train_candidate(
    connection,
    *,
    train_end: datetime | str,
    validation_end: datetime | str,
    as_of: datetime | str,
    source_commit: str | None = None,
    registered_model_name: str | None = None,
    artifacts_dir: str | Path | None = None,
    min_train_size: int = 2,
    min_validation_size: int = 2,
) -> TrainingResult:
    """Train, compare, package, and track all three candidate approaches.

    Training and validation cohorts are selected from the shared labeled feature
    builder. The later test period is intentionally outside this function.
    The source snapshot is identified by ``PROMISEOPS_DATASET_ID`` when set;
    otherwise, a deterministic fingerprint of the selected labeled rows is used.
    """
    train_cutoff = _instant(train_end, field="train_end")
    validation_cutoff = _instant(validation_end, field="validation_end")
    evaluation_time = _instant(as_of, field="as_of")
    if not train_cutoff < validation_cutoff <= evaluation_time:
        raise ValueError("cutoffs must satisfy train_end < validation_end <= as_of")

    commit = source_commit or os.environ.get("CI_COMMIT_SHA")
    if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("source_commit or CI_COMMIT_SHA must be a 40-character lowercase Git SHA")
    dataset_id = os.environ.get("PROMISEOPS_DATASET_ID")

    order_ids = _cohort_order_ids(
        connection,
        validation_end=validation_cutoff,
        as_of=evaluation_time,
    )
    if not order_ids:
        raise ValueError("no labeled orders are available for the selected validation window")

    feature_rows = build_features(
        connection,
        order_ids,
        as_of=evaluation_time,
        labeled=True,
    )
    validated_rows = validate_feature_rows(feature_rows, labeled=True)
    labeled_frame = pd.DataFrame(
        [row.model_dump() for row in validated_rows]
    )
    split = select_temporal_split(
        labeled_frame,
        train_end=train_cutoff,
        validation_end=validation_cutoff,
        as_of=evaluation_time,
        min_train_size=min_train_size,
        min_validation_size=min_validation_size,
    )

    candidates = build_candidates()
    fitted_candidates = fit_candidates(candidates, split.train_rows, split.train_labels)
    validation_scores = score_candidates(fitted_candidates, split.validation_rows)
    evaluations = evaluate_candidates(
        validation_scores,
        split.validation_labels,
        split.validation_order_ids,
    )

    baseline_evaluation = evaluations[_BASELINE_NAME]
    decisions = {
        name: apply_quality_gates(evaluation, baseline_evaluation)
        for name, evaluation in evaluations.items()
    }
    selected_name = _select_candidate(evaluations, decisions)

    output_root = Path(artifacts_dir or os.environ.get("PROMISEOPS_ARTIFACTS", "/tmp/artifacts"))
    registry_name = registered_model_name or os.environ.get(
        "PROMISEOPS_REGISTERED_MODEL_NAME", "promiseops-late-delivery"
    )
    dataset_fingerprint = _dataset_identity(labeled_frame)
    dataset_id = dataset_id or f"sha256:{dataset_fingerprint}"
    tracking_results: dict[str, TrackingResult] = {}
    bundle_manifests: dict[str, dict[str, Any]] = {}

    for name, fitted_model in fitted_candidates.items():
        evaluation = evaluations[name]
        gate_decision = decisions[name]
        bundle_dir = output_root / "candidates" / name
        manifest = save_bundle(
            fitted_model,
            bundle_dir,
            {
                "model_name": name,
                "source_commit": commit,
                "dataset_identity": dataset_id,
                "dataset_fingerprint": dataset_fingerprint,
                "schema_identity": _schema_identity(),
                "train_end": train_cutoff,
                "validation_end": validation_cutoff,
                "as_of": evaluation_time,
                "label_cutoff": train_cutoff,
                "cohort_hash": evaluation.cohort_hash,
                "metrics": evaluation.metrics,
                "gate_decision": gate_decision,
                "feature_names": list(FEATURE_COLUMNS),
            },
        )
        bundle_manifests[name] = manifest

        params = _model_params(fitted_model)
        params.update(
            {
                "model_name": name,
                "train_end": train_cutoff.isoformat(),
                "validation_end": validation_cutoff.isoformat(),
                "as_of": evaluation_time.isoformat(),
            }
        )
        tracking_results[name] = log_run(
            run_name=f"{name}-{commit[:12]}",
            params=params,
            metrics=evaluation.metrics,
            tags={
                "model_name": name,
                "source_commit": commit,
                "cohort_hash": evaluation.cohort_hash,
                "gate_passed": str(gate_decision["promote"]).lower(),
                "selected": str(name == selected_name).lower(),
            },
            artifacts={
                f"candidates/{name}/model.joblib": bundle_dir / "model.joblib",
                f"candidates/{name}/release.json": bundle_dir / "release.json",
            },
            model=fitted_model,
            model_artifact_path=f"models/{name}",
            registered_model_name=registry_name,
        )

    return TrainingResult(
        candidates=fitted_candidates,
        evaluations=evaluations,
        decisions=decisions,
        tracking=tracking_results,
        bundles=bundle_manifests,
        selected_model_name=selected_name,
        selected_model=fitted_candidates[selected_name],
        split=split,
    )


def compare_candidate(candidate, champion, cohort):
    """Apply public gates after verifying the caller's validation cohort."""
    if isinstance(cohort, str):
        expected_hash = cohort
    else:
        expected_hash = cohort_hash(cohort)
    if candidate.cohort_hash != expected_hash or champion.cohort_hash != expected_hash:
        raise ValueError("candidate, champion, and supplied cohort must have identical IDs")
    return apply_quality_gates(candidate, champion)
