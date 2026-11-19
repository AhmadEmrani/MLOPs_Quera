"""Training orchestration for PromiseOps candidate models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sklearn.base import BaseEstimator

from promiseops.data import build_features, training_snapshot, training_source_lineage
from promiseops.metrics import cohort_hash
from promiseops.model_pipeline.bundle import artifact_sha256, save_bundle
from promiseops.model_pipeline.candidates import (
    build_candidates,
    fit_candidates,
    positive_class_probability,
    score_candidates,
)
from promiseops.model_pipeline.evaluation import (
    CandidateEvaluation,
    compare_candidate as apply_quality_gates,
    evaluate_candidates,
)
from promiseops.model_pipeline.split import TemporalSplit, select_temporal_split
from promiseops.model_pipeline.tracking import (
    TrackingResult,
    finalize_model_registry,
    load_registry_reference,
    log_run,
    prepare_tracking,
)
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
    selected_model_name: str | None
    selected_model: BaseEstimator | None
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
    statement = """
        SELECT orders.order_id
        FROM orders
        JOIN outcomes ON outcomes.order_id = orders.order_id
        WHERE orders.approved_at <= :validation_end
          AND outcomes.available_at <= :as_of
          AND orders.batch_available_at <= :as_of
        ORDER BY orders.approved_at, orders.order_id
        """
    if hasattr(connection, "exec_driver_sql"):
        result = connection.execute(
            text(statement), {"validation_end": validation_end, "as_of": as_of}
        )
        return [str(order_id) for order_id in result.scalars().all()]

    with connection.cursor() as cursor:
        cursor.execute(
            statement.replace(":validation_end", "%(validation_end)s")
            .replace(":as_of", "%(as_of)s"),
            {"validation_end": validation_end, "as_of": as_of},
        )
        return [str(row[0]) for row in cursor.fetchall()]


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


def _write_json(path: Path, content: dict[str, Any]) -> str:
    """Write a deterministic manifest and return its file checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return artifact_sha256(path)


def _cohort_stats(labels: tuple[int, ...], order_ids: tuple[str, ...]) -> dict[str, Any]:
    n = len(labels)
    late = sum(int(label) for label in labels)
    return {
        "n": n,
        "late": late,
        "prevalence": late / n,
        "cohort_hash": cohort_hash(order_ids),
    }


def _feature_contract() -> dict[str, Any]:
    schema = LabeledFeatureRow.model_json_schema()
    properties = schema["properties"]
    return {
        "schema_version": 1,
        "feature_names": list(FEATURE_COLUMNS),
        "properties": {name: properties[name] for name in FEATURE_COLUMNS},
        "required": [name for name in FEATURE_COLUMNS if name in schema.get("required", [])],
        "$defs": schema.get("$defs", {}),
        "prediction_time": "approved_at",
    }


def _cohort_metrics(split: TemporalSplit, source_lineage: dict[str, Any]) -> dict[str, float]:
    train = _cohort_stats(split.train_labels, split.train_order_ids)
    validation = _cohort_stats(split.validation_labels, split.validation_order_ids)
    return {
        "train_n": float(train["n"]),
        "train_late": float(train["late"]),
        "train_prevalence": train["prevalence"],
        "validation_n": float(validation["n"]),
        "validation_late": float(validation["late"]),
        "validation_prevalence": validation["prevalence"],
        "customer_overlap_n": float(source_lineage["customer_overlap_n"]),
        "order_batch_after_approval_n": float(source_lineage["order_batch_after_approval_n"]),
    }


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
) -> str | None:
    # Include future candidate names without silently excluding them from selection.
    passing = [
        name
        for name in evaluations
        if name != _BASELINE_NAME and decisions[name]["promote"]
    ]
    if not passing:
        # The prevalence baseline is a benchmark, never a deployable challenger.
        # If all learned models fail the gates, keep the existing alias unchanged.
        return None
    return max(
        passing,
        key=lambda name: (
            evaluations[name].metrics["recall_at_10pct"],
            evaluations[name].metrics["average_precision"],
            -evaluations[name].metrics["brier"],
            name == "logistic_regression",
            name,
        ),
    )


def _metric_preference(evaluation: CandidateEvaluation) -> tuple[float, float, float]:
    """Rank only on the SAME cohort, following the project's metric priority."""
    metrics = evaluation.metrics
    return (
        float(metrics["recall_at_10pct"]),
        float(metrics["average_precision"]),
        -float(metrics["brier"]),
    )


def _strictly_better_on_same_cohort(
    candidate: CandidateEvaluation, reference: CandidateEvaluation
) -> bool:
    """Prefer recall, then AP, then lower Brier; exact ties never move aliases."""
    if candidate.cohort_hash != reference.cohort_hash:
        raise ValueError("comparison models must use the identical validation cohort")
    for candidate_value, reference_value in zip(
        _metric_preference(candidate), _metric_preference(reference)
    ):
        if candidate_value > reference_value + 1e-12:
            return True
        if candidate_value < reference_value - 1e-12:
            return False
    return False


def train_candidate(
    connection,
    *,
    train_end: datetime | str,
    validation_end: datetime | str,
    as_of: datetime | str,
    source_commit: str | None = None,
    champion_model: BaseEstimator | None = None,
    registered_model_name: str | None = None,
    artifacts_dir: str | Path | None = None,
    min_train_size: int = 1000,
    min_validation_size: int = 200,
    allow_local_tracking: bool = False,
) -> TrainingResult:
    """Train, compare, package, and track all three candidate approaches.

    Training and validation cohorts are selected from the shared labeled feature
    builder. The later test period is intentionally outside this function.
    The source snapshot is identified by ``PROMISEOPS_DATASET_ID`` when set;
    otherwise, a deterministic fingerprint of the selected labeled rows is used.
    Local JSON/SQLite tracking requires explicit allow_local_tracking=True;
    the worker uses the strict default.
    """
    train_cutoff = _instant(train_end, field="train_end")
    validation_cutoff = _instant(validation_end, field="validation_end")
    evaluation_time = _instant(as_of, field="as_of")
    if not train_cutoff < validation_cutoff <= evaluation_time:
        raise ValueError("cutoffs must satisfy train_end < validation_end <= as_of")

    commit = source_commit or os.environ.get("CI_COMMIT_SHA")
    if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("source_commit or CI_COMMIT_SHA must be a 40-character lowercase Git SHA")
    # Fail before fetching data or fitting models if tracking is misconfigured.
    # Server connectivity and artifact permissions are checked by actual logging.
    prepare_tracking(allow_local_tracking=allow_local_tracking)
    registry_name = registered_model_name or os.environ.get(
        "PROMISEOPS_REGISTERED_MODEL_NAME", "promiseops-late-delivery"
    )
    dataset_id = os.environ.get("PROMISEOPS_DATASET_ID")

    # All source reads (cohort, features, receipt inventory and leakage audit)
    # use the same repeatable-read snapshot. init_db commits only before it.
    with training_snapshot(connection):
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
            ensure_schema=False,
        )
        validated_rows = validate_feature_rows(feature_rows, labeled=True)
        labeled_frame = pd.DataFrame([row.model_dump() for row in validated_rows])
        split = select_temporal_split(
            labeled_frame,
            train_end=train_cutoff,
            validation_end=validation_cutoff,
            as_of=evaluation_time,
            min_train_size=min_train_size,
            min_validation_size=min_validation_size,
        )
        source_lineage = training_source_lineage(
            connection,
            as_of=evaluation_time,
            train_order_ids=split.train_order_ids,
            validation_order_ids=split.validation_order_ids,
        )

    # Explicit checks against accidental post-approval/outcome data in features.
    prohibited = {
        "late", "delivered_at", "outcome_available_at", "label_available_at",
        "order_id", "customer_id", "status", "review",
    }
    if prohibited.intersection(FEATURE_COLUMNS):
        raise ValueError("feature contract includes an identifier or an outcome column")

    candidates = build_candidates()
    fitted_candidates = fit_candidates(candidates, split.train_rows, split.train_labels)
    validation_scores = score_candidates(fitted_candidates, split.validation_rows)
    evaluations = evaluate_candidates(
        validation_scores,
        split.validation_labels,
        split.validation_order_ids,
    )

    # Read existing aliases BEFORE registering any new candidates. Missing
    # aliases are expected for a brand-new registry; broken models/API errors
    # fail closed instead of silently replacing a good challenger.
    incumbents: dict[str, CandidateEvaluation] = {}
    reference_versions: dict[str, str | None] = {
        "champion": None,
        "challenger": None,
    }
    for alias in ("champion", "challenger"):
        reference = load_registry_reference(
            registered_model_name=registry_name,
            alias=alias,
            allow_local_tracking=allow_local_tracking,
        )
        if reference is not None:
            reference_versions[alias] = reference.version
            reference_scores = positive_class_probability(
                reference.model, split.validation_rows
            )
            incumbents[alias] = evaluate_candidates(
                {alias: reference_scores},
                split.validation_labels,
                split.validation_order_ids,
            )[alias]
            del reference, reference_scores  # Do not retain loaded forest models.

    if champion_model is not None:
        deployed_scores = positive_class_probability(champion_model, split.validation_rows)
        incumbents["deployed_incumbent"] = evaluate_candidates(
            {"deployed_incumbent": deployed_scores},
            split.validation_labels,
            split.validation_order_ids,
        )["deployed_incumbent"]
        del deployed_scores

    # Apply existing quality gates against the strongest incumbent reference,
    # or the prevalence baseline when no incumbent exists. Additionally demand
    # strict ranking improvement over EACH existing incumbent; otherwise the
    # existing challenger remains in place even if a new model is locally best.
    gate_references = {
        key: value
        for key, value in incumbents.items()
        if key in {"champion", "deployed_incumbent"}
    }
    if not gate_references:
        gate_references = {"prevalence_baseline": evaluations[_BASELINE_NAME]}
    strongest_reference = max(gate_references.values(), key=_metric_preference)

    decisions: dict[str, dict[str, Any]] = {}
    for name, evaluation in evaluations.items():
        decision = dict(apply_quality_gates(evaluation, strongest_reference))
        outperforms = {
            alias: _strictly_better_on_same_cohort(evaluation, reference)
            for alias, reference in incumbents.items()
        }
        original_gate = bool(decision["promote"])
        decision["quality_gate_passed"] = original_gate
        decision["incumbent_comparisons"] = outperforms
        decision["incumbent_metrics"] = {
            alias: dict(reference.metrics) for alias, reference in incumbents.items()
        }
        decision["incumbent_versions"] = dict(reference_versions)
        decision["selection_policy"] = "same_cohort_recall_then_ap_then_brier"
        # A prevalence-only baseline is diagnostic: it may pass its own
        # comparison, but must never become a challenger, even on first run.
        decision["promote"] = (
            original_gate
            and name != _BASELINE_NAME
            and all(outperforms.values())
        )
        decision["selection_reason"] = (
            "baseline_reference_only" if name == _BASELINE_NAME
            else "quality_gate_failed" if not original_gate
            else "incumbent_not_outperformed" if not all(outperforms.values())
            else "eligible_for_challenger"
        )
        decisions[name] = decision
    selected_name = _select_candidate(evaluations, decisions)

    output_root = Path(artifacts_dir or os.environ.get("PROMISEOPS_ARTIFACTS", "/tmp/artifacts"))
    dataset_fingerprint = _dataset_identity(labeled_frame)
    dataset_id = dataset_id or f"sha256:{dataset_fingerprint}"
    schema_identity = _schema_identity()
    registered_model_tags = {
        "team": "g02",
        "task": "late_delivery_probability",
        "primary_metric": "recall_at_10pct",
        "review_capacity": "0.10",
        "timezone": "America/Sao_Paulo",
        "data_source": "olist",
        "deployment_policy": "manual_ci_approval",
    }
    tracking_results: dict[str, TrackingResult] = {}
    bundle_manifests: dict[str, dict[str, Any]] = {}

    output_root.mkdir(parents=True, exist_ok=True)
    feature_contract_path = output_root / "feature_contract.json"
    feature_contract_sha256 = _write_json(feature_contract_path, _feature_contract())

    train_stats = _cohort_stats(split.train_labels, split.train_order_ids)
    validation_stats = _cohort_stats(split.validation_labels, split.validation_order_ids)
    manifest_data = {
        "manifest_version": 1,
        "dataset_identity": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "schema_identity": schema_identity,
        "feature_contract_sha256": feature_contract_sha256,
        "source_commit": commit,
        "train_end": train_cutoff.isoformat(),
        "validation_end": validation_cutoff.isoformat(),
        "as_of": evaluation_time.isoformat(),
        "label_cutoff": train_cutoff.isoformat(),
        "cohort_policy": {
            "min_train_size": min_train_size,
            "min_validation_size": min_validation_size,
        },
        "train": train_stats,
        "validation": validation_stats,
        "excluded": {
            "n": len(split.excluded),
            "reasons": {
                str(key): int(value)
                for key, value in split.excluded["exclusion_reason"].value_counts().items()
            },
        },
        "source_lineage": source_lineage,
        "test_policy": "untouched_not_used_in_training_or_selection",
    }
    dataset_manifest_path = output_root / "dataset_manifest.json"
    dataset_manifest_sha256 = _write_json(dataset_manifest_path, manifest_data)
    cohort_metrics = _cohort_metrics(split, source_lineage)

    for name, fitted_model in fitted_candidates.items():
        evaluation = evaluations[name]
        gate_decision = decisions[name]
        bundle_dir = output_root / "candidates" / name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for auxiliary in (dataset_manifest_path, feature_contract_path):
            shutil.copy2(auxiliary, bundle_dir / auxiliary.name)
        manifest = save_bundle(
            fitted_model,
            bundle_dir,
            {
                "model_name": name,
                "source_commit": commit,
                "dataset_identity": dataset_id,
                "dataset_fingerprint": dataset_fingerprint,
                "schema_identity": schema_identity,
                "feature_contract_file": "feature_contract.json",
                "feature_contract_sha256": feature_contract_sha256,
                "dataset_manifest_file": "dataset_manifest.json",
                "dataset_manifest_sha256": dataset_manifest_sha256,
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
        selected = name == selected_name
        quality_passed = bool(gate_decision["quality_gate_passed"])
        version_tags = {
            "algorithm": name,
            "source_commit": commit,
            "train_end": train_cutoff.isoformat(),
            "validation_end": validation_cutoff.isoformat(),
            "as_of": evaluation_time.isoformat(),
            "label_cutoff": train_cutoff.isoformat(),
            "dataset_identity": dataset_id,
            "dataset_sha256": dataset_fingerprint,
            "schema_identity": schema_identity,
            "feature_contract_sha256": feature_contract_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "artifact_sha256": str(manifest["artifact_sha256"]),
            "train_cohort_hash": train_stats["cohort_hash"],
            "validation_cohort_hash": validation_stats["cohort_hash"],
            "quality_gate": (
                "reference_only" if name == _BASELINE_NAME
                else "passed" if quality_passed else "failed"
            ),
            "selection_status": "selected" if selected else "not_selected",
            "approval_status": "pending" if selected and quality_passed else "not_requested",
            "champion_version_at_evaluation": reference_versions["champion"] or "absent",
            "challenger_version_at_evaluation": reference_versions["challenger"] or "absent",
            "beats_champion": (
                str(decisions[name]["incumbent_comparisons"]["champion"]).lower()
                if "champion" in incumbents else "not_applicable"
            ),
            "beats_challenger": (
                str(decisions[name]["incumbent_comparisons"]["challenger"]).lower()
                if "challenger" in incumbents else "not_applicable"
            ),
            "beats_deployed_incumbent": (
                str(decisions[name]["incumbent_comparisons"]["deployed_incumbent"]).lower()
                if "deployed_incumbent" in incumbents else "not_applicable"
            ),
            "comparison_policy": "same_cohort_recall_then_ap_then_brier",
            "selection_reason": (
                "selected_for_review" if selected else gate_decision["selection_reason"]
            ),
        }
        tracking_results[name] = log_run(
            run_name=f"{name}-{commit[:12]}",
            params=params,
            metrics={**evaluation.metrics, **cohort_metrics},
            tags={
                **version_tags,
                "dataset_identity": dataset_id,
                "dataset_fingerprint": dataset_fingerprint,
                "schema_identity": schema_identity,
                "feature_contract_sha256": feature_contract_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "model_version": manifest["model_version"],
                "artifact_sha256": manifest["artifact_sha256"],
                "train_end": train_cutoff.isoformat(),
                "validation_end": validation_cutoff.isoformat(),
                "as_of": evaluation_time.isoformat(),
                "label_cutoff": train_cutoff.isoformat(),
                "gate_decision": json.dumps(gate_decision, sort_keys=True, default=str),
                "software_versions": json.dumps(manifest["software_versions"], sort_keys=True),
                "model_name": name,
                "source_commit": commit,
                "cohort_hash": evaluation.cohort_hash,
                "gate_passed": str(quality_passed).lower(),
                "selected": str(name == selected_name).lower(),
            },
            artifacts={
                f"candidates/{name}/model.joblib": bundle_dir / "model.joblib",
                f"candidates/{name}/release.json": bundle_dir / "release.json",
                f"candidates/{name}/dataset_manifest.json": bundle_dir / "dataset_manifest.json",
                f"candidates/{name}/feature_contract.json": bundle_dir / "feature_contract.json",
            },
            model=fitted_model,
            input_example=split.train_rows.loc[:, FEATURE_COLUMNS].head(5),
            model_artifact_path=name,
            registered_model_name=registry_name,
            model_version_tags=version_tags,
            allow_local_tracking=allow_local_tracking,
        )

    # Mutate the challenger alias only AFTER every run, artifact and version tag
    # has been stored. Never create or move the champion alias from training.
    selected_version = None
    if selected_name is not None:
        selected_result = tracking_results[selected_name]
        if selected_result.backend == "mlflow":
            selected_version = selected_result.registered_model_version
            if not selected_version:
                raise RuntimeError("selected model is missing a registered MLflow version")
    finalize_model_registry(
        registered_model_name=registry_name,
        registered_model_tags=registered_model_tags,
        challenger_version=selected_version,
        expected_challenger_version=reference_versions["challenger"],
        verify_challenger_version=selected_version is not None,
        allow_local_tracking=allow_local_tracking,
    )

    return TrainingResult(
        candidates=fitted_candidates,
        evaluations=evaluations,
        decisions=decisions,
        tracking=tracking_results,
        bundles=bundle_manifests,
        selected_model_name=selected_name,
        selected_model=(fitted_candidates[selected_name] if selected_name is not None else None),
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