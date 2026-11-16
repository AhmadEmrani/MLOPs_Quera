"""Strict MLflow tracking with an explicitly opted-in local JSON backend."""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class TrackingResult:
    """Identifiers returned by MLflow or the local development logger."""

    run_id: str
    model_uri: str | None
    registered_model_version: str | None
    backend: str


def _json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _validated_metrics(metrics: Mapping[str, int | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        numeric = float(value)
        if not name or not math.isfinite(numeric):
            raise ValueError("metric names must be non-empty and values must be finite")
        result[name] = numeric
    return result


def _artifact_destination(name: str) -> PurePosixPath:
    destination = PurePosixPath(name)
    if not name or destination.is_absolute() or ".." in destination.parts:
        raise ValueError(f"invalid artifact destination: {name!r}")
    return destination


def _local_log_run(
    *,
    metrics: dict[str, float],
    params: Mapping[str, Any],
    artifacts: Mapping[str, str | Path],
    tags: Mapping[str, str],
    model: BaseEstimator | None,
    registered_model_name: str | None,
    local_tracking_dir: str | Path | None,
) -> TrackingResult:
    run_id = f"local-{uuid.uuid4().hex}"
    base_dir = Path(
        local_tracking_dir
        or Path(os.environ.get("PROMISEOPS_ARTIFACTS", ".promiseops-artifacts")) / "local-tracking"
    )
    run_dir = base_dir / run_id
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    logged_artifacts: list[str] = []
    for name, source_value in artifacts.items():
        source = Path(source_value)
        if not source.exists():
            raise FileNotFoundError(f"tracking artifact does not exist: {source}")
        destination = _artifact_destination(name)
        target = artifact_dir.joinpath(*destination.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        logged_artifacts.append(destination.as_posix())

    record = {
        "run_id": run_id,
        "backend": "local_json",
        "params": {str(key): _json_value(value) for key, value in params.items()},
        "metrics": metrics,
        "tags": {str(key): str(value) for key, value in tags.items()},
        "artifacts": logged_artifacts,
        "model_class": (
            f"{type(model).__module__}.{type(model).__qualname__}" if model is not None else None
        ),
        "requested_registered_model_name": registered_model_name,
        "registered_model_version": None,
    }
    record_path = run_dir / "run.json"
    with record_path.open("w", encoding="utf-8") as record_file:
        json.dump(record, record_file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        record_file.write("\n")
    return TrackingResult(
        run_id=run_id,
        model_uri=None,
        registered_model_version=None,
        backend="local_json",
    )


def prepare_tracking(
    *,
    tracking_uri: str | None = None,
    allow_local_tracking: bool = False,
):
    """Validate configuration and dependencies, without claiming server reachability.

    Normal training requires an explicit HTTP(S) MLflow endpoint. Local/test
    callers can opt in: with no URI they use JSON; with a URI they use MLflow
    (including SQLite/file stores). An explicit URI never falls back to JSON.
    Imports remain lazy so serving does not need MLflow installed.
    """
    raw_uri = tracking_uri if tracking_uri is not None else os.environ.get("MLFLOW_TRACKING_URI")
    uri = raw_uri.strip() if raw_uri else ""
    if not uri:
        if allow_local_tracking:
            return None
        raise RuntimeError("MLFLOW_TRACKING_URI is required for training")

    if not allow_local_tracking:
        try:
            parsed = urlsplit(uri)
            valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
            parsed.port  # Validate malformed ports without printing credentials.
        except ValueError:
            valid = False
        if not valid:
            raise RuntimeError("Training requires an HTTP(S) MLFLOW_TRACKING_URI")

    try:
        mlflow = importlib.import_module("mlflow")
        importlib.import_module("mlflow.sklearn")
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking dependencies are missing or broken; "
            "install requirements-tracking.lock in the worker image"
        ) from error
    mlflow.set_tracking_uri(uri)
    return mlflow


def log_run(
    *,
    metrics: Mapping[str, int | float],
    params: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    registered_model_name: str | None = None,
    model: BaseEstimator | None = None,
    input_example: Any | None = None,
    model_artifact_path: str = "model",
    tags: Mapping[str, str] | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    local_tracking_dir: str | Path | None = None,
    allow_local_tracking: bool = False,
) -> TrackingResult:
    """Log one candidate run, artifacts, and optionally register its model.

    MLflow is imported only when this function is called, keeping this module
    safe to import from the serving application. Registration creates a model
    version only; this function never assigns or changes an alias. Artifact
    mapping keys are destination paths relative to the run and must retain the
    source file's basename.
    """
    metric_values = _validated_metrics(metrics)
    parameter_values = dict(params or {})
    artifact_values = dict(artifacts or {})
    tag_values = dict(tags or {})
    if registered_model_name and model is None:
        raise ValueError("model is required when registered_model_name is set")
    if input_example is not None and model is None:
        raise ValueError("input_example requires a fitted model")
    if not model_artifact_path or _artifact_destination(model_artifact_path).as_posix() != model_artifact_path:
        raise ValueError("model_artifact_path must be a safe relative path")
    for name, source in artifact_values.items():
        destination = _artifact_destination(name)
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"tracking artifact does not exist: {source}")
        if destination.name != source_path.name:
            raise ValueError(f"artifact destination {name!r} must use source basename {source_path.name!r}")

    mlflow = prepare_tracking(
        tracking_uri=tracking_uri,
        allow_local_tracking=allow_local_tracking,
    )
    if mlflow is None:
        return _local_log_run(
            metrics=metric_values,
            params=parameter_values,
            artifacts=artifact_values,
            tags=tag_values,
            model=model,
            registered_model_name=registered_model_name,
            local_tracking_dir=local_tracking_dir,
        )

    mlflow_sklearn = importlib.import_module("mlflow.sklearn") if model is not None else None

    model_info = None
    with mlflow.start_run(run_name=run_name) as run:
        if parameter_values:
            mlflow.log_params({str(key): _json_value(value) for key, value in parameter_values.items()})
        if metric_values:
            mlflow.log_metrics(metric_values)
        if tag_values:
            mlflow.set_tags({str(key): str(value) for key, value in tag_values.items()})
        for name, source_value in artifact_values.items():
            destination = _artifact_destination(name)
            artifact_path = destination.parent.as_posix()
            if artifact_path == ".":
                artifact_path = None
            mlflow.log_artifact(str(source_value), artifact_path=artifact_path)
        if model is not None:
            log_options: dict[str, Any] = {}
            if input_example is not None:
                # MLflow's default sklearn pyfunc flavor calls predict(), not
                # predict_proba(). Keep the signature aligned with that behavior;
                # PromiseOps serving calls predict_proba() on the same pipeline.
                infer_signature = importlib.import_module("mlflow.models").infer_signature
                log_options["signature"] = infer_signature(
                    input_example, model.predict(input_example)
                )
                log_options["input_example"] = input_example
            model_info = mlflow_sklearn.log_model(
                sk_model=model,
                name=model_artifact_path,
                registered_model_name=registered_model_name,
                serialization_format="cloudpickle",
                **log_options,
            )

    return TrackingResult(
        run_id=run.info.run_id,
        model_uri=getattr(model_info, "model_uri", None),
        registered_model_version=(
            str(model_info.registered_model_version)
            if model_info is not None and getattr(model_info, "registered_model_version", None) is not None
            else None
        ),
        backend="mlflow",
    )