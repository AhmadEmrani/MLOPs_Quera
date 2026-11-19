"""Strict MLflow tracking with an explicitly opted-in local JSON backend."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class RegistryReference:
    """Immutable MLflow version selected by an alias at lookup time."""

    alias: str
    version: str
    model: BaseEstimator


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
    model_version_tags: Mapping[str, str],
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
        # Local JSON records intended registry metadata, but creates no MLflow
        # version or aliases. This is a development-only test backend.
        "intended_model_version_tags": {
            str(key): str(value) for key, value in model_version_tags.items()
        },
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


def _registry_alias_version(client, *, registered_model_name: str, alias: str) -> str | None:
    """Return the version, or None ONLY for a missing alias/model.

    Authentication failures, timeouts and other MLflow API errors must not
    look like a first training run; they propagate to fail the worker.
    """
    try:
        result = client.get_model_version_by_alias(registered_model_name, alias)
    except Exception as error:
        # Avoid a hard MLflow dependency for the local-only development mode.
        exceptions = importlib.import_module("mlflow.exceptions")
        if isinstance(error, exceptions.MlflowException):
            error_code = getattr(error, "error_code", None)
            if error_code == "RESOURCE_DOES_NOT_EXIST":
                return None
            # MLflow servers backed by the model registry can report a missing
            # alias as INVALID_PARAMETER_VALUE rather than RESOURCE_DOES_NOT_EXIST.
            # Match the *requested alias* and the exact missing-alias message;
            # other INVALID_PARAMETER_VALUE failures must remain fatal.
            if error_code == "INVALID_PARAMETER_VALUE" and re.fullmatch(
                rf"(?:INVALID_PARAMETER_VALUE:\s*)?Registered model alias "
                rf"['\"]?{re.escape(alias)}['\"]? not found\.?",
                str(error).strip(),
                flags=re.IGNORECASE,
            ):
                return None
        raise
    version = str(result.version)
    if not version.isdigit() or int(version) < 1:
        raise RuntimeError(f"MLflow alias {alias!r} has an invalid model version")
    return version


def load_registry_reference(
    *,
    registered_model_name: str,
    alias: str,
    tracking_uri: str | None = None,
    allow_local_tracking: bool = False,
) -> RegistryReference | None:
    """Load an optional incumbent from a pinned version, never a moving alias.

    A missing `champion`/`challenger` is expected on first use. An existing
    alias whose artifact cannot load is a hard failure, never a fresh start.
    """
    if not registered_model_name:
        raise ValueError("registered_model_name must not be empty")
    if alias not in {"champion", "challenger"}:
        raise ValueError("only champion and challenger aliases may be inspected")
    mlflow = prepare_tracking(
        tracking_uri=tracking_uri, allow_local_tracking=allow_local_tracking
    )
    if mlflow is None:
        return None  # Development JSON tracking has no Registry.

    client = importlib.import_module("mlflow.tracking").MlflowClient()
    version = _registry_alias_version(
        client, registered_model_name=registered_model_name, alias=alias
    )
    if version is None:
        return None

    # Pin the version immediately. A concurrent alias move must never change
    # which model was compared midway through this training job.
    model_uri = f"models:/{registered_model_name}/{version}"
    try:
        model = importlib.import_module("mlflow.sklearn").load_model(model_uri)
    except Exception as error:
        raise RuntimeError(
            f"Cannot load MLflow {alias} version {version}; "
            "the registry reference is not safe to replace automatically"
        ) from error
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"MLflow {alias} version {version} does not implement predict_proba"
        )
    return RegistryReference(alias=alias, version=version, model=model)


def log_run(
    *,
    metrics: Mapping[str, int | float],
    params: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    registered_model_name: str | None = None,
    model: BaseEstimator | None = None,
    input_example: Any | None = None,
    model_artifact_path: str = "model",
    model_version_tags: Mapping[str, str] | None = None,
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
    version_tag_values = {
        str(key): str(value) for key, value in (model_version_tags or {}).items()
    }
    if any(not key for key in version_tag_values):
        raise ValueError("model_version_tags keys must be non-empty")
    if version_tag_values and not registered_model_name:
        raise ValueError("model_version_tags require a registered_model_name")
    if input_example is not None and model is None:
        raise ValueError("input_example requires a fitted model")
    if registered_model_name and model is None:
        raise ValueError("model is required when registered_model_name is set")
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
            model_version_tags=version_tag_values,
        )

    # Keep real training runs out of the generic Default experiment.
    experiment_name = os.environ.get("PROMISEOPS_MLFLOW_EXPERIMENT") or "promiseops-training"
    mlflow.set_experiment(experiment_name)
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
                # MLflow sklearn's pyfunc flavor calls predict(), whereas the
                # PromiseOps serving path calls predict_proba().
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

    # MLflow Run tags are NOT automatically copied to Model Version tags.
    # Make Registry metadata explicit; fail rather than report a complete job
    # whose registered models lack required traceability.
    if registered_model_name and version_tag_values:
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            raise RuntimeError("MLflow model was logged without a registered model version")
        client = importlib.import_module("mlflow.tracking").MlflowClient()
        for key, value in version_tag_values.items():
            client.set_model_version_tag(registered_model_name, str(version), key, value)

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


def finalize_model_registry(
    *,
    registered_model_name: str,
    registered_model_tags: Mapping[str, str],
    challenger_version: str | None = None,
    expected_challenger_version: str | None = None,
    verify_challenger_version: bool = False,
    tracking_uri: str | None = None,
    allow_local_tracking: bool = False,
) -> None:
    """Set stable registry tags and, optionally, the approved-for-review alias.

    The worker may set ``challenger`` only after all candidate records are
    written. A separate, human-approved CI step owns ``champion`` and serving.
    An unsuccessful training run MUST NOT move either alias.
    """
    if not registered_model_name:
        raise ValueError("registered_model_name must not be empty")
    if challenger_version is not None and (
        not isinstance(challenger_version, str)
        or not challenger_version.isdigit()
        or int(challenger_version) < 1
    ):
        raise ValueError("challenger_version must be a positive model version")
    mlflow = prepare_tracking(
        tracking_uri=tracking_uri,
        allow_local_tracking=allow_local_tracking,
    )
    if mlflow is None:
        return  # No real registry when using the local JSON backend.

    client = importlib.import_module("mlflow.tracking").MlflowClient()
    for key, value in registered_model_tags.items():
        client.set_registered_model_tag(registered_model_name, str(key), str(value))

    if challenger_version is not None:
        if verify_challenger_version:
            # Best-effort optimistic guard: do not overwrite a challenger that
            # moved while this job was training. MLflow does not expose an
            # atomic compare-and-swap operation for aliases.
            current_version = _registry_alias_version(
                client, registered_model_name=registered_model_name, alias="challenger"
            )
            if current_version != expected_challenger_version:
                raise RuntimeError(
                    "challenger alias changed during training; refusing to overwrite it"
                )
        client.set_registered_model_alias(
            registered_model_name, "challenger", challenger_version
        )