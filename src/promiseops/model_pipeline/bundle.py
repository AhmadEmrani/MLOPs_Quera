"""Save and validate immutable model bundles for offline serving."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib

from promiseops.model_pipeline.validation import FEATURE_COLUMNS


BUNDLE_FORMAT_VERSION = 1
MODEL_FILENAME = "model.joblib"
RELEASE_FILENAME = "release.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_METADATA = {
    "model_name",
    "source_commit",
    "dataset_identity",
    "schema_identity",
    "train_end",
    "validation_end",
    "as_of",
    "label_cutoff",
    "cohort_hash",
    "metrics",
    "gate_decision",
    "feature_names",
}


def artifact_sha256(path: str | Path) -> str:
    """Return the SHA256 digest of a file, reading it in bounded-size chunks."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {artifact_path}")
    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_version(artifact_digest: str) -> str:
    """Create an immutable model identity from its serialized artifact digest."""
    if not _SHA256_PATTERN.fullmatch(artifact_digest):
        raise ValueError("artifact_digest must be a lowercase SHA256 hex digest")
    return f"sha256:{artifact_digest}"


def _parse_instant(value: Any, *, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    missing = _REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"bundle metadata is missing fields: {', '.join(sorted(missing))}")
    if not isinstance(metadata["model_name"], str) or not metadata["model_name"]:
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(metadata["source_commit"], str) or not _COMMIT_PATTERN.fullmatch(
        metadata["source_commit"]
    ):
        raise ValueError("source_commit must be a 40-character lowercase Git commit SHA")
    for field in ("dataset_identity", "schema_identity"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"{field} must be a non-empty string")

    train_end = _parse_instant(metadata["train_end"], field="train_end")
    validation_end = _parse_instant(metadata["validation_end"], field="validation_end")
    as_of = _parse_instant(metadata["as_of"], field="as_of")
    _parse_instant(metadata["label_cutoff"], field="label_cutoff")
    if not train_end < validation_end <= as_of:
        raise ValueError("bundle cutoffs must satisfy train_end < validation_end <= as_of")

    if not isinstance(metadata["cohort_hash"], str) or not _SHA256_PATTERN.fullmatch(
        metadata["cohort_hash"]
    ):
        raise ValueError("cohort_hash must be a lowercase SHA256 hex digest")
    feature_names = metadata["feature_names"]
    if list(feature_names) != list(FEATURE_COLUMNS):
        raise ValueError("feature_names must match the canonical feature contract and order")
    if not isinstance(metadata["metrics"], dict) or not metadata["metrics"]:
        raise ValueError("metrics must be a non-empty mapping")
    for name, value in metadata["metrics"].items():
        if not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("metrics must have string names and finite numeric values")
    if not isinstance(metadata["gate_decision"], dict):
        raise ValueError("gate_decision must be a mapping")

    normalized = dict(metadata)
    normalized["train_end"] = train_end.isoformat().replace("+00:00", "Z")
    normalized["validation_end"] = validation_end.isoformat().replace("+00:00", "Z")
    normalized["as_of"] = as_of.isoformat().replace("+00:00", "Z")
    normalized["label_cutoff"] = _parse_instant(
        metadata["label_cutoff"], field="label_cutoff"
    ).isoformat().replace("+00:00", "Z")
    normalized["feature_names"] = list(feature_names)
    normalized.setdefault("software_versions", _software_versions())
    return normalized


def _software_versions() -> dict[str, str]:
    packages = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
    }
    versions = {"python": __import__("platform").python_version()}
    for key, package in packages.items():
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("bundle timestamps must be timezone-aware")
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                value,
                temporary_file,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_bundle(model: Any, release_dir: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Serialize a fitted probability model and its release manifest.

    The directory contains ``model.joblib`` and ``release.json``. Only load
    bundles produced by a trusted training and release process because joblib
    deserialization can execute code from the artifact.
    """
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("bundle model must be a fitted classifier with predict_proba")
    if 1 not in model.classes_:
        raise ValueError("bundle model must include class 1 for late orders")
    release = _validate_metadata(dict(metadata))

    destination = Path(release_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    temporary_model: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".model.", suffix=".joblib", dir=destination
        )
        os.close(descriptor)
        temporary_model = Path(temporary_name)
        joblib.dump(model, temporary_model, compress=3)
        digest = artifact_sha256(temporary_model)
        os.replace(temporary_model, model_path)
    finally:
        if temporary_model is not None and temporary_model.exists():
            temporary_model.unlink()

    release.update(
        {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "artifact_file": MODEL_FILENAME,
            "artifact_sha256": digest,
            "model_version": model_version(digest),
        }
    )
    _write_json_atomic(destination / RELEASE_FILENAME, release)
    return release


def load_bundle(release_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Verify a release manifest and load its trusted fitted model artifact."""
    directory = Path(release_dir)
    manifest_path = directory / RELEASE_FILENAME
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            release = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("release manifest is missing or invalid") from error

    if release.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("unsupported bundle format version")
    if release.get("artifact_file") != MODEL_FILENAME:
        raise ValueError("release manifest names an unexpected model artifact")
    expected_digest = release.get("artifact_sha256")
    if not isinstance(expected_digest, str) or not _SHA256_PATTERN.fullmatch(expected_digest):
        raise ValueError("release manifest has an invalid artifact digest")
    if release.get("model_version") != model_version(expected_digest):
        raise ValueError("model_version does not match the artifact digest")
    _validate_metadata(release)

    model_path = directory / MODEL_FILENAME
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("model artifact is missing or is not a regular file")
    actual_digest = artifact_sha256(model_path)
    if actual_digest != expected_digest:
        raise ValueError("model artifact checksum does not match release manifest")

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_") or 1 not in model.classes_:
        raise ValueError("loaded artifact is not a fitted late-probability classifier")
    return model, release
