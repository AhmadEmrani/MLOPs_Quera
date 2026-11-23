"""Load a pinned model run's immutable feature reference; no alias lookups."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from promiseops.monitoring.drift import reference_contract

_SHA = re.compile(r"[a-f0-9]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9_-]{8,128}\Z")


def validate_reference(raw: bytes, *, expected_sha256: str, expected_contract: str | None = None) -> dict:
    if not _SHA.fullmatch(expected_sha256):
        raise ValueError("expected feature reference SHA256 is required")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("feature reference SHA256 mismatch")
    data = json.loads(raw)
    contract, _ = reference_contract(data)
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("feature contract SHA256 mismatch")
    return data


def download_mlflow_reference_bytes(*, run_id: str, expected_sha256: str,
                                    expected_contract: str | None = None) -> bytes:
    if not _RUN.fullmatch(run_id):
        raise ValueError("invalid immutable MLflow run ID")
    from mlflow.tracking import MlflowClient  # imported only on approved CI/worker runtime
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="promiseops-feature-ref-") as temp:
        destination = Path(MlflowClient().download_artifacts(
            run_id, "monitoring/feature_reference.json", temp))
        if not destination.is_file() or destination.is_symlink():
            raise ValueError("missing regular MLflow reference artifact")
        if not destination.resolve().is_relative_to(Path(temp).resolve()):
            raise ValueError("reference artifact escaped temporary directory")
        raw = destination.read_bytes()
        validate_reference(raw, expected_sha256=expected_sha256,
                           expected_contract=expected_contract)
        return raw


def load_mlflow_reference(*, run_id: str, expected_sha256: str,
                          expected_contract: str | None = None) -> dict:
    raw = download_mlflow_reference_bytes(run_id=run_id, expected_sha256=expected_sha256,
                                          expected_contract=expected_contract)
    return validate_reference(raw, expected_sha256=expected_sha256,
                              expected_contract=expected_contract)
