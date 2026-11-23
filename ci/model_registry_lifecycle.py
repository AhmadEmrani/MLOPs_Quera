#!/usr/bin/env python3
"""Fail-closed MLflow champion lifecycle for immutable PromiseOps releases.

preflight: check the reviewed MLflow version and expected champion before rollout.
promote: require matching production Deployment and smoke evidence before alias move.
reconcile: explicit manual recovery/rollback to a previously approved version.

All calls to MLflow are made via the CI tracking environment; serving does not
resolve moving aliases. No credentials, model bytes or raw feature rows are logged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SHA = re.compile(r"[0-9a-f]{64}\Z")
VERSION = re.compile(r"[1-9][0-9]*\Z")
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class RegistrySafetyError(RuntimeError):
    """A failed safety guard; never mutate aliases on these errors."""


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RegistrySafetyError(message)


def load_json(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    ensure(isinstance(result, dict), "evidence must be a JSON object")
    return result


def save_evidence(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def validate_sha(value: str) -> str:
    ensure(isinstance(value, str) and SHA.fullmatch(value) is not None,
           "model artifact hash must be lowercase SHA256")
    return value


def validate_version(value: str) -> str:
    ensure(isinstance(value, str) and VERSION.fullmatch(value) is not None,
           "MLflow version must be a positive integer")
    return value


def registered_model_name() -> str:
    value = os.environ.get("PROMISEOPS_REGISTERED_MODEL_NAME", "promiseops-late-delivery")
    ensure(bool(MODEL_NAME.fullmatch(value)), "invalid registered model name")
    return value


def validate_selection(selection: dict, version: str, digest: str) -> None:
    validate_version(version)
    validate_sha(digest)
    ensure(selection.get("selected_model_version") == version,
           "MLflow version differs from reviewed selection.json")
    ensure(selection.get("artifact_sha256") == digest,
           "artifact checksum differs from reviewed selection.json")
    ensure(selection.get("public_gates") == "passed",
           "release was not approved by public quality gates")
    ensure(selection.get("model_name") not in (None, "prevalence_baseline"),
           "baseline or unknown model is not eligible for promotion")
    ensure(isinstance(selection.get("mlflow_run_id"), str)
           and bool(selection["mlflow_run_id"]), "missing reviewed MLflow run")
    files = selection.get("file_sha256", {})
    ensure(isinstance(files, dict) and files.get("model.joblib") == digest,
           "reviewed file checksum does not match model artifact")
    for tag in ("champion_at_release", "challenger_at_release"):
        value = selection.get(tag)
        ensure(value == "absent" or (isinstance(value, str) and VERSION.fullmatch(value)),
               f"missing or invalid reviewed alias snapshot: {tag}")


def alias_version(client: Any, model_name: str, alias: str) -> str | None:
    try:
        value = client.get_model_version_by_alias(model_name, alias)
    except Exception as error:
        code = str(getattr(error, "error_code", ""))
        # The Model Version itself has already been verified to exist.
        if code == "RESOURCE_DOES_NOT_EXIST":
            return None
        if code == "INVALID_PARAMETER_VALUE" and re.fullmatch(
            rf"(?:INVALID_PARAMETER_VALUE:\s*)?Registered model alias "
            rf"['\"]?{re.escape(alias)}['\"]? not found\.?",
            str(error).strip(), flags=re.IGNORECASE,
        ):
            return None
        raise RegistrySafetyError(f"MLflow alias lookup failed: {type(error).__name__}") from error
    result = str(value.version)
    validate_version(result)
    return result


def inspect_version(client: Any, name: str, version: str, digest: str,
                    *, require_approved: bool = False, selection: dict | None = None) -> Any:
    validate_version(version)
    validate_sha(digest)
    try:
        item = client.get_model_version(name, version)
    except Exception as error:
        raise RegistrySafetyError(f"Cannot fetch registered MLflow version: {type(error).__name__}") from error
    ensure(str(item.version) == version, "MLflow model version mismatch")
    ensure(getattr(item, "status", "READY") in ("READY", None),
           "MLflow model version is not READY")
    tags = getattr(item, "tags", {}) or {}
    ensure(tags.get("artifact_sha256") == digest,
           "MLflow artifact checksum differs from deployed/reviewed model")
    ensure(tags.get("algorithm") != "prevalence_baseline", "baseline cannot be champion")
    ensure(tags.get("quality_gate") == "passed", "MLflow quality gate is not passed")
    ensure(tags.get("selection_status") == "selected", "MLflow version not selected")
    allowed_status = ("approved",) if require_approved else ("pending", "approved")
    ensure(tags.get("approval_status") in allowed_status,
           "MLflow version does not have the required approval status")
    if selection is not None:
        ensure(str(getattr(item, "run_id", "")) == selection["mlflow_run_id"],
               "reviewed MLflow run ID differs from registry")
        ensure(tags.get("algorithm") == selection["model_name"],
               "reviewed model algorithm differs from registry")
        ensure(tags.get("source_commit") == selection.get("source_commit"),
               "source commit differs from reviewed selection")
        checksums = selection.get("file_sha256", {})
        for file_name, tag_name in (("dataset_manifest.json", "dataset_manifest_sha256"),
                                    ("feature_contract.json", "feature_contract_sha256")):
            ensure(tags.get(tag_name) == checksums.get(file_name),
                   f"Registry tag {tag_name} differs from reviewed file digest")
        ensure(tags.get("validation_cohort_hash") == selection.get("cohort_hash"),
               "reviewed validation cohort differs from registry")
    return item


def preflight(client: Any, model_name: str, selection: dict,
              version: str, digest: str) -> dict[str, Any]:
    validate_selection(selection, version, digest)
    item = inspect_version(client, model_name, version, digest, selection=selection)
    current = alias_version(client, model_name, "champion")
    expected = (None if selection["champion_at_release"] == "absent"
                else selection["champion_at_release"])
    # Replaying a previously completed promotion is safe, but not a different
    # release which changed champion since the reviewed candidate comparison.
    already_promoted = current == version and item.tags.get("approval_status") == "approved"
    ensure(current == expected or already_promoted,
           "Champion moved since candidate review; reevaluate before deploying")
    if not already_promoted:
        current_challenger = alias_version(client, model_name, "challenger")
        reviewed_challenger = (None if selection["challenger_at_release"] == "absent"
                               else selection["challenger_at_release"])
        ensure(current_challenger == reviewed_challenger,
               "Challenger moved since review; repeat model release preparation")
    return {
        "phase": "preflight", "status": "passed", "registered_model": model_name,
        "selected_version": version, "artifact_sha256": digest,
        "champion_before": current or "absent", "reviewed_champion": expected or "absent",
        "already_promoted": already_promoted,
    }


def verify_smoke(smoke: dict, digest: str) -> None:
    ensure(smoke.get("status") == "passed", "Production smoke checks are not passed")
    ensure(smoke.get("model_version") == f"sha256:{digest}",
           "Production /predict model SHA does not match approved MLflow artifact")
    checks = smoke.get("checks", [])
    ensure(isinstance(checks, list) and "prediction" in checks,
           "Production smoke evidence does not include a successful prediction")


def verify_deployment(deployment: dict, expected_image: str) -> dict[str, int]:
    ensure(re.fullmatch(r"[^\s]+/app@sha256:[0-9a-f]{64}", expected_image) is not None,
           "expected production image must be digest-pinned")
    metadata = deployment.get("metadata", {})
    spec = deployment.get("spec", {})
    status = deployment.get("status", {})
    ensure(metadata.get("name") == "promiseops", "unexpected production Deployment")
    containers = spec.get("template", {}).get("spec", {}).get("containers", [])
    images = [c.get("image") for c in containers if c.get("name") == "api"]
    ensure(images == [expected_image], "Production Deployment image digest differs from reviewed image")
    generation = metadata.get("generation", 0)
    ensure(status.get("observedGeneration", 0) >= generation,
           "Production Deployment has not observed its latest generation")
    desired = spec.get("replicas", 0)
    ensure(isinstance(desired, int) and desired >= 2, "Production Deployment replica count is below two")
    for status_name in ("updatedReplicas", "readyReplicas", "availableReplicas"):
        ensure(status.get(status_name, 0) >= desired,
               f"Production Deployment {status_name} is below desired replicas")
    return {"desired": desired, "ready": status["readyReplicas"],
            "available": status["availableReplicas"]}


def read_production_deployment(namespace: str) -> dict:
    ensure(re.fullmatch(r"g0[1-7]-prod", namespace) is not None,
           "invalid production namespace")
    try:
        response = subprocess.run(
            ["kubectl", "get", "deployment", "promiseops", "-n", namespace, "-o", "json"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        result = json.loads(response.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise RegistrySafetyError(f"Cannot inspect Production Deployment: {type(error).__name__}") from error
    ensure(isinstance(result, dict), "invalid deployment API response")
    return result


def move_alias(client: Any, name: str, version: str, *, prior: str | None) -> None:
    # Best-effort optimistic guard. MLflow has no alias-level CAS transaction;
    # CI resource_group serializes our production operations, but external alias
    # edits can race. Verify again after the mutation.
    ensure(alias_version(client, name, "champion") == prior,
           "Champion alias changed during approval; no mutation made")
    if prior != version:
        client.set_registered_model_alias(name, "champion", version)
    ensure(alias_version(client, name, "champion") == version,
           "Champion alias verification failed after write")


def promote(client: Any, model_name: str, selection: dict, version: str,
            digest: str, smoke: dict, deployment: dict,
            expected_image: str, *, ci_job_id: str = "unknown") -> dict[str, Any]:
    # The CI job already ran the stricter read-only preflight *before* deploy.
    # Do not re-check the moving challenger after deploy: Training is allowed
    # to produce a new challenger while an approved image is rolling out.
    # We still reject any change to champion since the reviewed release.
    validate_selection(selection, version, digest)
    item = inspect_version(client, model_name, version, digest, selection=selection)
    current = alias_version(client, model_name, "champion")
    expected = (None if selection["champion_at_release"] == "absent"
                else selection["champion_at_release"])
    ensure(current == expected or
           (current == version and item.tags.get("approval_status") == "approved"),
           "Champion moved since candidate review; no registry mutation")
    verify_smoke(smoke, digest)
    ready = verify_deployment(deployment, expected_image)
    previous = current or "absent"
    if current != version:
        client.set_model_version_tag(model_name, version, "approval_status", "approved")
        client.set_model_version_tag(model_name, version, "production_image", expected_image)
        client.set_model_version_tag(model_name, version, "promotion_ci_job_id", str(ci_job_id))
        client.set_model_version_tag(model_name, version, "previous_champion_version", previous)
        client.set_model_version_tag(model_name, version, "promoted_at_utc", datetime.now(UTC).isoformat())
    move_alias(client, model_name, version, prior=current)
    return {
        "phase": "promote", "status": "passed", "registered_model": model_name,
        "champion_before": previous, "champion_after": version,
        "artifact_sha256": digest, "production_image": expected_image,
        "production_ready_replicas": ready["ready"], "ci_job_id": str(ci_job_id),
    }


def reconcile(client: Any, model_name: str, version: str, smoke: dict,
              deployment: dict, expected_image: str,
              *, ci_job_id: str = "unknown") -> dict[str, Any]:
    # For rollback/recovery: explicit operator picks a PREVIOUSLY APPROVED
    # Registry version. It cannot turn a pending or unreviewed version into champion.
    validate_version(version)
    metadata = client.get_model_version(model_name, version)
    digest = validate_sha((metadata.tags or {}).get("artifact_sha256", ""))
    inspect_version(client, model_name, version, digest, require_approved=True)
    verify_smoke(smoke, digest)
    ready = verify_deployment(deployment, expected_image)
    current = alias_version(client, model_name, "champion")
    client.set_model_version_tag(model_name, version, "production_image", expected_image)
    client.set_model_version_tag(model_name, version, "reconciled_ci_job_id", str(ci_job_id))
    client.set_model_version_tag(model_name, version, "reconciled_at_utc", datetime.now(UTC).isoformat())
    move_alias(client, model_name, version, prior=current)
    return {
        "phase": "reconcile", "status": "passed", "registered_model": model_name,
        "champion_before": current or "absent", "champion_after": version,
        "artifact_sha256": digest, "production_image": expected_image,
        "production_ready_replicas": ready["ready"], "ci_job_id": str(ci_job_id),
    }


def make_client():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    ensure(uri.startswith(("https://", "http://")), "MLFLOW_TRACKING_URI must be a configured HTTP(S) URL")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(uri)
        return MlflowClient()
    except ImportError as error:
        raise RegistrySafetyError("MLflow tracking packages are missing") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "promote", "reconcile", "check-approved"))
    parser.add_argument("--version", required=True, help="Reviewed MLflow registered model version")
    parser.add_argument("--sha256", help="Reviewed model.joblib digest; required for promote/preflight")
    parser.add_argument("--selection", help="Reviewed selection.json; required for promote/preflight")
    parser.add_argument("--smoke", help="Production smoke JSON; required for promote/reconcile")
    parser.add_argument("--namespace", help="Production Kubernetes namespace")
    parser.add_argument("--expected-image", help="Expected digest-pinned deployment image")
    parser.add_argument("--evidence", required=True, help="Output JSON evidence path")
    args = parser.parse_args(argv)
    try:
        name = registered_model_name()
        client = make_client()
        if args.action == "check-approved":
            validate_version(args.version)
            item = client.get_model_version(name, args.version)
            digest = validate_sha((item.tags or {}).get("artifact_sha256", ""))
            inspect_version(client, name, args.version, digest, require_approved=True)
            outcome = {"phase": "check-approved", "status": "passed", "selected_version": args.version, "artifact_sha256": digest}
        elif args.action in ("preflight", "promote"):
            ensure(args.selection and args.sha256, "selection and sha256 are required")
            selection = load_json(args.selection)
            if args.action == "preflight":
                outcome = preflight(client, name, selection, args.version, args.sha256)
            else:
                ensure(args.smoke and args.namespace and args.expected_image,
                       "smoke, namespace and expected-image are required")
                outcome = promote(
                    client, name, selection, args.version, args.sha256,
                    load_json(args.smoke), read_production_deployment(args.namespace),
                    args.expected_image, ci_job_id=os.environ.get("CI_JOB_ID", "unknown"),
                )
        else:
            ensure(args.smoke and args.namespace and args.expected_image,
                   "smoke, namespace and expected-image are required")
            outcome = reconcile(
                client, name, args.version, load_json(args.smoke),
                read_production_deployment(args.namespace), args.expected_image,
                ci_job_id=os.environ.get("CI_JOB_ID", "unknown"),
            )
        save_evidence(args.evidence, outcome)
        print(f"Model Registry {args.action}: PASS; champion={outcome.get('champion_after', outcome.get('champion_before'))}")
        return 0
    except RegistrySafetyError as error:
        print(f"Model Registry {args.action}: BLOCKED: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # MLflow transport/auth failures may include endpoint details in their
        # original exception message; print the exception *type* only.
        print(f"Model Registry {args.action}: BLOCKED ({type(error).__name__}); inspect protected connection/permissions", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
