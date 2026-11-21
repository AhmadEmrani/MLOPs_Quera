"""Prepare a reproducible, immutable PromiseOps G02 model package from MLflow.

This is an approval-time CI task, NOT a training job. It never changes aliases,
loads pickle/joblib objects, deploys to Kubernetes, or logs credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

REGISTERED_MODEL = "promiseops-late-delivery"
REQUIRED_FILES = ("model.joblib", "release.json", "feature_contract.json", "dataset_manifest.json")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
HEX40 = re.compile(r"[a-f0-9]{40}\Z")
RUN_ID_RE = re.compile(r"[a-zA-Z0-9_-]{8,128}\Z")
ALGORITHM = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")
MAX_AP_DROP = 0.02
MAX_BRIER_WORSENING = 0.005
EPS = 1e-12


class ReleaseRejected(ValueError):
    """Candidate is not safe or eligible for the next release phase."""


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseRejected(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fd:
        for part in iter(lambda: fd.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ensure(isinstance(obj, dict), f"invalid JSON object in {path.name}")
    return obj


def safe_number(value: Any, label: str) -> float:
    ensure(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    ensure(math.isfinite(number), f"{label} must be finite")
    return number


def metrics_of(value: Any, label: str) -> dict[str, float]:
    ensure(isinstance(value, dict), f"{label} must be a mapping")
    return {k: safe_number(value.get(k), f"{label}.{k}")
            for k in ("recall_at_10pct", "average_precision", "brier")}


def strict_public_gates(candidate: dict[str, float], reference: dict[str, float]) -> list[str]:
    """The public contract requires ALL THREE gates, not recall-only promotion."""
    reasons = []
    if candidate["recall_at_10pct"] < reference["recall_at_10pct"] - EPS:
        reasons.append("recall_at_10pct_decreased")
    if candidate["average_precision"] < reference["average_precision"] - MAX_AP_DROP - EPS:
        reasons.append("average_precision_dropped_by_more_than_0.02")
    if candidate["brier"] > reference["brier"] + MAX_BRIER_WORSENING + EPS:
        reasons.append("brier_worsened_by_more_than_0.005")
    return reasons


def missing_alias(error: Exception, alias: str) -> bool:
    """Recognize only MLflow's documented/observed absent-alias responses.

    Network/auth errors and unrelated INVALID_PARAMETER_VALUE failures propagate.
    """
    code = str(getattr(error, "error_code", ""))
    message = str(error).strip()
    alias_expr = re.escape(alias)
    exact = re.fullmatch(
        rf"(?:INVALID_PARAMETER_VALUE:\s*)?Registered model alias ['\"]?{alias_expr}['\"]? not found\.?",
        message,
        flags=re.IGNORECASE,
    )
    return bool(exact) and code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def alias_version(client: Any, name: str, alias: str) -> str | None:
    try:
        result = client.get_model_version_by_alias(name, alias)
    except Exception as error:
        if missing_alias(error, alias):
            return None
        raise
    result_version = str(result.version)
    ensure(result_version.isdecimal() and int(result_version) > 0, f"invalid {alias} version")
    return result_version


def pick_version(client: Any, name: str, requested: str) -> tuple[str, str | None]:
    challenger = alias_version(client, name, "challenger")
    if requested:
        ensure(requested.isdecimal() and int(requested) > 0, "MODEL_VERSION must be a positive integer")
        return requested, challenger
    ensure(challenger is not None, "No challenger alias; set MODEL_VERSION explicitly or train a passing challenger")
    return challenger, challenger


def compare_evidence(*, release: dict[str, Any], version_tags: dict[str, str],
                     champion: str | None, challenger: str | None, selected_version: str) -> dict[str, Any]:
    """Check persisted *same-cohort* evaluation evidence, never compare old run scores.

    If champion has changed since training, block release and require reevaluation.
    """
    name = release.get("model_name")
    ensure(isinstance(name, str) and ALGORITHM.fullmatch(name) is not None, "invalid algorithm name")
    ensure(name != "prevalence_baseline", "baseline cannot be published as a challenger")
    ensure(version_tags.get("algorithm") == name, "Registry algorithm tag disagrees with release")
    ensure(version_tags.get("quality_gate") == "passed", "Registry quality_gate is not passed")
    ensure(version_tags.get("selection_status") == "selected", "Model Version is not selected")
    ensure(version_tags.get("approval_status") == "pending", "Model Version is not pending review")

    decision = release.get("gate_decision")
    ensure(isinstance(decision, dict), "missing gate_decision")
    ensure(decision.get("promote") is True, "worker decision does not authorize challenger review")
    ensure(decision.get("quality_gate_passed") is True, "worker quality gate is not passed")
    ensure(decision.get("cohort_hash") == release.get("cohort_hash"), "gate cohort hash mismatch")
    ensure(version_tags.get("validation_cohort_hash") == release.get("cohort_hash"),
           "registry validation cohort hash mismatch")
    ensure(version_tags.get("artifact_sha256") == release.get("artifact_sha256"),
           "registry artifact checksum mismatch")
    ensure(version_tags.get("source_commit") == release.get("source_commit"),
           "registry source commit mismatch")
    ensure(version_tags.get("dataset_manifest_sha256") == release.get("dataset_manifest_sha256"),
           "dataset manifest identity mismatch")
    ensure(version_tags.get("feature_contract_sha256") == release.get("feature_contract_sha256"),
           "feature contract identity mismatch")

    candidate = metrics_of(decision.get("candidate_metrics"), "candidate_metrics")
    validation = metrics_of(release.get("metrics"), "release.metrics")
    for key in candidate:
        ensure(abs(candidate[key] - validation[key]) <= 1e-9, f"candidate {key} differs from release")
    reference = metrics_of(decision.get("reference_metrics"), "reference_metrics")
    ensure(not strict_public_gates(candidate, reference),
           "public quality gates failed against the recorded primary reference: "
           + ", ".join(strict_public_gates(candidate, reference)))

    compared_champion = version_tags.get("champion_version_at_evaluation", "absent")
    compared_challenger = version_tags.get("challenger_version_at_evaluation", "absent")
    if champion:
        ensure(compared_champion == champion,
               "Champion changed or was created since this evaluation; retrain/reevaluate on common cohort")
        versions = decision.get("incumbent_versions")
        metrics = decision.get("incumbent_metrics")
        better = decision.get("incumbent_comparisons")
        ensure(isinstance(versions, dict) and versions.get("champion") == champion,
               "missing pinned champion evaluation version")
        ensure(isinstance(metrics, dict) and "champion" in metrics,
               "missing same-cohort champion metrics")
        ensure(isinstance(better, dict) and better.get("champion") is True,
               "candidate was not better than the current champion")
        champion_metrics = metrics_of(metrics["champion"], "champion_metrics")
        ensure(not strict_public_gates(candidate, champion_metrics),
               "public quality gates failed against the same-cohort champion: "
               + ", ".join(strict_public_gates(candidate, champion_metrics)))
    else:
        ensure(compared_champion == "absent", "champion disappeared since evaluation; recheck before release")
        # On the initial release the worker must compare against the prevalence baseline.
        ensure(decision.get("reference") == "prevalence_baseline",
               "first model requires explicit same-cohort prevalence baseline gates")

    if challenger is not None and selected_version != challenger:
        ensure(compared_challenger == challenger,
               "Challenger changed; manually selected version needs a fresh same-cohort comparison")
        comps = decision.get("incumbent_comparisons")
        ensure(isinstance(comps, dict) and comps.get("challenger") is True,
               "manually selected version does not outperform current challenger")
        challenger_metrics = decision.get("incumbent_metrics", {}).get("challenger")
        ensure(challenger_metrics is not None, "missing current challenger comparison metrics")
        ensure(not strict_public_gates(candidate, metrics_of(challenger_metrics, "challenger_metrics")),
               "public quality gates failed against the current challenger")

    ensure(isinstance(release.get("source_commit"), str)
           and HEX40.fullmatch(release["source_commit"]) is not None,
           "source_commit must be a full lowercase SHA")
    ensure(isinstance(release.get("cohort_hash"), str)
           and HEX64.fullmatch(release["cohort_hash"]) is not None,
           "cohort_hash must be SHA256")
    for key in ("dataset_identity", "schema_identity", "train_end", "validation_end", "as_of", "label_cutoff"):
        ensure(isinstance(release.get(key), str) and bool(release[key]), f"missing {key}")
    return {
        "selected_model_version": selected_version,
        "champion_at_release": champion or "absent",
        "challenger_at_release": challenger or "absent",
        "cohort_hash": release["cohort_hash"],
        "model_name": name,
        "public_gates": "passed",
        "public_gate_policy": "recall_non_decreasing_ap_drop_le_0.02_brier_up_le_0.005",
        "metrics": candidate,
    }


def prepare(*, client: Any, model_name: str, requested_version: str,
            destination: Path) -> dict[str, Any]:
    version, current_challenger = pick_version(client, model_name, requested_version)
    current_champion = alias_version(client, model_name, "champion")
    model_version = client.get_model_version(model_name, version)
    ensure(str(model_version.version) == version, "version lookup mismatch")
    ensure(getattr(model_version, "status", "READY") in (None, "READY"), "Model Version is not READY")
    run_id = str(model_version.run_id)
    ensure(RUN_ID_RE.fullmatch(run_id) is not None, "registered version has invalid run_id")
    tags = dict(model_version.tags or {})
    algorithm = tags.get("algorithm", "")
    ensure(ALGORITHM.fullmatch(algorithm) is not None, "model version missing a safe algorithm tag")
    run = client.get_run(run_id)
    ensure(run.info.status == "FINISHED", "MLflow run did not finish successfully")

    destination.mkdir(parents=True, exist_ok=True)
    # The worker logs the exact trusted serving bundle at this path. Do not
    # replace it with mlflow's serialized sklearn model flavor directory.
    with tempfile.TemporaryDirectory(prefix="promiseops-mlflow-") as temp_string:
        temporary = Path(temp_string)
        for file_name in REQUIRED_FILES:
            rel = f"candidates/{algorithm}/{file_name}"
            remote = Path(client.download_artifacts(run_id, rel, str(temporary)))
            ensure(remote.is_file() and not remote.is_symlink(), f"missing regular MLflow artifact {rel}")
            ensure(remote.resolve().is_relative_to(temporary.resolve()), "artifact path escaped download dir")
            shutil.copyfile(remote, destination / file_name)

    release = json_file(destination / "release.json")
    ensure(release.get("model_name") == algorithm, "release algorithm mismatches Registry")
    ensure(release.get("artifact_file") == "model.joblib", "release requires model.joblib")
    digest = release.get("artifact_sha256")
    ensure(isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
           "invalid model artifact SHA256")
    ensure(sha256(destination / "model.joblib") == digest, "model.joblib checksum mismatch")
    for file_name, checksum_key in (("feature_contract.json", "feature_contract_sha256"),
                                    ("dataset_manifest.json", "dataset_manifest_sha256")):
        digest_value = release.get(checksum_key)
        ensure(isinstance(digest_value, str) and HEX64.fullmatch(digest_value) is not None,
               f"invalid {checksum_key}")
        ensure(sha256(destination / file_name) == digest_value,
               f"{file_name} checksum mismatch")
        json_file(destination / file_name)
    ensure(release.get("model_version") == f"sha256:{digest}",
           "immutable bundle model identity mismatch")
    evidence = compare_evidence(
        release=release,
        version_tags=tags,
        champion=current_champion,
        challenger=current_challenger,
        selected_version=version,
    )
    for key in ("recall_at_10pct", "average_precision", "brier"):
        run_metric = run.data.metrics.get(key)
        ensure(run_metric is not None and abs(safe_number(run_metric, key) - evidence["metrics"][key]) <= 1e-9,
               f"MLflow Run {key} disagrees with Bundle")

    # Re-check aliases before finishing to catch most in-flight changes. The
    # CI resource_group serializes this job, but training itself may run concurrently.
    ensure(alias_version(client, model_name, "champion") == current_champion,
           "Champion changed during release preparation")
    ensure(alias_version(client, model_name, "challenger") == current_challenger,
           "Challenger changed during release preparation")

    package_version = f"mlflow-v{version}-{digest[:16]}"
    evidence.update({
        "package_name": "promiseops-release",
        "package_version": package_version,
        "registered_model_name": model_name,
        "mlflow_run_id": run_id,
        "source_commit": release["source_commit"],
        "artifact_sha256": digest,
        "file_sha256": {file_name: sha256(destination / file_name)
                        for file_name in REQUIRED_FILES},
    })
    (destination / "selection.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "release.env").write_text(
        "\n".join((
            f"MODEL_PACKAGE_VERSION={package_version}",
            f"MODEL_MLFLOW_VERSION={version}",
            f"MODEL_SOURCE_COMMIT={release['source_commit']}",
            f"MODEL_ARTIFACT_SHA256={digest}",
        )) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="dist/model-release")
    args = parser.parse_args()
    raw_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    ensure(raw_uri.startswith(("http://", "https://")), "MLFLOW_TRACKING_URI must be a configured HTTP(S) endpoint")
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(raw_uri)
    result = prepare(
        client=MlflowClient(),
        model_name=os.environ.get("PROMISEOPS_REGISTERED_MODEL_NAME", REGISTERED_MODEL),
        requested_version=os.environ.get("MODEL_VERSION", "").strip(),
        destination=Path(args.destination),
    )
    print(f"Approved candidate: MLflow version {result['selected_model_version']} "
          f"({result['model_name']}), package={result['package_version']}")
    print("Comparison: strict public contract gates PASSED; no alias or production state modified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseRejected, OSError, ValueError) as exc:
        print(f"MODEL RELEASE REJECTED: {exc}", file=__import__('sys').stderr)
        raise SystemExit(1)
