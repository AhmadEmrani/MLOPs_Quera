#!/usr/bin/env python3
"""Publish immutable CI golden-test evidence as a separate MLflow run.

This logs metrics/artifacts only; it never registers a model or mutates aliases.
The job must run with protected MLFLOW_TRACKING_URI credentials.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit


def publish(report_path: Path, *, tracking_uri: str) -> str:
    url = urlsplit(tracking_uri)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) server endpoint")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "passed":
        raise ValueError("Refusing to publish missing/failed golden evidence as passing")
    if int(report.get("tests_passed", 0)) < 8 or any(int(report.get(k, 0)) for k in ("tests_failed", "tests_errors")):
        raise ValueError("Invalid report test counts")
    fixture_hash = str(report.get("fixture_sha256", ""))
    if not re.fullmatch(r"[a-f0-9]{64}", fixture_hash):
        raise ValueError("Invalid golden fixture checksum")
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("PROMISEOPS_GOLDEN_EXPERIMENT", "promiseops-golden-tests"))
    version = os.environ.get("PROMISEOPS_GOLDEN_MODEL_VERSION", "not-bound-to-model")
    with mlflow.start_run(run_name="golden-ci-" + str(report.get("ci_job_id", "local"))) as run:
        mlflow.log_params({
            "fixture_sha256": fixture_hash,
            "source_commit": report["source_commit"],
            "model_version": version,
            "suite": report["suite"],
        })
        mlflow.log_metrics({k: float(report[k]) for k in (
            "tests_total", "tests_passed", "tests_failed", "tests_errors", "tests_skipped"
        )})
        mlflow.set_tags({
            "evidence_type": "golden_temporal_contracts",
            "status": "passed",
            "ci_pipeline_id": str(report["ci_pipeline_id"]),
            "ci_job_id": str(report["ci_job_id"]),
            "postgres_integration_executed": str(bool(report["postgres_integration_executed"])).lower(),
            "model_output_golden_executed": str(bool(report.get("model_output_golden_executed", False))).lower(),
        })
        mlflow.log_artifact(str(report_path), artifact_path="golden")
        junit = report_path.parent / "junit.xml"
        if junit.exists():
            mlflow.log_artifact(str(junit), artifact_path="golden")
        return run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/golden/report.json"))
    args = parser.parse_args()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        raise SystemExit("MLFLOW_TRACKING_URI protected variable is required")
    run_id = publish(args.report, tracking_uri=uri)
    print("MLflow golden evidence run ID: " + run_id)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
