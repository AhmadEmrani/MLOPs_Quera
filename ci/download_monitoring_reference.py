#!/usr/bin/env python3
"""Private MLflow reference staging for the read-only CI preview.

This runs in the offline tracking wheelhouse environment and never queries DB.
It writes no untrusted model binaries or raw rows, only reference and metrics.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

from promiseops.monitoring.reference import download_mlflow_reference_bytes, validate_reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", args.model_version) is None:
            raise ValueError("invalid immutable model version")
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        run = MlflowClient().get_run(args.run_id)
        if run.info.status != "FINISHED":
            raise ValueError("training run is not finished")
        tags = run.data.tags or {}
        if (tags.get("model_version") != args.model_version or
                tags.get("feature_reference_sha256") != args.reference_sha256):
            raise ValueError("training reference is not pinned to selected model")
        raw = download_mlflow_reference_bytes(run_id=args.run_id,
                                                expected_sha256=args.reference_sha256)
        reference = validate_reference(raw, expected_sha256=args.reference_sha256)
        from promiseops.monitoring.drift import reference_contract
        reference_contract(reference)
        metrics = {key: run.data.metrics[key]
                   for key in ("recall_at_10pct", "average_precision", "brier")
                   if key in run.data.metrics}
        if len(metrics) != 3:
            raise ValueError("training run lacks required validation metrics")
        args.destination.mkdir(parents=True, exist_ok=True)
        # Preserve the *original bytes* to keep the approved MLflow checksum.
        (args.destination / "feature_reference.json").write_bytes(raw)
        (args.destination / "validation_metrics.json").write_text(
            json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
        print("Pinned feature reference and validation metrics checked (values withheld).")
        return 0
    except Exception as error:
        print(f"Reference preparation blocked: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
