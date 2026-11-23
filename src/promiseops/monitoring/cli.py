"""Read-only monitoring report CLI; deliberately no train/deploy commands."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from promiseops.monitoring.contracts import instant
from promiseops.monitoring.reference import load_mlflow_reference, validate_reference
from promiseops.monitoring.repository import connect_read_only, fetch_predictions
from promiseops.monitoring.service import build_report

_SHA = re.compile(r"[a-f0-9]{64}\Z")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="Company replay cutoff, ISO-8601 with offset")
    parser.add_argument("--model-version", required=True, help="Immutable serving model_version sha256:...")
    parser.add_argument("--run-id", help="Exact training run, not a moving Registry alias")
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--reference-file", type=Path, help="Local immutable reference for offline inspection")
    parser.add_argument("--validation-metrics", type=Path,
                        help="Optional approved local validation metrics JSON; no default guessing")
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring/preview.json"))
    args = parser.parse_args(argv)
    try:
        when = instant(args.as_of)
        if re.fullmatch(r"sha256:[a-f0-9]{64}", args.model_version) is None:
            raise ValueError("--model-version must be an immutable model artifact sha256")
        if not _SHA.fullmatch(args.reference_sha256):
            raise ValueError("--reference-sha256 must be a full SHA256")
        if args.reference_file:
            reference = validate_reference(args.reference_file.read_bytes(),
                                           expected_sha256=args.reference_sha256)
        elif args.run_id:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            pinned_run = MlflowClient().get_run(args.run_id)
            tags = pinned_run.data.tags or {}
            if pinned_run.info.status != "FINISHED":
                raise ValueError("pinned monitoring reference run did not finish")
            if tags.get("model_version") != args.model_version or \
                    tags.get("feature_reference_sha256") != args.reference_sha256:
                raise ValueError("MLflow run does not match requested immutable model and feature reference")
            reference = load_mlflow_reference(run_id=args.run_id,
                                               expected_sha256=args.reference_sha256)
            auto_reference_metrics = {
                key: pinned_run.data.metrics[key]
                for key in ("recall_at_10pct", "average_precision", "brier")
                if key in pinned_run.data.metrics
            }
            if len(auto_reference_metrics) != 3:
                raise ValueError("pinned MLflow run lacks validation quality reference")
        else:
            raise ValueError("provide a pinned --run-id or --reference-file")
        validation = auto_reference_metrics if args.run_id and not args.reference_file else None
        if args.validation_metrics is not None:
            validation = json.loads(args.validation_metrics.read_text(encoding="utf-8"))
        with connect_read_only(os.environ.get("PROMISEOPS_DATABASE_URL", "")) as connection:
            rows = fetch_predictions(connection, model_version=args.model_version, as_of=when)
        # No salt means active-learning hashes are NOT exported; this is deliberate.
        review_salt = os.environ.get("PROMISEOPS_REVIEW_HASH_SALT") or None
        report = build_report(rows, None, reference, as_of=when,
                              validation_reference=validation, review_salt=review_salt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print("Monitoring report created: status=%s retraining_recommended=%s (no training submitted)" %
              (report["decision"]["status"], report["decision"]["retraining_recommended"]))
        return 0
    except Exception as error:
        # Never log raw MLflow/DB exceptions, paths with secrets or endpoints.
        print(f"Monitoring blocked: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
