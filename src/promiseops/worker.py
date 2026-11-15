"""Command-line entry point for the bounded training job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from promiseops.data import get_connection
from promiseops.model import TrainingResult, train_candidate
from promiseops.model_pipeline.bundle import load_bundle


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and track PromiseOps model candidates")
    parser.add_argument("--mode", choices=("train", "monitor-train"), default="train")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--as-of", required=True)
    return parser


def _database_url() -> str:
    database_url = os.environ.get("PROMISEOPS_DATABASE_URL")
    if not database_url:
        raise RuntimeError("PROMISEOPS_DATABASE_URL is required")
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def _persist_selected_bundle(result: TrainingResult, artifacts_dir: Path) -> Path | None:
    # Rejection is a successful evaluation, not permission to publish a fallback.
    # Leave the incumbent and any existing release untouched.
    if result.selected_model_name is None:
        return None
    if not result.decisions[result.selected_model_name]["promote"]:
        raise ValueError("refusing to export a candidate that failed quality gates")
    candidate_dir = artifacts_dir / "candidates" / result.selected_model_name
    release_path = Path(os.environ.get("PROMISEOPS_RELEASE", artifacts_dir / "release.json"))
    release_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_dir.resolve() != release_path.parent.resolve():
        shutil.copy2(candidate_dir / "model.joblib", release_path.parent / "model.joblib")
        shutil.copy2(candidate_dir / "release.json", release_path)
    return release_path


def _load_incumbent(artifacts_dir: Path):
    source_dir = Path(os.environ.get("PROMISEOPS_INCUMBENT", "/opt/promiseops/release"))
    manifest = source_dir / "release.json"
    model_file = source_dir / "model.joblib"
    if not manifest.exists() and not model_file.exists():
        return None
    if not manifest.is_file() or not model_file.is_file():
        raise RuntimeError(f"incumbent bundle is incomplete: {source_dir}")

    copied_dir = artifacts_dir / "incumbent"
    copied_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_file, copied_dir / model_file.name)
    shutil.copy2(manifest, copied_dir / manifest.name)
    model, _ = load_bundle(copied_dir)
    return model


def _result_payload(result: TrainingResult, release_path: Path | None) -> dict:
    return {
        "status": "completed",
        "mode": "train",
        "selected_model": result.selected_model_name,
        "release_path": str(release_path) if release_path is not None else None,
        "release_created": release_path is not None,
        "selection_decision": (
            "candidate_ready_for_review"
            if release_path is not None
            else "no_candidate_passed_gates"
        ),
        "candidates": {
            name: {
                "metrics": evaluation.metrics,
                "gate_decision": result.decisions[name],
                "tracking": {
                    "backend": tracking.backend,
                    "run_id": tracking.run_id,
                    "model_uri": tracking.model_uri,
                    "registered_model_version": tracking.registered_model_version,
                },
                "bundle": result.bundles[name],
            }
            for name, evaluation in result.evaluations.items()
            for tracking in (result.tracking[name],)
        },
    }


def run_training(*, train_end: str, validation_end: str, as_of: str) -> dict:
    artifacts_dir = Path(os.environ.get("PROMISEOPS_ARTIFACTS", "/tmp/artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    incumbent_model = _load_incumbent(artifacts_dir)
    connection = get_connection(_database_url())
    try:
        result = train_candidate(
            connection,
            train_end=train_end,
            validation_end=validation_end,
            as_of=as_of,
            source_commit=os.environ.get("CI_COMMIT_SHA"),
            champion_model=incumbent_model,
            artifacts_dir=artifacts_dir,
        )
    finally:
        connection.close()

    release_path = _persist_selected_bundle(result, artifacts_dir)
    payload = _result_payload(result, release_path)
    report_path = artifacts_dir / "training-report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        if args.mode == "monitor-train":
            raise NotImplementedError(
                "monitor-train requires the maturity-aware monitoring contract; "
                "src/promiseops/monitor.py is not implemented yet"
            )
        payload = run_training(
            train_end=args.train_end,
            validation_end=args.validation_end,
            as_of=args.as_of,
        )
    except Exception as error:
        print(f"promiseops worker failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
