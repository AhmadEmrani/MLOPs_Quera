"""Command line entry point: ``promiseops score-batch``."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from promiseops.data import FEATURE_COLUMNS, build_features, get_connection
from promiseops.metrics import cohort_hash, cutoff
from promiseops.model_pipeline.bundle import load_bundle
from promiseops.settings import DEFAULT_RELEASE_PATH

SCHEMA_VERSION = 1
CHUNK = 500
BATCH_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


class ConfigError(Exception):
    """Missing configuration. The message names the variable, never its value."""


def _aware(text: str) -> datetime:
    try:
        value = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid ISO timestamp") from error
    if value.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return value.astimezone(UTC)


def _batch_id(text: str) -> str:
    if not BATCH_ID_PATTERN.fullmatch(text):
        raise argparse.ArgumentTypeError("batch id must match [A-Za-z0-9._-]{1,64}")
    return text


def _cohort_ids(conn: Any, since: datetime, as_of: datetime) -> list[str]:
    """Orders whose batch became available in (since, as_of].

    Daily scoring passes the previous run's as_of as ``since``, so each order
    is scored once, when its orders batch is released. Orders are scored at
    approval time, so delivery is not required.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT order_id FROM orders WHERE batch_available_at > %s AND batch_available_at <= %s"
            " ORDER BY order_id",
            (since, as_of),
        )
        return [row[0] for row in cur.fetchall()]


def _model_version(release: dict) -> str:
    version = release.get("model_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("release has no model_version")
    return version


def score_batch(
    conn: Any, model: Any, release: dict, *, as_of: datetime, since: datetime, batch_id: str,
    monitoring_rows: list[dict] | None = None,
) -> dict:
    version = _model_version(release)
    ids = _cohort_ids(conn, since, as_of)
    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "since": since.isoformat(),
        "batch_id": batch_id,
        "cohort_hash": None,
        "model_version": version,
        "cohort_count": len(ids),
        "scored_count": 0,
        "top_k": 0,
        "top": [],
    }
    if not ids:
        return result
    result["cohort_hash"] = cohort_hash(ids)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("model has no positive class")
    column = classes.index(1)
    scored: list[tuple[str, float]] = []
    for start in range(0, len(ids), CHUNK):
        frame = build_features(conn, ids[start : start + CHUNK], as_of=as_of, ensure_schema=False)
        if frame.empty:
            continue
        probs = model.predict_proba(frame.loc[:, list(FEATURE_COLUMNS)])
        for offset, (order_id, row) in enumerate(zip(frame["order_id"], probs, strict=True)):
            prob = float(row[column])
            if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
                raise ValueError("probability out of range")
            scored.append((str(order_id), prob))
            if monitoring_rows is not None:
                from promiseops.feature_lineage import observation
                snapshot = observation(frame.iloc[[offset]], feature_as_of=as_of, release=release)
                monitoring_rows.append({
                    "batch_id": batch_id,
                    "order_id": str(order_id),
                    "model_version": version,
                    "risk_probability": prob,
                    "scored_as_of": as_of,
                    "feature_contract_sha256": snapshot["feature_contract_sha256"],
                    "feature_hash": snapshot["feature_hash"],
                    "features": snapshot["features"],
                })
    if len(scored) != len(ids):
        raise ValueError("features missing for part of the cohort")
    scored.sort(key=lambda item: (-item[1], item[0]))
    k = cutoff(len(scored))
    result["scored_count"] = len(scored)
    result["top_k"] = k
    result["top"] = [{"order_id": o, "risk_probability": p} for o, p in scored[:k]]
    return result


def _setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"missing setting: {name}")
    return value


def _release_dir() -> Path:
    raw = os.environ.get("PROMISEOPS_RELEASE", "").strip()
    path = Path(raw) if raw else DEFAULT_RELEASE_PATH
    return path if path.is_dir() else path.parent


def _run(args: argparse.Namespace) -> int:
    database_url = _setting("PROMISEOPS_DATABASE_URL")
    model, release = load_bundle(_release_dir())
    with closing(get_connection(database_url)) as conn:
        result = score_batch(conn, model, release, as_of=args.as_of, since=args.since, batch_id=args.batch_id)
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="promiseops")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score-batch", help="score a cohort and return the top 10%% by risk")
    p.add_argument("--as-of", type=_aware, required=True)
    p.add_argument("--since", type=_aware, required=True)
    p.add_argument("--batch-id", type=_batch_id, required=True)
    args = parser.parse_args(argv)
    if args.since >= args.as_of:
        parser.error("--since must be earlier than --as-of")
    try:
        return _run(args)
    except ConfigError as error:
        sys.stderr.write(f"score-batch failed: batch_id={args.batch_id} {error}\n")
        return 1
    except Exception as error:  # noqa: BLE001 - never leak DB details or secrets
        sys.stderr.write(f"score-batch failed: batch_id={args.batch_id} error={type(error).__name__}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
