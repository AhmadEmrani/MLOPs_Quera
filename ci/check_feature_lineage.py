#!/usr/bin/env python3
"""Read-only evidence for feature lineage persistence (aggregate counts only)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def normalize_dsn(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://"):]
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def read_stats(database_url: str, hours: int) -> dict:
    import psycopg

    with psycopg.connect(
        normalize_dsn(database_url),
        connect_timeout=10,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    ) as connection:
        with connection.cursor() as cursor:
            # All values are aggregate counts; no IDs, feature values, or secrets.
            cursor.execute(
                """
                SELECT count(*)::bigint,
                       count(DISTINCT order_id)::bigint,
                       count(DISTINCT model_version)::bigint,
                       count(DISTINCT feature_contract_sha256)::bigint,
                       count(*) FILTER (
                           WHERE contract_identity_source = 'fallback_names_only'
                       )::bigint
                FROM prediction_feature_observations
                WHERE observed_at >= NOW() - (%s * INTERVAL '1 hour')
                """,
                (hours,),
            )
            rows, orders, models, contracts, fallbacks = cursor.fetchone()
    return {
        "schema_version": 1,
        "inspection_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "window_hours": hours,
        "distinct_feature_snapshots": int(rows),
        "unique_orders_with_snapshot": int(orders),
        "model_versions": int(models),
        "feature_contract_versions": int(contracts),
        "fallback_contract_snapshots": int(fallbacks),
        "no_observations": rows == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("reports/feature-lineage.json"))
    args = parser.parse_args()
    if not 1 <= args.hours <= 24 * 90:
        raise SystemExit("--hours must be in 1..2160")
    url = os.environ.get("PROMISEOPS_DATABASE_URL")
    if not url:
        raise SystemExit("PROMISEOPS_DATABASE_URL is required")
    try:
        report = read_stats(url, args.hours)
    except Exception as exc:
        # Never print connection error details; they may contain credentials.
        print(f"Feature lineage inspection failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not report["no_observations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
