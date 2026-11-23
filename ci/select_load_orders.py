#!/usr/bin/env python3
"""Safely select bounded, released post-validation orders for an API load test.

Database credentials are supplied by the protected CI runtime, never as CLI args.
No label/outcome data is queried. The private order-ID file is never an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Explicit published-by-as_of and approved-in-holdout guards prevent using
# future/source-unreleased orders. This is only load traffic, not a holdout evaluation.
SELECT_ORDERS_SQL = """
SELECT o.order_id
FROM orders AS o
WHERE o.approved_at > %(validation_end)s
  AND o.approved_at <= %(as_of)s
  AND o.batch_available_at <= %(as_of)s
  AND EXISTS (SELECT 1 FROM items AS i WHERE i.order_id = o.order_id)
  AND EXISTS (SELECT 1 FROM payments AS p WHERE p.order_id = o.order_id)
ORDER BY md5(o.order_id)
LIMIT %(limit)s
"""


def normalize_psycopg_dsn(raw: str) -> str:
    """Convert SQLAlchemy psycopg URLs into a libpq-compatible PostgreSQL URI.

    Production's promiseops.data.get_connection already does this conversion.
    Never log the input: it can include database username and password.
    """
    sqlalchemy_prefix = "postgresql+psycopg://"
    if raw.startswith(sqlalchemy_prefix):
        return "postgresql://" + raw[len(sqlalchemy_prefix):]
    return raw


def parse_aware_datetime(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise ValueError("timestamps must be ISO-8601 with an explicit timezone offset") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def validate_selection(validation_end: datetime, as_of: datetime, limit: int, minimum: int) -> None:
    if validation_end.tzinfo is None or as_of.tzinfo is None:
        raise ValueError("cutoffs must be timezone-aware")
    if not validation_end < as_of:
        raise ValueError("as_of must be strictly after validation_end")
    if not 1 <= minimum <= limit <= 500:
        raise ValueError("require 1 <= minimum <= limit <= 500")


def select_order_ids(connection, validation_end: datetime, as_of: datetime, limit: int) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            SELECT_ORDERS_SQL,
            {"validation_end": validation_end, "as_of": as_of, "limit": limit},
        )
        rows = cursor.fetchall()
    ids = [row[0] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("order selection returned invalid IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("order selection returned duplicate IDs")
    if len(ids) > limit:
        raise ValueError("order selection exceeded the limit")
    return ids


def cohort_hash(order_ids: list[str]) -> str:
    # A stable, non-reversible cohort identifier for evidence, not the order IDs.
    return hashlib.sha256(
        json.dumps(sorted(order_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--minimum", type=int, default=20)
    parser.add_argument("--order-ids", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    phase = "validate_parameters"
    try:
        validation_end = parse_aware_datetime(args.validation_end)
        as_of = parse_aware_datetime(args.as_of)
        validate_selection(validation_end, as_of, args.limit, args.minimum)
        database_url = os.environ.get("PROMISEOPS_DATABASE_URL", "")
        if not database_url:
            raise ValueError("PROMISEOPS_DATABASE_URL is not available in the CI runtime")

        import psycopg

        phase = "connect_postgres"
        # A hard read-only setting applies to ALL transactions, independent of
        # the permissions of the credential supplied by the platform.
        with psycopg.connect(
            normalize_psycopg_dsn(database_url),
            connect_timeout=10,
            options="-c default_transaction_read_only=on -c statement_timeout=30000",
        ) as connection:
            phase = "execute_selection_sql"
            selected = select_order_ids(connection, validation_end, as_of, args.limit)

        phase = "validate_selected_cohort"
        if len(selected) < args.minimum:
            raise ValueError("not enough released post-validation orders in the selected window")

        phase = "write_selection_report"
        args.order_ids.write_text(json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
        args.order_ids.chmod(0o600)
        result = {
            "selection_policy": "post_validation_approved_and_batch_released",
            "validation_end_utc": validation_end.isoformat().replace("+00:00", "Z"),
            "as_of_utc": as_of.isoformat().replace("+00:00", "Z"),
            "selected_unique_orders": len(selected),
            "cohort_sha256": cohort_hash(selected),
            "no_outcomes_used": True,
            "source": "production PostgreSQL read-only query",
        }
        args.summary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(
            f"Selected {len(selected)} distinct released orders for load test; "
            f"validation_end={result['validation_end_utc']} as_of={result['as_of_utc']}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - never print DSNs, SQL parameters or order IDs
        sqlstate = getattr(exc, "sqlstate", None)
        sqlstate = sqlstate if isinstance(sqlstate, str) and len(sqlstate) == 5 else "unavailable"
        diagnostic = (
            f"Load cohort selection failed: phase={phase}; "
            f"error_type={type(exc).__name__}; SQLSTATE={sqlstate}"
        )
        # Only expose a short PostgreSQL schema/permission error message. The
        # SQL is fixed and parameterized, and these server messages do not
        # include bound values or connection credentials.
        if sqlstate in {"42P01", "42703", "42501", "42883", "42601", "42702"}:
            primary = getattr(getattr(exc, "diag", None), "message_primary", None)
            if isinstance(primary, str):
                diagnostic += f"; pg_message={primary[:200]}"
        print(diagnostic, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
