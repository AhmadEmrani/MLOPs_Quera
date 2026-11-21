"""Run one durable, chronological scoring window from the Helm CronJob."""

from __future__ import annotations

import json
import os
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta

from promiseops.cli import score_batch
from promiseops.data import get_connection, init_db
from promiseops.model_pipeline.bundle import load_bundle
from promiseops.settings import DEFAULT_RELEASE_PATH

_LOCK_KEY = 741_928_117_203
_PIPELINE_KEY = "scheduled-priority-scoring"


def _lookback() -> timedelta:
    raw = os.environ.get("PROMISEOPS_BATCH_LOOKBACK_HOURS", "24")
    try:
        hours = int(raw)
    except ValueError as error:
        raise ValueError("PROMISEOPS_BATCH_LOOKBACK_HOURS must be an integer") from error
    if not 1 <= hours <= 168:
        raise ValueError("PROMISEOPS_BATCH_LOOKBACK_HOURS must be between 1 and 168")
    return timedelta(hours=hours)


def _release_dir():
    raw = os.environ.get("PROMISEOPS_RELEASE", "").strip()
    path = os.path.abspath(raw) if raw else str(DEFAULT_RELEASE_PATH)
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def run_scheduled_batch() -> dict:
    """Score the next chronological window and persist its cursor and output.

    A PostgreSQL advisory lock prevents overlapping runs across CronJob retries
    and manual invocations. The cursor advances only in the transaction that
    stores the exact JSON result, so replays never count old orders as new.
    """
    step = _lookback()
    model, release = load_bundle(_release_dir())
    with closing(get_connection()) as conn:
        init_db(conn)
        with conn.cursor() as cur:
            # Session lock is retained across commits and released when this
            # connection closes, including on errors.
            cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        conn.commit()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS promiseops_scoring_cursor (
                        pipeline_key VARCHAR(64) PRIMARY KEY,
                        last_as_of TIMESTAMPTZ NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS promiseops_scoring_runs (
                        batch_id VARCHAR(64) PRIMARY KEY,
                        since_at TIMESTAMPTZ NOT NULL,
                        as_of TIMESTAMPTZ NOT NULL,
                        result JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "SELECT last_as_of FROM promiseops_scoring_cursor WHERE pipeline_key = %s FOR UPDATE",
                    (_PIPELINE_KEY,),
                )
                row = cur.fetchone()
                cur.execute("SELECT MIN(batch_available_at), MAX(batch_available_at) FROM orders")
                first_available, last_available = cur.fetchone()

                if first_available is None or last_available is None:
                    conn.commit()
                    return {"status": "no_orders_available", "scored_count": 0}

                since = row[0] if row else first_available - timedelta(microseconds=1)
                since = since.astimezone(UTC)
                last_available = last_available.astimezone(UTC)
                if last_available <= since:
                    conn.commit()
                    return {
                        "status": "no_new_orders",
                        "since": _iso(since),
                        "scored_count": 0,
                    }

                as_of = min(since + step, last_available)
                batch_id = "score-" + as_of.strftime("%Y%m%dT%H%M%SZ")
                result = score_batch(
                    conn,
                    model,
                    release,
                    as_of=as_of,
                    since=since,
                    batch_id=batch_id,
                )
                encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO promiseops_scoring_runs(batch_id, since_at, as_of, result)
                        VALUES (%s, %s, %s, %s::jsonb)
                        """,
                        (batch_id, since, as_of, encoded),
                    )
                    cur.execute(
                        """
                        INSERT INTO promiseops_scoring_cursor(pipeline_key, last_as_of)
                        VALUES (%s, %s)
                        ON CONFLICT (pipeline_key) DO UPDATE SET last_as_of = EXCLUDED.last_as_of
                        """,
                        (_PIPELINE_KEY, as_of),
                    )
                conn.commit()
                return result
        except Exception:
            conn.rollback()
            raise


def main() -> int:
    try:
        result = run_scheduled_batch()
    except Exception as error:  # noqa: BLE001 - never print database URLs or credentials
        print(f"scheduled scoring failed: error={type(error).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
