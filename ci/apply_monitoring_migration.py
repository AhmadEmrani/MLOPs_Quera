#!/usr/bin/env python3
"""Approved, one-time, additive schema migration 004. Never logs credentials.

This command is intentionally distinct from the read-only inspection jobs.
The GitLab job must explicitly be started by an authorized reviewer on a
protected main ref. This script does not create or alter any training models.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

EXPECTED_SHA256 = "3acb4c111207e7ec9d7dd7f20e654c4bf09b36f0437add98e9ab713728b366fb"
LOCK_KEY = 741_928_117_203  # Same advisory key as scheduled_batch.py.
MIGRATION = Path(__file__).resolve().parents[1] / "db/migrations/004_monitoring_predictions.sql"
REQUIRED_COLUMNS = {
    "batch_id": ("character varying(64)", True),
    "order_id": ("character varying(128)", True),
    "model_version": ("character varying(128)", True),
    "risk_probability": ("double precision", True),
    "scored_as_of": ("timestamp with time zone", True),
    "feature_contract_sha256": ("character(64)", True),
    "feature_hash": ("character(64)", True),
    "features_jsonb": ("jsonb", True),
    "persisted_at": ("timestamp with time zone", True),
}
EXPECTED_PK = ("batch_id", "order_id", "model_version")


class MigrationBlocked(RuntimeError):
    pass


def normalize_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return "postgresql://" + dsn.removeprefix("postgresql+psycopg://")
    if dsn.startswith("postgres://"):
        return "postgresql://" + dsn.removeprefix("postgres://")
    return dsn


def migration_commands(path: Path = MIGRATION) -> list[str]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise MigrationBlocked("migration_004_file_changed")
    source = re.sub(r"(?m)^\s*--[^\n]*\n", "", data.decode("utf-8")).strip()
    commands = [part.strip() for part in source.split(";") if part.strip()]
    if len(commands) != 4 or commands[0] != "BEGIN" or commands[-1] != "COMMIT":
        raise MigrationBlocked("migration_004_unexpected_statements")
    if not commands[1].startswith("CREATE TABLE IF NOT EXISTS monitoring_predictions ("):
        raise MigrationBlocked("migration_004_unexpected_table")
    if not commands[2].startswith("CREATE INDEX IF NOT EXISTS idx_monitoring_model_company_window"):
        raise MigrationBlocked("migration_004_unexpected_index")
    return commands[1:3]


def schema_is_valid(cur) -> bool:
    """Verify the expected table, exact column types, PK, and lookup index."""
    cur.execute("SELECT to_regclass('monitoring_predictions') IS NOT NULL")
    if not cur.fetchone()[0]:
        return False
    cur.execute("""
        SELECT attname, format_type(atttypid, atttypmod), attnotnull
        FROM pg_attribute
        WHERE attrelid = to_regclass('monitoring_predictions')
          AND attnum > 0 AND NOT attisdropped
    """)
    actual = {name: (datatype, notnull) for name, datatype, notnull in cur.fetchall()}
    if any(actual.get(name) != expected for name, expected in REQUIRED_COLUMNS.items()):
        return False
    cur.execute("""
        SELECT att.attname
        FROM pg_constraint c
        JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, position) ON TRUE
        JOIN pg_attribute att ON att.attrelid = c.conrelid AND att.attnum = k.attnum
        WHERE c.conrelid = to_regclass('monitoring_predictions') AND c.contype = 'p'
        ORDER BY k.position
    """)
    if tuple(row[0] for row in cur.fetchall()) != EXPECTED_PK:
        return False
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM pg_index
            WHERE indrelid = to_regclass('monitoring_predictions')
              AND indexrelid = to_regclass('idx_monitoring_model_company_window')
        )
    """)
    return bool(cur.fetchone()[0])


def migrate(connection, commands: list[str]) -> str:
    """One guarded transaction; refuse concurrent scheduled scoring."""
    with connection.cursor() as cur:
        cur.execute("BEGIN")
        try:
            cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (LOCK_KEY,))
            if not cur.fetchone()[0]:
                raise MigrationBlocked("scheduled_scoring_in_progress")
            before = schema_is_valid(cur)
            if before:
                cur.execute("COMMIT")
                return "already_applied"
            for command in commands:
                cur.execute(command)
            if not schema_is_valid(cur):
                raise MigrationBlocked("schema_verification_failed")
            cur.execute("COMMIT")
            return "applied"
        except BaseException:
            cur.execute("ROLLBACK")
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-file-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        commands = migration_commands()
        if args.check_file_only:
            print("MIGRATION_004: source_verified; no database connection")
            return 0
        if os.environ.get("CI_COMMIT_BRANCH") != "main" or os.environ.get("CI_COMMIT_REF_PROTECTED") != "true":
            raise MigrationBlocked("requires_protected_main")
        dsn = os.environ.get("PROMISEOPS_DATABASE_URL", "")
        if not dsn:
            raise MigrationBlocked("authorized_database_secret_missing")
        import psycopg
        with psycopg.connect(
            normalize_dsn(dsn), connect_timeout=10, autocommit=True,
            options="-c statement_timeout=30000 -c lock_timeout=5000",
        ) as connection:
            outcome = migrate(connection, commands)
        print(f"MIGRATION_004: {outcome}; schema_verified=true; backfill=not_performed")
        return 0
    except MigrationBlocked as error:
        print(f"MIGRATION_004: blocked reason={error}", file=sys.stderr)
        return 1
    except Exception as error:
        # Transport/SQL exception texts may contain DSNs; only show class + SQLSTATE.
        sqlstate = getattr(error, "sqlstate", None)
        if not isinstance(sqlstate, str) or not re.fullmatch(r"[0-9A-Z]{5}", sqlstate):
            sqlstate = "unknown"
        print(f"MIGRATION_004: failed error_type={type(error).__name__} sqlstate={sqlstate}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
