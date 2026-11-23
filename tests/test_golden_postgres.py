"""Opt-in PostgreSQL Golden parity and batch-provenance integration test.

Never connects to production. Requires an explicitly configured disposable DB
whose name contains 'test' AND CREATE SCHEMA privileges.
"""
from __future__ import annotations
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg import sql

from promiseops.contracts import digest
from promiseops.data import build_features, ingest_batch, init_db
from promiseops.feature_lineage import canonical_features, digest_json

ROOT = Path(__file__).resolve().parents[1]
GOLD = json.loads((ROOT / "fixtures/golden_features_v1.json").read_text())


def test_golden_builder_respects_feed_and_label_cutoffs():
    dsn = os.environ.get("PROMISEOPS_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set PROMISEOPS_TEST_DATABASE_URL for isolated PostgreSQL golden test")
    dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    from urllib.parse import urlparse
    db_name = urlparse(dsn).path.rsplit("/", 1)[-1]
    if "test" not in db_name.lower():
        pytest.fail("Refusing golden integration against a DB without 'test' in its name")
    schema = "golden_" + uuid.uuid4().hex[:16]
    conn = psycopg.connect(dsn, connect_timeout=5)
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        conn.commit()
        init_db(conn)
        orders = json.loads((ROOT / "fixtures/synthetic-orders.json").read_text())
        outcomes = json.loads((ROOT / "fixtures/synthetic-outcomes.json").read_text())
        ingest_batch(conn, orders, as_of=GOLD["label_cutoff"], expected_sha256=digest(orders))
        ingest_batch(conn, outcomes, as_of=GOLD["label_cutoff"], expected_sha256=digest(outcomes))
        ids=[c["order_id"] for c in GOLD["cases"]]
        before=build_features(conn, ids, as_of="2017-04-14T23:59:59Z", ensure_schema=False)
        assert before.empty
        frame=build_features(conn, ids, as_of=GOLD["feature_cutoff"], ensure_schema=False)
        assert len(frame)==2
        for case in GOLD["cases"]:
            actual = canonical_features(frame.loc[frame.order_id == case["order_id"]])
            assert actual == case["features"]
            assert digest_json(actual) == case["feature_hash"]
        early=build_features(conn, ids, as_of=GOLD["feature_cutoff"],
                             label_as_of="2017-04-30T23:59:59Z", labeled=True, ensure_schema=False)
        assert early.empty  # batch available May 1, despite earlier delivered_at
        labeled=build_features(conn, ids, as_of=GOLD["feature_cutoff"],
                               label_as_of=GOLD["label_cutoff"], labeled=True, ensure_schema=False)
        assert dict(zip(labeled.order_id, labeled.late)) == {c["order_id"]:c["late"] for c in GOLD["cases"]}
    finally:
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
            conn.commit()
        finally:
            conn.close()
