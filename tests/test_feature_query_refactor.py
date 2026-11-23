"""SQL-shape regression: serving must not join outcomes or execute DDL."""
from __future__ import annotations

import pytest

from promiseops import data


class Cursor:
    description = [("order_id",), ("approved_at",)]
    def __init__(self, connection):
        self.connection=connection
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def execute(self, sql, params):
        self.connection.queries.append((sql,params))
    def fetchall(self):
        return []

class Connection:
    def __init__(self):
        self.queries=[]
    def cursor(self):
        return Cursor(self)


def test_serving_query_skips_outcome_join_and_schema_changes(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda _: pytest.fail("per-request DDL must never happen"))
    conn=Connection()
    assert data.build_features(conn, ["o1"], as_of="2018-04-30T23:59:59-03:00", ensure_schema=False).empty
    sql, params=conn.queries[0]
    assert "LEFT JOIN outcomes" not in sql
    assert "NULL::timestamptz AS delivered_at" in sql
    assert "batch_available_at <= %(as_of)s" in sql
    assert params["as_of"].isoformat()=="2018-05-01T02:59:59+00:00"


def test_label_query_bounds_receipt_provenance_and_separate_cutoff():
    conn=Connection()
    result=data.build_features(conn, ["o1"],
                               as_of="2018-04-30T23:59:59-03:00",
                               label_as_of="2018-06-29T23:59:59-03:00",
                               labeled=True,ensure_schema=False)
    assert result.empty
    sql,params=conn.queries[0]
    assert "LEFT JOIN outcomes" in sql
    assert "GREATEST(out.available_at" in sql
    assert "batch_receipts WHERE kind = 'outcomes'" in sql
    assert "%(label_as_of)s" in sql
    assert params["as_of"] < params["label_as_of"]


def test_label_cutoff_before_features_is_rejected():
    with pytest.raises(ValueError, match="label_as_of"):
        data.build_features(Connection(), ["o1"], as_of="2018-06-01T00:00:00Z",
                            label_as_of="2018-05-01T00:00:00Z",labeled=True,ensure_schema=False)
