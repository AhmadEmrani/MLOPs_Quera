"""No real PostgreSQL access, no modification of persistent application data."""

from datetime import UTC, datetime

import pytest

from ci.select_load_orders import (
    SELECT_ORDERS_SQL,
    cohort_hash,
    parse_aware_datetime,
    select_order_ids,
    validate_selection,
)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query, self.params = query, params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self._cursor = FakeCursor(rows)

    def cursor(self):
        return self._cursor


def test_cutoffs_respect_brazil_offset():
    end = parse_aware_datetime("2018-04-30T23:59:59-03:00")
    as_of = parse_aware_datetime("2018-06-29T23:59:59-03:00")
    assert end == datetime(2018, 5, 1, 2, 59, 59, tzinfo=UTC)
    assert end < as_of
    validate_selection(end, as_of, 200, 20)


def test_rejects_naive_or_reversed_windows():
    with pytest.raises(ValueError, match="timezone"):
        parse_aware_datetime("2018-06-29T23:59:59")
    cutoff = parse_aware_datetime("2018-06-29T23:59:59-03:00")
    with pytest.raises(ValueError, match="strictly after"):
        validate_selection(cutoff, cutoff, 200, 20)
    with pytest.raises(ValueError, match="500"):
        validate_selection(cutoff, cutoff.replace(year=2019), 501, 20)


def test_uses_parameterized_released_post_validation_selection():
    start = parse_aware_datetime("2018-04-30T23:59:59-03:00")
    finish = parse_aware_datetime("2018-06-29T23:59:59-03:00")
    conn = FakeConnection([("ord-a",), ("ord-b",)])
    assert select_order_ids(conn, start, finish, 200) == ["ord-a", "ord-b"]
    assert conn._cursor.query == SELECT_ORDERS_SQL
    assert conn._cursor.params == {"validation_end": start, "as_of": finish, "limit": 200}
    assert "o.approved_at > %(validation_end)s" in conn._cursor.query
    assert "o.batch_available_at <= %(as_of)s" in conn._cursor.query
    assert "outcomes" not in conn._cursor.query.lower()
    assert "SELECT" in conn._cursor.query
    assert "DELETE" not in conn._cursor.query
    assert "UPDATE" not in conn._cursor.query


def test_rejects_duplicate_or_invalid_order_ids():
    date = datetime(2018, 1, 1, tzinfo=UTC)
    later = datetime(2018, 2, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="duplicate"):
        select_order_ids(FakeConnection([("dup",), ("dup",)]), date, later, 10)
    with pytest.raises(ValueError, match="invalid"):
        select_order_ids(FakeConnection([(None,)]), date, later, 10)


def test_cohort_hash_does_not_depend_on_row_order():
    assert cohort_hash(["abc", "def"]) == cohort_hash(["def", "abc"])
    assert len(cohort_hash(["abc", "def"])) == 64


def test_order_cap():
    date = datetime(2018, 1, 1, tzinfo=UTC)
    later = datetime(2018, 2, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="limit"):
        select_order_ids(FakeConnection([("a",), ("b",)]), date, later, 1)


def test_ci_installer_is_idempotent_and_preserves_other_jobs():
    from ci.install_load_job import add_include, ENTRY

    base = 'stages: [test, inspect]\ninclude:\n  - local: "/.gitlab/ci/test.yml"\nother: thing\n'
    result = add_include(base)
    assert ENTRY in result
    assert result.index(ENTRY) < result.index('other: thing')
    assert add_include(result) == result
