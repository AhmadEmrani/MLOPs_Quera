"""Tests for the score-batch CLI command."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from promiseops import cli
from promiseops.data import FEATURE_COLUMNS

RELEASE = {"model_version": "model-xyz789"}
AS_OF = "2025-01-10T12:00:00Z"
SINCE = "2025-01-09T12:00:00Z"
PASSWORD = "dev-pass-123"
URL = f"postgresql://user:{PASSWORD}@db/app"


class FakeModel:
    def __init__(self, scores=None, classes=(0, 1), default=0.5):
        self.classes_ = classes
        self.scores = scores or {}
        self.default = default
        self.columns = []

    def predict_proba(self, frame):
        self.columns.append(list(frame.columns))
        return [[1.0 - s, s] for s in (self.scores.get(str(o), self.default) for o in frame.index)]


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.queries.append((sql, params))

    def fetchall(self):
        return [(order_id,) for order_id in self.conn.ids]


class FakeConn:
    def __init__(self, ids):
        self.ids = ids
        self.queries = []
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


class Env:
    """Holds the fakes installed by ``install_fakes`` so tests can inspect them."""

    def __init__(self):
        self.conn = None
        self.model = None
        self.feature_calls = []


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("PROMISEOPS_DATABASE_URL", URL)
    monkeypatch.setenv("PROMISEOPS_RELEASE", "/fake/release")
    monkeypatch.delenv("PROMISEOPS_API_KEY", raising=False)
    return Env()


def install_fakes(env, monkeypatch, ids, scores=None, *, model=None, release=RELEASE, drop=()):
    env.conn = FakeConn(ids)
    env.model = model or FakeModel(scores)

    def fake_features(_conn, order_ids, **kwargs):
        env.feature_calls.append((list(order_ids), kwargs))
        keep = [o for o in order_ids if o not in drop]
        frame = pd.DataFrame(
            [{"order_id": o, **{name: 0.0 for name in FEATURE_COLUMNS}} for o in keep],
            columns=["order_id", *FEATURE_COLUMNS],
        )
        frame.index = frame["order_id"]
        return frame

    monkeypatch.setattr(cli, "load_bundle", lambda _path: (env.model, release))
    monkeypatch.setattr(cli, "get_connection", lambda _url: env.conn)
    monkeypatch.setattr(cli, "build_features", fake_features)


def run(capsys, as_of=AS_OF, since=SINCE, batch_id="b1"):
    code = cli.main(["score-batch", "--as-of", as_of, "--since", since, "--batch-id", batch_id])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_empty_cohort(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, [])
    code, out, _ = run(capsys)
    result = json.loads(out)
    assert code == 0
    assert result["cohort_count"] == 0
    assert result["scored_count"] == 0
    assert result["top_k"] == 0
    assert result["top"] == []
    assert result["cohort_hash"] is None
    assert env.feature_calls == []


def test_empty_cohort_does_not_need_class_one(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, [], model=FakeModel(classes=(0, 2)))
    assert run(capsys)[0] == 0


def test_single_order(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["ord-1"], {"ord-1": 0.82})
    code, out, err = run(capsys)
    result = json.loads(out)
    assert code == 0
    assert err == ""
    assert result["top"] == [{"order_id": "ord-1", "risk_probability": 0.82}]
    assert result["top_k"] == 1
    assert result["model_version"] == "model-xyz789"
    assert result["cohort_hash"]


@pytest.mark.parametrize(("n", "k"), [(1, 1), (9, 1), (10, 1), (11, 2), (30, 3), (100, 10), (101, 11)])
def test_top_k(env, monkeypatch, capsys, n, k):
    ids = [f"ord-{i:03d}" for i in range(n)]
    scores = {o: 0.01 + i * 0.001 for i, o in enumerate(ids)}
    install_fakes(env, monkeypatch, ids, scores)
    code, out, _ = run(capsys)
    result = json.loads(out)
    assert code == 0
    assert result["scored_count"] == n
    assert result["top_k"] == k
    top_ids = [t["order_id"] for t in result["top"]]
    assert top_ids == ids[::-1][:k]
    assert set(top_ids) == set(ids[::-1][:k])


def test_ties_sorted_by_order_id(env, monkeypatch, capsys):
    ids = ["ord-c", "ord-a", "ord-b"]
    install_fakes(env, monkeypatch, ids, {o: 0.7 for o in ids})
    result = json.loads(run(capsys)[1])
    assert result["top"][0]["order_id"] == "ord-a"


def test_same_input_same_output(env, monkeypatch, capsys):
    ids = [f"ord-{i:02d}" for i in range(15)]
    install_fakes(env, monkeypatch, ids)
    first = run(capsys)[1]
    second = run(capsys)[1]
    assert first == second


def test_large_cohort_uses_chunks(env, monkeypatch, capsys):
    ids = [f"ord-{i:04d}" for i in range(1200)]
    scores = {o: i / 2000 for i, o in enumerate(ids)}
    install_fakes(env, monkeypatch, ids, scores)
    result = json.loads(run(capsys)[1])
    assert [len(c[0]) for c in env.feature_calls] == [500, 500, 200]
    assert result["top_k"] == 120
    assert result["top"][0]["order_id"] == "ord-1199"


def test_cohort_hash_depends_on_ids(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a", "b"])
    first = json.loads(run(capsys)[1])["cohort_hash"]
    install_fakes(env, monkeypatch, ["a", "c"])
    second = json.loads(run(capsys)[1])["cohort_hash"]
    again = json.loads(run(capsys)[1])["cohort_hash"]
    assert first != second
    assert second == again


def test_query_window_and_feature_arguments(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["ord-1"])
    run(capsys, as_of="2025-01-10T16:00:00+03:30", since=SINCE)
    sql, params = env.conn.queries[0]
    assert "batch_available_at > %s" in sql
    assert "batch_available_at <= %s" in sql
    since, as_of = params
    assert since.isoformat() == "2025-01-09T12:00:00+00:00"
    assert as_of.isoformat() == "2025-01-10T12:30:00+00:00"
    _ids, kwargs = env.feature_calls[0]
    assert kwargs["as_of"] == as_of
    assert kwargs["ensure_schema"] is False
    assert env.model.columns[0] == list(FEATURE_COLUMNS)


def test_output_is_utc(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["ord-1"])
    result = json.loads(run(capsys, as_of="2025-01-10T16:00:00+03:30")[1])
    assert result["as_of"] == "2025-01-10T12:30:00+00:00"


def test_output_is_single_json_line(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["ord-1"])
    _code, out, err = run(capsys)
    assert out.count("\n") == 1
    assert out.endswith("\n")
    assert err == ""


def test_missing_features_fail(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a", "b", "c"], drop={"b"})
    code, out, err = run(capsys)
    assert code == 1
    assert out == ""
    assert "ValueError" in err


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_bad_probability_fails(env, monkeypatch, capsys, bad):
    install_fakes(env, monkeypatch, ["a"], {"a": bad})
    code, out, _ = run(capsys)
    assert code == 1
    assert out == ""


def test_model_without_positive_class_fails(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a"], model=FakeModel(classes=(0, 2)))
    assert run(capsys)[0] == 1


@pytest.mark.parametrize("release", [{}, {"model_version": ""}, {"model_version": "  "}])
def test_bad_model_version_fails(env, monkeypatch, capsys, release):
    install_fakes(env, monkeypatch, ["a"], release=release)
    assert run(capsys)[0] == 1


def test_missing_database_url(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a"])
    monkeypatch.delenv("PROMISEOPS_DATABASE_URL")
    code, out, err = run(capsys)
    assert code == 1
    assert out == ""
    assert "PROMISEOPS_DATABASE_URL" in err


def test_api_key_not_required(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a"])
    assert run(capsys)[0] == 0


def test_connection_closed_on_error(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a"], {"a": 5.0})
    assert run(capsys)[0] == 1
    assert env.conn.closed is True


def test_connection_closed_on_success(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a"])
    run(capsys)
    assert env.conn.closed is True


def test_connection_error_hides_password(env, monkeypatch, capsys):
    install_fakes(env, monkeypatch, ["a"])

    def boom(url):
        raise RuntimeError(f"could not connect to {url}")

    monkeypatch.setattr(cli, "get_connection", boom)
    code, out, err = run(capsys)
    assert code == 1
    assert "RuntimeError" in err
    assert PASSWORD not in err
    assert URL not in err
    assert PASSWORD not in out


def test_since_must_be_before_as_of(env, capsys):
    for since in (AS_OF, "2025-01-11T12:00:00Z"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["score-batch", "--as-of", AS_OF, "--since", since, "--batch-id", "b"])
        assert exc.value.code == 2


def test_timestamp_needs_timezone(env):
    argv = ["score-batch", "--as-of", "2025-01-10T12:00:00", "--since", SINCE, "--batch-id", "b"]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("missing", ["--as-of", "--since", "--batch-id"])
def test_missing_arguments(env, missing):
    full = {"--as-of": AS_OF, "--since": SINCE, "--batch-id": "b"}
    argv = ["score-batch"]
    for name, value in full.items():
        if name != missing:
            argv += [name, value]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "a;rm", "نام"])
def test_invalid_batch_id(env, bad):
    argv = ["score-batch", "--as-of", AS_OF, "--since", SINCE, "--batch-id", bad]
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_release_dir_file_and_dir(tmp_path, monkeypatch):
    f = tmp_path / "release.json"
    f.write_text("{}")
    monkeypatch.setenv("PROMISEOPS_RELEASE", str(f))
    assert cli._release_dir() == tmp_path
    monkeypatch.setenv("PROMISEOPS_RELEASE", str(tmp_path))
    assert cli._release_dir() == tmp_path
