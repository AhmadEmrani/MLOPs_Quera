"""Atomic write semantics without a live database; PostgreSQL integration is separate."""

import copy

import pytest

psycopg = pytest.importorskip("psycopg")

from promiseops import data  # noqa: E402


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.rowcount = 0
        self.one = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        self.rowcount = 0
        self.one = None
        if s.startswith("SELECT order_id, risk_probability, model_version FROM predictions"):
            req = params[0]
            record = self.db.predictions.get(req) or self.db.pending_predictions.get(req)
            self.one = record
            self.rowcount = int(record is not None)
        elif s.startswith("INSERT INTO predictions"):
            req, order, risk, version = params
            if req not in self.db.predictions and req not in self.db.pending_predictions:
                self.db.pending_predictions[req] = (order, risk, version)
                self.rowcount = 1
        elif s.startswith("INSERT INTO prediction_feature_observations"):
            if self.db.fail_observation:
                raise RuntimeError("injected snapshot write error")
            (req, order, version, at, contract, identity, feature_hash, feature_json) = params
            key = (order, version, contract, feature_hash)
            if key not in self.db.observations and key not in self.db.pending_observations:
                self.db.pending_observations[key] = dict(
                    request_id=req, as_of=at, source=identity, features=feature_json
                )
                self.rowcount = 1
        else:
            raise AssertionError(f"Unexpected SQL in mock: {s[:120]}")

    def fetchone(self):
        return self.one


class FakeDB:
    def __init__(self):
        self.predictions = {}
        self.observations = {}
        self.pending_predictions = {}
        self.pending_observations = {}
        self.fail_observation = False
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.predictions.update(self.pending_predictions)
        self.observations.update(self.pending_observations)
        self.pending_predictions.clear()
        self.pending_observations.clear()
        self.commits += 1

    def rollback(self):
        self.pending_predictions.clear()
        self.pending_observations.clear()
        self.rollbacks += 1


def feature_observation(value=120):
    from datetime import UTC, datetime
    import pandas as pd
    from promiseops.feature_lineage import observation
    row = {
        "item_count": 2, "seller_count": 1, "total_price": value,
        "total_freight": 20, "total_weight_g": 350, "payment_total": 140,
        "max_installments": 1, "promise_days": 8, "approval_hour": 17,
        "approval_weekday": 2, "interstate_fraction": 0.5,
        "customer_state": "SP", "primary_seller_state": "RJ",
        "primary_category": "tools", "primary_payment_type": "credit_card",
    }
    return observation(pd.DataFrame([row]),
                       feature_as_of=datetime(2026, 1, 1, tzinfo=UTC), release={})


def predict(db, req="r1", value=120, order="o1"):
    return data.get_or_save_prediction(
        db, request_id=req, order_id=order,
        compute_fn=lambda: {
            "risk_probability": 0.35,
            "model_version": "sha256:" + "f" * 64,
            "_feature_observation": feature_observation(value),
        },
    )


def test_insert_observation_atomic_and_no_public_metadata(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda db: pytest.fail("per-request DDL is forbidden"))
    db = FakeDB()
    response = predict(db)
    assert response == {
        "request_id": "r1", "order_id": "o1", "risk_probability": 0.35,
        "model_version": "sha256:" + "f" * 64,
    }
    assert len(db.predictions) == len(db.observations) == 1
    assert db.commits == 1
    assert "total_price" in next(iter(db.observations.values()))["features"]


def test_same_request_id_returns_old_result_and_skips_computation(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda db: pytest.fail("per-request DDL is forbidden"))
    db = FakeDB()
    first = predict(db, req="same", value=120)
    existing = data.get_or_save_prediction(
        db, request_id="same", order_id="o1",
        compute_fn=lambda: pytest.fail("idempotency replay must not recompute"),
    )
    assert first == existing
    assert len(db.observations) == 1


def test_new_request_same_order_same_features_deduplicates(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda db: pytest.fail("per-request DDL is forbidden"))
    db = FakeDB()
    predict(db, req="r1", value=120)
    predict(db, req="r2", value=120)
    assert len(db.predictions) == 2
    assert len(db.observations) == 1


def test_new_request_same_order_changed_features_versions_snapshot(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda db: pytest.fail("per-request DDL is forbidden"))
    db = FakeDB()
    predict(db, req="r1", value=120)
    predict(db, req="r2", value=121)
    assert len(db.observations) == 2


def test_failed_observation_rolls_back_prediction(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda db: pytest.fail("per-request DDL is forbidden"))
    db = FakeDB()
    db.fail_observation = True
    with pytest.raises(RuntimeError, match="snapshot write error"):
        predict(db)
    assert db.predictions == {}
    assert db.observations == {}
    assert db.rollbacks == 1


def test_request_id_conflict_does_not_mutate_observations(monkeypatch):
    monkeypatch.setattr(data, "init_db", lambda db: pytest.fail("per-request DDL is forbidden"))
    db = FakeDB()
    predict(db, req="same", order="o1")
    snapshot = copy.deepcopy(db.observations)
    with pytest.raises(data.ConflictError):
        predict(db, req="same", order="o2")
    assert db.observations == snapshot
