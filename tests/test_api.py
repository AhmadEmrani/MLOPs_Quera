import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from promiseops import api
from promiseops.api import services
from promiseops.api import state as api_state
from promiseops.data import FEATURE_COLUMNS, ConflictError
from promiseops.settings import Settings

KEY = "s3cret-key"
DB_URL = "postgresql://user:dbpass@db/app"
VERSION = "model-abc123"


class FakeModel:
    classes_ = (0, 1)

    def predict_proba(self, frame):
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        return [[0.25, 0.75]] * len(frame)


class FakeCursor:
    def __init__(self, fail):
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.fail:
            raise RuntimeError("SELECT secret FROM hidden_table failed")


class FakeConn:
    def __init__(self, fail=False):
        self.fail = fail

    def cursor(self):
        return FakeCursor(self.fail)

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch):
    store = {}
    known = {"order-1", "order-2"}
    calls = {"init": 0, "compute": 0}

    def fake_init(_conn):
        calls["init"] += 1

    def fake_features(_conn, order_ids, *, as_of, labeled=False):
        ids = [o for o in order_ids if o in known]
        row = {name: 1 for name in FEATURE_COLUMNS}
        row["primary_category"] = None
        return pd.DataFrame([{"order_id": o, **row} for o in ids])

    def fake_save(_conn, *, request_id, order_id, compute_fn):
        if request_id in store:
            saved = store[request_id]
            if saved["order_id"] != order_id:
                raise ConflictError("conflict")
            return saved
        calls["compute"] += 1
        result = compute_fn()
        store[request_id] = {"order_id": order_id, "request_id": request_id, **result}
        return store[request_id]

    monkeypatch.setattr(api_state, "init_db", fake_init)
    monkeypatch.setattr(services, "build_features", fake_features)
    monkeypatch.setattr(services, "get_or_save_prediction", fake_save)
    return calls


def make_client(model=True, conn=None, loader=None):
    settings = Settings(database_url=DB_URL, api_key=KEY)
    if loader is None:

        def loader(_path):
            if not model:
                raise ValueError("no bundle")
            return FakeModel(), {"model_version": VERSION}

    app = api.create_app(settings, bundle_loader=loader, connection_factory=lambda: conn or FakeConn())
    return TestClient(app, raise_server_exceptions=False)


AUTH = {"X-API-Key": KEY}


def test_health_and_ready_need_no_key(env):
    with make_client() as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
    assert env["init"] >= 1


def test_ready_503_without_model(env):
    with make_client(model=False) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503


def test_ready_503_when_db_down(env):
    with make_client(conn=FakeConn(fail=True)) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert "hidden_table" not in response.text


def test_predict_requires_key(env):
    with make_client() as client:
        body = {"order_id": "order-1"}
        assert client.post("/predict", json=body).status_code == 401
        assert client.post("/predict", json=body, headers={"X-API-Key": "wrong"}).status_code == 401


def test_predict_success_and_idempotent(env):
    with make_client() as client:
        body = {"order_id": "order-1", "request_id": "req-1"}
        first = client.post("/predict", json=body, headers=AUTH)
        again = client.post("/predict", json=body, headers=AUTH)
    assert first.status_code == 200
    data = first.json()
    assert data == {
        "order_id": "order-1",
        "request_id": "req-1",
        "risk_probability": 0.75,
        "model_version": VERSION,
    }
    assert again.json() == data
    assert env["compute"] == 1


def test_request_id_generated_when_absent(env):
    with make_client() as client:
        response = client.post("/predict", json={"order_id": "order-1"}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["request_id"]


def test_same_request_id_other_order_is_409(env):
    with make_client() as client:
        client.post("/predict", json={"order_id": "order-1", "request_id": "r"}, headers=AUTH)
        response = client.post("/predict", json={"order_id": "order-2", "request_id": "r"}, headers=AUTH)
    assert response.status_code == 409


def test_unknown_order_is_404(env):
    with make_client() as client:
        response = client.post("/predict", json={"order_id": "nope"}, headers=AUTH)
    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"order_id": ""},
        {"order_id": "bad id!"},
        {"order_id": "o", "extra": 1},
        {"order_id": "o", "request_id": "x y"},
    ],
)
def test_invalid_input_is_422_generic(env, body):
    with make_client() as client:
        response = client.post("/predict", json=body, headers=AUTH)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid request"}


def test_malformed_json_is_422(env):
    with make_client() as client:
        response = client.post(
            "/predict", content=b"{nope", headers={**AUTH, "Content-Type": "application/json"}
        )
    assert response.status_code == 422


def test_no_model_is_503(env):
    with make_client(model=False) as client:
        response = client.post("/predict", json={"order_id": "order-1"}, headers=AUTH)
    assert response.status_code == 503


def test_internal_error_hides_details(env, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("SELECT * FROM orders; traceback")

    monkeypatch.setattr(services, "get_or_save_prediction", boom)
    with make_client() as client:
        response = client.post("/predict", json={"order_id": "order-1"}, headers=AUTH)
    assert response.status_code == 500
    assert "SELECT" not in response.text and "traceback" not in response.text


def test_metrics_need_key_and_have_bounded_labels(env):
    with make_client() as client:
        assert client.get("/metrics").status_code == 401
        client.post("/predict", json={"order_id": "order-1", "request_id": "secret-req-id"}, headers=AUTH)
        client.get("/does-not-exist-order-1")
        response = client.get("/metrics", headers=AUTH)
    assert response.status_code == 200
    text = response.text
    assert 'route="/predict"' in text
    assert 'route="unmatched"' in text
    assert 'feature="primary_category"' in text
    for forbidden in ("order-1", "secret-req-id", KEY, "dbpass"):
        assert forbidden not in text


def test_secrets_not_logged(env, caplog):
    caplog.set_level(logging.DEBUG)
    with make_client(model=False) as client:
        client.post("/predict", json={"order_id": "order-1"}, headers={"X-API-Key": "wrong-guess"})
    assert KEY not in caplog.text and "dbpass" not in caplog.text


def test_release_path_parent_is_passed_to_loader(env):
    seen = []

    def loader(path):
        seen.append(path)
        return FakeModel(), {"model_version": VERSION}

    with make_client(loader=loader):
        pass
    assert str(seen[0]) == "/opt/promiseops/release"


def test_versioned_predict_matches_contract_path(env):
    body = {"order_id": "order-1", "request_id": "req-v1"}
    with make_client() as client:
        legacy = client.post("/predict", json=body, headers=AUTH)
        versioned = client.post("/v1/predict", json=body, headers=AUTH)
        assert client.post("/v1/predict", json=body).status_code == 401
    assert legacy.status_code == versioned.status_code == 200
    assert legacy.json() == versioned.json()
