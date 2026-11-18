"""Unit tests for secret-safe Airflow runtime configuration resolution."""

import sys
import types
from types import SimpleNamespace

import pytest

from promiseops import airflow_runtime


@pytest.fixture
def fake_airflow(monkeypatch):
    class AirflowNotFoundException(Exception):
        pass

    class AirflowFailException(Exception):
        pass

    sdk = types.ModuleType("airflow.sdk")
    errors = types.ModuleType("airflow.sdk.exceptions")
    errors.AirflowNotFoundException = AirflowNotFoundException
    errors.AirflowFailException = AirflowFailException
    state = {"connection": None, "calls": []}

    class BaseHook:
        @staticmethod
        def get_connection(conn_id):
            state["calls"].append(conn_id)
            if state["connection"] is None:
                raise AirflowNotFoundException("not found")
            return state["connection"]

    sdk.BaseHook = BaseHook
    monkeypatch.setitem(sys.modules, "airflow", types.ModuleType("airflow"))
    monkeypatch.setitem(sys.modules, "airflow.sdk", sdk)
    monkeypatch.setitem(sys.modules, "airflow.sdk.exceptions", errors)
    return state, AirflowFailException


def test_platform_gateway_env_takes_priority(monkeypatch, fake_airflow):
    state, _ = fake_airflow
    monkeypatch.setenv("COMPANY_GATEWAY_URL", "https://gateway.example.test/")
    monkeypatch.setenv("COMPANY_FEED_KEY", "private-token")
    assert airflow_runtime.gateway_credentials() == (
        "https://gateway.example.test", "private-token"
    )
    assert state["calls"] == []


def test_gateway_falls_back_to_airflow_connection(monkeypatch, fake_airflow):
    state, _ = fake_airflow
    monkeypatch.delenv("COMPANY_GATEWAY_URL", raising=False)
    monkeypatch.delenv("COMPANY_FEED_KEY", raising=False)
    state["connection"] = SimpleNamespace(
        extra_dejson={"base_url": "http://internal.example:8080"},
        host="", password="connection-key",
    )
    assert airflow_runtime.gateway_credentials() == (
        "http://internal.example:8080", "connection-key"
    )


def test_partial_placeholder_connection_uses_env(monkeypatch, fake_airflow):
    state, _ = fake_airflow
    monkeypatch.setenv("COMPANY_GATEWAY_URL", "http://company.example")
    monkeypatch.setenv("COMPANY_FEED_KEY", "real-key")
    state["connection"] = SimpleNamespace(extra_dejson={}, host="", password="")
    assert airflow_runtime.gateway_credentials()[0] == "http://company.example"
    assert state["calls"] == []


def test_gateway_rejects_url_containing_password(monkeypatch, fake_airflow):
    _, error_type = fake_airflow
    monkeypatch.setenv("COMPANY_GATEWAY_URL", "https://user:password@gateway.example.test")
    monkeypatch.setenv("COMPANY_FEED_KEY", "private-token")
    with pytest.raises(error_type, match="HTTP\\(S\\) origin"):
        airflow_runtime.gateway_credentials()


def test_database_prefers_staff_env(monkeypatch, fake_airflow):
    state, _ = fake_airflow
    sentinel = object()
    expected_url = "postgresql://example"
    data = types.ModuleType("promiseops.data")
    data.get_connection = lambda url: sentinel if url == expected_url else None
    monkeypatch.setitem(sys.modules, "promiseops.data", data)
    monkeypatch.setenv("PROMISEOPS_DATABASE_URL", expected_url)
    assert airflow_runtime.open_database_connection() is sentinel
    assert state["calls"] == []


def test_database_airflow_connection_fallback(monkeypatch, fake_airflow):
    state, _ = fake_airflow
    monkeypatch.delenv("PROMISEOPS_DATABASE_URL", raising=False)
    state["connection"] = SimpleNamespace(
        conn_type="postgres", host="postgres.example", schema="promiseops_feed",
        port=5432, login="test", password="secret",
    )
    data = types.ModuleType("promiseops.data")
    data.get_connection = lambda url: None
    monkeypatch.setitem(sys.modules, "promiseops.data", data)
    pg = types.ModuleType("psycopg")
    sentinel = object()
    pg.connect = lambda **kwargs: sentinel if kwargs["dbname"] == "promiseops_feed" else None
    monkeypatch.setitem(sys.modules, "psycopg", pg)
    assert airflow_runtime.open_database_connection() is sentinel
    assert state["calls"] == ["promiseops_postgres"]
