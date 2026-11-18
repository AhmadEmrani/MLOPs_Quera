"""Resolve the staff-provided PromiseOps runtime configuration for Airflow tasks.

No credential values are logged, stored in XCom, or copied into GitLab CI.
All Airflow imports happen at call time so importing the application package
outside Airflow does not depend on Airflow being installed.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit


def _airflow_connection(conn_id: str):
    from airflow.sdk import BaseHook
    from airflow.sdk.exceptions import AirflowNotFoundException

    try:
        return BaseHook.get_connection(conn_id)
    except AirflowNotFoundException:
        return None


def gateway_credentials() -> tuple[str, str]:
    """Return (HTTP(S) origin, bearer key) without exposing either in logs.

    Staff injects COMPANY_GATEWAY_URL and COMPANY_FEED_KEY in the Airflow task
    environment. A complete Airflow HTTP connection remains supported.
    """
    from airflow.sdk.exceptions import AirflowFailException

    env_origin = os.environ.get("COMPANY_GATEWAY_URL", "").strip()
    env_key = os.environ.get("COMPANY_FEED_KEY", "").strip()

    # Prefer the complete pair injected by the platform, and avoid depending
    # on a possibly placeholder Airflow connection in that case.
    if env_origin and env_key:
        origin, key = env_origin, env_key
    else:
        conn = _airflow_connection("promiseops_feed")
        conn_origin = ""
        conn_key = ""
        if conn is not None:
            conn_origin = (conn.extra_dejson.get("base_url") or conn.host or "").strip()
            conn_key = (conn.password or "").strip()

        if conn_origin and conn_key:
            origin, key = conn_origin, conn_key
        else:
            origin = env_origin or conn_origin
            key = env_key or conn_key

    if not origin or not key:
        raise AirflowFailException(
            "Gateway configuration is incomplete: provide COMPANY_GATEWAY_URL "
            "and COMPANY_FEED_KEY in the Airflow task environment, or a full "
            "promiseops_feed connection (extra.base_url/host and password)"
        )

    url = urlsplit(origin)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise AirflowFailException(
            "Gateway URL must be an HTTP(S) origin without path, query, or credentials"
        )

    return origin.rstrip("/"), key


def open_database_connection():
    """Open psycopg connection using staff env, falling back to Airflow fields.

    Avoid passing Airflow's get_uri() to psycopg: its connection URI is an
    Airflow serialization format, not necessarily a PostgreSQL libpq DSN.
    """
    from airflow.sdk.exceptions import AirflowFailException
    from promiseops.data import get_connection

    env_url = os.environ.get("PROMISEOPS_DATABASE_URL", "").strip()
    if env_url:
        return get_connection(env_url)

    conn = _airflow_connection("promiseops_postgres")
    if conn is None or not conn.host or not conn.schema:
        raise AirflowFailException(
            "Database configuration is incomplete: set PROMISEOPS_DATABASE_URL "
            "or configure promiseops_postgres with host and database/schema"
        )
    if conn.conn_type not in ("postgres", "postgresql"):
        raise AirflowFailException("promiseops_postgres must have PostgreSQL connection type")

    import psycopg

    return psycopg.connect(
        host=conn.host,
        port=conn.port or 5432,
        dbname=conn.schema,
        user=conn.login or None,
        password=conn.password or None,
    )
