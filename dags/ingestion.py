"""Airflow orchestration for contract-checked feed ingestion."""

import json
import logging
from datetime import timedelta
from urllib.parse import urlsplit

import pendulum
from airflow.sdk import Param, dag, task

FEED_CONNECTION_ID = "promiseops_feed"
DATABASE_CONNECTION_ID = "promiseops_postgres"
LOGGER = logging.getLogger(__name__)


@dag(
    dag_id="promiseops_ingestion",
    description="Fetch, validate, and transactionally ingest PromiseOps feed batches",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "as_of": Param(
            "",
            type="string",
            description="Required timezone-aware replay cutoff, for example 2018-11-05T00:00:00Z",
        )
    },
    tags=["promiseops", "ingestion"],
)
def promiseops_ingestion():
    @task(retries=0)
    def validate_cutoff() -> str:
        from airflow.sdk import get_current_context

        from promiseops.contracts import instant

        raw_as_of = get_current_context()["params"].get("as_of", "")
        if not isinstance(raw_as_of, str) or not raw_as_of.strip():
            raise ValueError("Set the required timezone-aware 'as_of' DAG parameter")

        return instant(raw_as_of).isoformat().replace("+00:00", "Z")

    @task(
        retries=2,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(hours=2),
    )
    def ingest_feed(as_of: str) -> dict:
        from airflow.sdk import BaseHook
        from airflow.sdk.exceptions import AirflowFailException

        from promiseops.data import get_connection, sync_feed

        feed_connection = BaseHook.get_connection(FEED_CONNECTION_ID)
        database_connection = BaseHook.get_connection(DATABASE_CONNECTION_ID)
        base_url = feed_connection.extra_dejson.get("base_url") or feed_connection.host
        token = feed_connection.password

        if not base_url:
            raise AirflowFailException(
                f"Airflow connection {FEED_CONNECTION_ID!r} must set host or extra.base_url"
            )
        parsed_base_url = urlsplit(base_url)
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.netloc
            or parsed_base_url.username
            or parsed_base_url.password
        ):
            raise AirflowFailException("Feed connection base_url must be an HTTP(S) origin without credentials")
        if not token:
            raise AirflowFailException(
                f"Airflow connection {FEED_CONNECTION_ID!r} must store the bearer token as its password"
            )

        connection = get_connection(database_connection.get_uri())
        try:
            try:
                receipts = sync_feed(connection, base_url=base_url, token=token, as_of=as_of)
            except ValueError as error:
                raise AirflowFailException(f"Feed batch rejected during contract validation: {error}") from error
        finally:
            connection.close()

        summary = {
            "as_of": as_of,
            "batch_count": len(receipts),
            "committed_count": sum(receipt["status"] == "committed" for receipt in receipts),
            "noop_count": sum(receipt["status"] == "noop" for receipt in receipts),
            "batches": [
                {
                    "batch_id": receipt["batch_id"],
                    "status": receipt["status"],
                    "payload_sha256": receipt["payload_sha256"],
                }
                for receipt in receipts
            ],
        }
        LOGGER.info("Feed ingestion completed: %s", json.dumps(summary, sort_keys=True))
        return summary

    ingest_feed(validate_cutoff())


promiseops_ingestion()
