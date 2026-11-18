"""Manual, retry-safe ingestion of the company feed into team PostgreSQL."""

import json
import logging
import sys
from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, task

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
            description="Required timezone-aware replay cutoff, e.g. 2018-11-05T00:00:00Z",
        )
    },
    tags=["promiseops", "ingestion"],
)
def promiseops_ingestion():
    @task(retries=0)
    def validate_cutoff() -> str:
        # TaskFlow tasks run in separate processes in Airflow 3. The mounted
        # ZIP has already been confirmed at this location in g02 Airflow.
        import sys

        sys.path.insert(0, "/opt/airflow/dags/promiseops.zip")
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
        import sys

        import httpx
        from airflow.sdk.exceptions import AirflowFailException

        sys.path.insert(0, "/opt/airflow/dags/promiseops.zip")
        from promiseops.airflow_runtime import gateway_credentials, open_database_connection
        from promiseops.data import sync_feed

        base_url, token = gateway_credentials()
        LOGGER.info("Starting feed ingestion for as_of=%s", as_of)
        connection = open_database_connection()
        try:
            try:
                receipts = sync_feed(
                    connection,
                    base_url=base_url,
                    token=token,
                    as_of=as_of,
                )
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if 400 <= status < 500 and status != 429:
                    raise AirflowFailException(
                        f"Company feed rejected the request with HTTP {status}"
                    ) from None
                raise  # transient 429/5xx may retry
            except ValueError as error:
                # Validation or quarantine failure: retries should not repeatedly
                # ingest the same known-invalid batch.
                raise AirflowFailException(
                    f"Feed batch rejected during contract validation: {error}"
                ) from error
        finally:
            connection.close()

        summary = {
            "as_of": as_of,
            "batch_count": len(receipts),
            "committed_count": sum(r["status"] == "committed" for r in receipts),
            "noop_count": sum(r["status"] == "noop" for r in receipts),
            "batches": [
                {
                    "batch_id": r["batch_id"],
                    "status": r["status"],
                    "payload_sha256": r["payload_sha256"],
                }
                for r in receipts
            ],
        }
        LOGGER.info("Feed ingestion completed: %s", json.dumps(summary, sort_keys=True))
        return summary

    ingest_feed(validate_cutoff())


promiseops_ingestion()
