
"""Airflow orchestration for contract-checked feed ingestion."""

import json
import logging
import sys
from datetime import timedelta
from urllib.parse import urlsplit

import pendulum
from airflow.sdk import Param, dag, task


# ZIP package delivered through the team-code ConfigMap.
_PROMISEOPS_ZIP = "/opt/airflow/team-code/promiseops.zip"

if _PROMISEOPS_ZIP not in sys.path:
    sys.path.insert(0, _PROMISEOPS_ZIP)


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
            description=(
                "Required timezone-aware replay cutoff, "
                "for example 2018-11-05T00:00:00Z"
            ),
        )
    },
    tags=["promiseops", "ingestion"],
)
def promiseops_ingestion():

    @task(retries=0)
    def validate_cutoff() -> str:
        import logging
        import sys

        from pathlib import Path
        from zipfile import ZipFile
        from importlib.util import find_spec
        from airflow.sdk import get_current_context

        log = logging.getLogger("promiseops.ingestion")

        # =====================================================
        # DIAGNOSTIC: Inspect the actual Airflow task environment
        # =====================================================

        log.warning("========== PROMISEOPS DIAGNOSTIC START ==========")

        directories = (
            "/opt/airflow",
            "/opt/airflow/dags",
            "/opt/airflow/team-code",
        )

        for directory in directories:
            path = Path(directory)

            log.warning(
                "DIAGNOSTIC directory=%s exists=%s is_dir=%s",
                directory,
                path.exists(),
                path.is_dir(),
            )

            if path.is_dir():
                try:
                    entries = sorted(
                        item.name for item in path.iterdir()
                    )

                    log.warning(
                        "DIAGNOSTIC directory=%s entries=%s",
                        directory,
                        entries[:30],
                    )

                except OSError as error:
                    log.warning(
                        "DIAGNOSTIC directory=%s listing_error=%s",
                        directory,
                        type(error).__name__,
                    )

        files_to_check = (
            "/opt/airflow/dags/ingestion.py",
            "/opt/airflow/dags/training.py",
            "/opt/airflow/dags/promiseops.zip",
            "/opt/airflow/team-code/promiseops.zip",
        )

        for filename in files_to_check:
            path = Path(filename)

            log.warning(
                "DIAGNOSTIC file=%s exists=%s is_file=%s",
                filename,
                path.exists(),
                path.is_file(),
            )

        relevant_python_paths = [
            path
            for path in sys.path
            if "airflow" in path or "promiseops" in path
        ]

        log.warning(
            "DIAGNOSTIC relevant_sys_path=%s",
            relevant_python_paths,
        )

        log.warning("========== PROMISEOPS DIAGNOSTIC END ==========")

        # =====================================================
        # Verify the project ZIP inside the task environment
        # =====================================================

        archive_path = Path(
            "/opt/airflow/team-code/promiseops.zip"
        )

        if not archive_path.is_file():
            raise RuntimeError(
                "PromiseOps ZIP is not mounted in this "
                f"Airflow task: {archive_path}"
            )

        with ZipFile(archive_path) as archive:
            if "promiseops/contracts.py" not in archive.namelist():
                raise RuntimeError(
                    "promiseops/contracts.py is missing "
                    "from the mounted ZIP"
                )

        # Ensure Python imports from the expected ZIP.
        zip_location = str(archive_path)

        if zip_location in sys.path:
            sys.path.remove(zip_location)

        sys.path.insert(0, zip_location)

        spec = find_spec("promiseops")

        origin = (
            str(spec.origin)
            if spec is not None
            else "<not found>"
        )

        if not origin.startswith(zip_location + "/"):
            raise RuntimeError(
                "Airflow task resolved promiseops from "
                f"{origin}, not {zip_location}"
            )

        # =====================================================
        # Validate the requested ingestion cutoff
        # =====================================================

        from promiseops.contracts import instant

        raw_as_of = (
            get_current_context()["params"].get("as_of", "")
        )

        if (
            not isinstance(raw_as_of, str)
            or not raw_as_of.strip()
        ):
            raise ValueError(
                "Set the required timezone-aware "
                "'as_of' DAG parameter"
            )

        return (
            instant(raw_as_of)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @task(
        retries=2,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(hours=2),
    )
    def ingest_feed(as_of: str) -> dict:
        import sys

        from pathlib import Path
        from zipfile import ZipFile
        from importlib.util import find_spec

        from airflow.sdk import BaseHook
        from airflow.sdk.exceptions import AirflowFailException

        # =====================================================
        # Configure project imports inside the task
        # =====================================================

        archive_path = Path(
            "/opt/airflow/team-code/promiseops.zip"
        )

        if not archive_path.is_file():
            raise RuntimeError(
                "PromiseOps ZIP is not mounted in this "
                f"Airflow task: {archive_path}"
            )

        with ZipFile(archive_path) as archive:
            if "promiseops/contracts.py" not in archive.namelist():
                raise RuntimeError(
                    "promiseops/contracts.py is missing "
                    "from the mounted ZIP"
                )

        zip_location = str(archive_path)

        if zip_location in sys.path:
            sys.path.remove(zip_location)

        sys.path.insert(0, zip_location)

        spec = find_spec("promiseops")

        origin = (
            str(spec.origin)
            if spec is not None
            else "<not found>"
        )

        if not origin.startswith(zip_location + "/"):
            raise RuntimeError(
                "Airflow task resolved promiseops from "
                f"{origin}, not {zip_location}"
            )

        # =====================================================
        # Resolve Airflow connections
        # =====================================================

        from promiseops.data import get_connection, sync_feed

        feed_connection = BaseHook.get_connection(
            FEED_CONNECTION_ID
        )

        database_connection = BaseHook.get_connection(
            DATABASE_CONNECTION_ID
        )

        base_url = (
            feed_connection.extra_dejson.get("base_url")
            or feed_connection.host
        )

        token = feed_connection.password

        if not base_url:
            raise AirflowFailException(
                f"Airflow connection {FEED_CONNECTION_ID!r} "
                "must set host or extra.base_url"
            )

        parsed_base_url = urlsplit(base_url)

        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.netloc
            or parsed_base_url.username
            or parsed_base_url.password
        ):
            raise AirflowFailException(
                "Feed connection base_url must be an "
                "HTTP(S) origin without credentials"
            )

        if not token:
            raise AirflowFailException(
                f"Airflow connection {FEED_CONNECTION_ID!r} "
                "must store the bearer token as its password"
            )

        # =====================================================
        # Run transactional ingestion
        # =====================================================

        connection = get_connection(
            database_connection.get_uri()
        )

        try:
            try:
                receipts = sync_feed(
                    connection,
                    base_url=base_url,
                    token=token,
                    as_of=as_of,
                )

            except ValueError as error:
                raise AirflowFailException(
                    "Feed batch rejected during "
                    f"contract validation: {error}"
                ) from error

        finally:
            connection.close()

        # =====================================================
        # Report ingestion result
        # =====================================================

        summary = {
            "as_of": as_of,
            "batch_count": len(receipts),
            "committed_count": sum(
                receipt["status"] == "committed"
                for receipt in receipts
            ),
            "noop_count": sum(
                receipt["status"] == "noop"
                for receipt in receipts
            ),
            "batches": [
                {
                    "batch_id": receipt["batch_id"],
                    "status": receipt["status"],
                    "payload_sha256": receipt["payload_sha256"],
                }
                for receipt in receipts
            ],
        }

        LOGGER.info(
            "Feed ingestion completed: %s",
            json.dumps(summary, sort_keys=True),
        )

        return summary

    ingest_feed(validate_cutoff())


promiseops_ingestion()
