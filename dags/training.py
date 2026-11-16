"""Airflow orchestration for queued candidate training."""

import hashlib
import json
import sys

# ConfigMap mounts the archive; Python must search the ZIP itself, not just its parent.
_PROMISEOPS_ZIP = "/opt/airflow/team-code/promiseops.zip"
if _PROMISEOPS_ZIP not in sys.path:
    sys.path.insert(0, _PROMISEOPS_ZIP)
import logging
import re
from datetime import timedelta
from urllib.parse import urlsplit

import pendulum
from airflow.sdk import Param, dag, task

FEED_CONNECTION_ID = "promiseops_feed"
DATABASE_CONNECTION_ID = "promiseops_postgres"
KUBERNETES_CONNECTION_ID = "promiseops_jobs_kubernetes"
LOGGER = logging.getLogger(__name__)
_IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_JOB_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$|^[a-z0-9]$")


@dag(
    dag_id="promiseops_training",
    description="Verify ingested data, queue candidate training, and observe the worker Job",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "worker_image": Param("", type="string", description="Own worker image pinned by @sha256 digest"),
        "source_commit": Param("", type="string", description="Full 40-character lowercase source commit SHA"),
        "train_end": Param("", type="string", description="Timezone-aware training cutoff"),
        "validation_end": Param("", type="string", description="Timezone-aware validation cutoff"),
        "as_of": Param("", type="string", description="Timezone-aware data and label availability cutoff"),
    },
    tags=["promiseops", "training", "mlflow"],
)
def promiseops_training():
    @task(retries=0)
    def validate_request() -> dict:
        from airflow.sdk import get_current_context

        # Airflow 3 runs TaskFlow functions in an independent task process.
        # Set the ZIP path *inside* the task, not only at DAG parse time.
        import sys
        from pathlib import Path
        from zipfile import ZipFile

        archive_path = Path("/opt/airflow/team-code/promiseops.zip")
        if not archive_path.is_file():
            raise RuntimeError(
                f"PromiseOps ZIP is not mounted in this Airflow task: {archive_path}"
            )
        with ZipFile(archive_path) as archive:
            if "promiseops/contracts.py" not in archive.namelist():
                raise RuntimeError("promiseops/contracts.py is missing from the mounted ZIP")
        # Fail clearly if another installed/stale package shadows the ZIP.
        zip_location = str(archive_path)
        if zip_location in sys.path:
            sys.path.remove(zip_location)
        sys.path.insert(0, zip_location)
        from importlib.util import find_spec

        spec = find_spec("promiseops")
        origin = str(spec.origin) if spec is not None else "<not found>"
        if not origin.startswith(zip_location + "/"):
            raise RuntimeError(
                f"Airflow task resolved promiseops from {origin}, not {zip_location}"
            )

        from promiseops.contracts import instant

        context = get_current_context()
        params = context["params"]
        required = ("worker_image", "source_commit", "train_end", "validation_end", "as_of")
        missing = [name for name in required if not isinstance(params.get(name), str) or not params[name].strip()]
        if missing:
            raise ValueError(f"Set required training parameters: {', '.join(missing)}")

        worker_image = params["worker_image"].strip()
        source_commit = params["source_commit"].strip()
        if not _IMAGE_DIGEST_PATTERN.fullmatch(worker_image):
            raise ValueError("worker_image must be an immutable image reference ending in @sha256:<64 lowercase hex>")
        if not _COMMIT_PATTERN.fullmatch(source_commit):
            raise ValueError("source_commit must be a full 40-character lowercase hexadecimal SHA")

        cutoffs = {name: instant(params[name]) for name in ("train_end", "validation_end", "as_of")}
        if not cutoffs["train_end"] < cutoffs["validation_end"] <= cutoffs["as_of"]:
            raise ValueError("Training cutoffs must satisfy train_end < validation_end <= as_of")

        dag_run = context.get("dag_run")
        if dag_run is None or not dag_run.run_id:
            raise ValueError("A DagRun ID is required to create a retry-stable training request ID")
        request_id = "train-" + hashlib.sha256(dag_run.run_id.encode("utf-8")).hexdigest()

        return {
            "worker_image": worker_image,
            "source_commit": source_commit,
            "train_end": cutoffs["train_end"].isoformat().replace("+00:00", "Z"),
            "validation_end": cutoffs["validation_end"].isoformat().replace("+00:00", "Z"),
            "as_of": cutoffs["as_of"].isoformat().replace("+00:00", "Z"),
            "mode": "train",
            "request_id": request_id,
        }

    @task(
        retries=2,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=10),
    )
    def verify_ingestion(as_of: str) -> dict:
        import httpx
        from airflow.sdk import BaseHook
        from airflow.sdk.exceptions import AirflowFailException

        # Airflow 3 runs TaskFlow functions in an independent task process.
        # Set the ZIP path *inside* the task, not only at DAG parse time.
        import sys
        from pathlib import Path
        from zipfile import ZipFile

        archive_path = Path("/opt/airflow/team-code/promiseops.zip")
        if not archive_path.is_file():
            raise RuntimeError(
                f"PromiseOps ZIP is not mounted in this Airflow task: {archive_path}"
            )
        with ZipFile(archive_path) as archive:
            if "promiseops/contracts.py" not in archive.namelist():
                raise RuntimeError("promiseops/contracts.py is missing from the mounted ZIP")
        # Fail clearly if another installed/stale package shadows the ZIP.
        zip_location = str(archive_path)
        if zip_location in sys.path:
            sys.path.remove(zip_location)
        sys.path.insert(0, zip_location)
        from importlib.util import find_spec

        spec = find_spec("promiseops")
        origin = str(spec.origin) if spec is not None else "<not found>"
        if not origin.startswith(zip_location + "/"):
            raise RuntimeError(
                f"Airflow task resolved promiseops from {origin}, not {zip_location}"
            )

        from promiseops.contracts import instant
        from promiseops.data import get_connection

        feed_connection = BaseHook.get_connection(FEED_CONNECTION_ID)
        database_connection = BaseHook.get_connection(DATABASE_CONNECTION_ID)
        base_url = feed_connection.extra_dejson.get("base_url") or feed_connection.host
        token = feed_connection.password
        parsed_base_url = urlsplit(base_url or "")
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.netloc
            or parsed_base_url.username
            or parsed_base_url.password
        ):
            raise AirflowFailException("Feed connection must provide an HTTP(S) base URL without credentials")
        if not token:
            raise AirflowFailException("Feed connection must store the bearer token as its password")

        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as client:
            response = client.get("/feed", params={"as_of": as_of})
            if 400 <= response.status_code < 500:
                raise AirflowFailException(f"Feed manifest request was rejected: HTTP {response.status_code}")
            response.raise_for_status()
            manifest = response.json()

        if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
            raise AirflowFailException("Feed response must contain an entries list and echoed as_of")
        if not isinstance(manifest.get("as_of"), str) or instant(manifest["as_of"]) != instant(as_of):
            raise AirflowFailException("Feed response as_of does not match the requested training cutoff")

        entries = manifest["entries"]
        if not entries:
            raise AirflowFailException("Feed returned no batch entries for the selected as_of")

        expected_hashes = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise AirflowFailException("Feed manifest contains an invalid entry")
            batch_id = entry.get("batch_id")
            payload_sha256 = entry.get("payload_sha256")
            if not isinstance(batch_id, str) or not isinstance(payload_sha256, str):
                raise AirflowFailException("Feed manifest entries must contain batch_id and payload_sha256")
            if batch_id in expected_hashes:
                raise AirflowFailException(f"Feed manifest repeats batch ID {batch_id!r}")
            expected_hashes[batch_id] = payload_sha256

        connection = get_connection(database_connection.get_uri())
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT batch_id, payload_sha256 FROM batch_receipts WHERE batch_id = ANY(%s)",
                    (list(expected_hashes),),
                )
                stored_hashes = {batch_id: payload_hash.strip() for batch_id, payload_hash in cursor.fetchall()}
        finally:
            connection.close()

        missing = sorted(set(expected_hashes) - set(stored_hashes))
        mismatched = sorted(
            batch_id
            for batch_id, expected_hash in expected_hashes.items()
            if batch_id in stored_hashes and stored_hashes[batch_id] != expected_hash
        )
        if missing or mismatched:
            raise AirflowFailException(
                "Ingestion is incomplete for this as_of; "
                f"missing_batches={missing[:20]}, checksum_mismatches={mismatched[:20]}"
            )

        summary = {"as_of": as_of, "verified_batch_count": len(expected_hashes)}
        LOGGER.info("Verified ingested feed snapshot: %s", json.dumps(summary, sort_keys=True))
        return summary

    @task(
        retries=2,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=2),
    )
    def submit_training(request: dict, ingestion: dict) -> dict:
        import httpx
        from airflow.sdk import BaseHook
        from airflow.sdk.exceptions import AirflowFailException

        if ingestion.get("as_of") != request["as_of"]:
            raise AirflowFailException("Ingestion verification cutoff does not match training as_of")

        feed_connection = BaseHook.get_connection(FEED_CONNECTION_ID)
        base_url = feed_connection.extra_dejson.get("base_url") or feed_connection.host
        token = feed_connection.password
        parsed_base_url = urlsplit(base_url or "")
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.netloc
            or parsed_base_url.username
            or parsed_base_url.password
        ):
            raise AirflowFailException("Feed connection must provide an HTTP(S) base URL without credentials")
        if not token:
            raise AirflowFailException("Feed connection must store the bearer token as its password")

        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as client:
            response = client.post("/training", json=request)
            if 400 <= response.status_code < 500:
                raise AirflowFailException(f"Training request was rejected: HTTP {response.status_code}")
            response.raise_for_status()
            if response.status_code != 202:
                raise AirflowFailException(
                    f"Training gateway must return HTTP 202, received {response.status_code}"
                )
            payload = response.json()

        if not isinstance(payload, dict):
            raise AirflowFailException("HTTP 202 response must be a JSON object containing a Job and namespace")

        job_data = payload.get("job")
        job_name = payload.get("job_name") or payload.get("name")
        namespace = payload.get("namespace") or payload.get("job_namespace")
        if isinstance(job_data, dict):
            job_name = job_name or job_data.get("name") or job_data.get("job_name")
            namespace = namespace or job_data.get("namespace")
        elif isinstance(job_data, str):
            job_name = job_name or job_data

        if (
            not isinstance(job_name, str)
            or not _JOB_NAME_PATTERN.fullmatch(job_name)
            or not isinstance(namespace, str)
            or not _JOB_NAME_PATTERN.fullmatch(namespace)
        ):
            raise AirflowFailException("HTTP 202 response must contain a valid Job name and jobs namespace")

        result = {
            "job_name": job_name,
            "namespace": namespace,
            "request_id": request["request_id"],
            "as_of": request["as_of"],
        }
        LOGGER.info("Training job accepted: %s", json.dumps(result, sort_keys=True))
        return result

    @task.sensor(
        poke_interval=30,
        timeout=6 * 60 * 60,
        mode="reschedule",
    )
    def wait_for_training(job: dict):
        from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
        from airflow.sdk import PokeReturnValue
        from airflow.sdk.exceptions import AirflowFailException

        hook = KubernetesHook(kubernetes_conn_id=KUBERNETES_CONNECTION_ID)
        try:
            kubernetes_job = hook.get_job(job["job_name"], job["namespace"])
        except Exception as error:
            if getattr(error, "status", None) == 404:
                return PokeReturnValue(is_done=False)
            raise

        conditions = getattr(getattr(kubernetes_job, "status", None), "conditions", None) or []
        complete = next(
            (condition for condition in conditions if condition.type == "Complete" and condition.status == "True"),
            None,
        )
        failed = next(
            (condition for condition in conditions if condition.type == "Failed" and condition.status == "True"),
            None,
        )
        if not complete and not failed:
            return PokeReturnValue(is_done=False)

        try:
            pods = hook.get_namespaced_pod_list(
                namespace=job["namespace"],
                label_selector=f"job-name={job['job_name']}",
            )
            for pod in pods.items:
                for container in pod.spec.containers:
                    pod_logs = hook.core_v1_client.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=job["namespace"],
                        container=container.name,
                        tail_lines=200,
                    )
                    LOGGER.info(
                        "Worker logs for pod %s container %s:\n%s",
                        pod.metadata.name,
                        container.name,
                        pod_logs,
                    )
        except Exception:
            LOGGER.exception("Could not retrieve logs for training Job %s", job["job_name"])

        if failed:
            raise AirflowFailException(
                f"Training Job {job['job_name']} failed: {failed.reason or 'unknown reason'}; "
                f"{failed.message or 'see worker logs'}"
            )

        return PokeReturnValue(
            is_done=True,
            xcom_value={**job, "status": "succeeded"},
        )

    request = validate_request()
    ingestion = verify_ingestion(request["as_of"])
    submitted_job = submit_training(request, ingestion)
    wait_for_training(submitted_job)


promiseops_training()
