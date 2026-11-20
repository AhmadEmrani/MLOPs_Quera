"""Manually queue candidate training through the company gateway, then observe it."""

import hashlib
import json
import logging
import re
from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, task

LOGGER = logging.getLogger(__name__)
_IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_JOB_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$|^[a-z0-9]$")
_JOBS_NAMESPACE = "g02-jobs"


@dag(
    dag_id="promiseops_training",
    description="Verify ingested data, queue candidate training, and observe the worker Job",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={
        "worker_image": Param("", type="string", description="Optional: own worker image pinned by @sha256; empty uses synced CI build"),
        "source_commit": Param("", type="string", description="Optional: full source SHA; empty uses synced CI build"),
        "train_end": Param("", type="string", description="Timezone-aware training cutoff"),
        "validation_end": Param("", type="string", description="Timezone-aware validation cutoff"),
        "as_of": Param("", type="string", description="Timezone-aware data and label availability cutoff"),
    },
    tags=["promiseops", "training", "mlflow"],
)
def promiseops_training():
    @task(retries=0)
    def validate_request() -> dict:
        import sys

        from airflow.sdk import get_current_context

        sys.path.insert(0, "/opt/airflow/dags/promiseops.zip")
        from promiseops.contracts import instant

        context = get_current_context()
        params = context["params"]
        # Cutoffs remain team-selected. The worker identity may be supplied as
        # an explicit pair for a historical replay, or loaded from the
        # immutable image metadata published by the same GitLab pipeline.
        required = ("train_end", "validation_end", "as_of")
        missing = [
            name for name in required
            if not isinstance(params.get(name), str) or not params[name].strip()
        ]
        if missing:
            raise ValueError(f"Set required training parameters: {', '.join(missing)}")

        worker_image = params.get("worker_image", "")
        source_commit = params.get("source_commit", "")
        if not isinstance(worker_image, str) or not isinstance(source_commit, str):
            raise ValueError("worker_image and source_commit must be strings")
        worker_image = worker_image.strip()
        source_commit = source_commit.strip()

        if bool(worker_image) != bool(source_commit):
            raise ValueError(
                "Provide BOTH worker_image and source_commit, or leave BOTH empty "
                "to use the worker image published by GitLab CI"
            )
        if not worker_image:
            from pathlib import Path

            release_file = Path("/opt/airflow/dags/worker-release.json")
            try:
                metadata = json.loads(release_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "Worker build metadata is unavailable or invalid at "
                    f"{release_file}; run build-worker-image, then sync-airflow"
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError("worker-release.json must contain a JSON object")
            worker_image = metadata.get("worker_image", "")
            source_commit = metadata.get("source_commit", "")
            if not isinstance(worker_image, str) or not isinstance(source_commit, str):
                raise ValueError("Invalid worker_image/source_commit in worker-release.json")
            LOGGER.info("Using worker image and source commit from %s", release_file)

        if not _IMAGE_DIGEST_PATTERN.fullmatch(worker_image):
            raise ValueError("worker_image must end in @sha256:<64 lowercase hex>")
        if not _COMMIT_PATTERN.fullmatch(source_commit):
            raise ValueError("source_commit must be a full 40-character lowercase hexadecimal SHA")

        cutoffs = {name: instant(params[name]) for name in ("train_end", "validation_end", "as_of")}
        if not cutoffs["train_end"] < cutoffs["validation_end"] <= cutoffs["as_of"]:
            raise ValueError("Cutoffs must satisfy train_end < validation_end <= as_of")

        dag_run = context.get("dag_run")
        if dag_run is None or not dag_run.run_id:
            raise ValueError("DagRun ID is required for retry-stable training request ID")
        request_id = "train-" + hashlib.sha256(dag_run.run_id.encode("utf-8")).hexdigest()

        return {
            "image": worker_image,  # Gateway contract: POST /training uses image, not worker_image
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
        import sys

        import httpx
        from airflow.sdk.exceptions import AirflowFailException

        sys.path.insert(0, "/opt/airflow/dags/promiseops.zip")
        from promiseops.airflow_runtime import gateway_credentials, open_database_connection
        from promiseops.contracts import instant

        base_url, token = gateway_credentials()
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as client:
            response = client.get("/feed", params={"as_of": as_of})
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise AirflowFailException(
                    f"Feed manifest request was rejected: HTTP {response.status_code}"
                )
            response.raise_for_status()
            manifest = response.json()

        # The gateway returns {schema_version, as_of, releases}. A release
        # has no batch_id; identify its receipt via kind + publication time +
        # payload checksum, all persisted by ingest_batch().
        from promiseops.data import parse_feed_manifest

        try:
            entries = parse_feed_manifest(manifest, as_of=as_of)
        except (TypeError, ValueError) as error:
            raise AirflowFailException(f"Invalid feed manifest: {error}") from error
        if not entries:
            raise AirflowFailException("Feed returned no releases for training as_of")

        expected: list[tuple[str, object, str]] = [
            (entry["kind"], instant(entry["available_at"]), entry["payload_sha256"])
            for entry in entries
        ]
        if len(set(expected)) != len(expected):
            raise AirflowFailException("Feed manifest repeats a release identity")

        connection = open_database_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT kind, available_at, payload_sha256
                       FROM batch_receipts WHERE available_at <= %s""",
                    (instant(as_of),),
                )
                receipt_rows = cursor.fetchall()
        finally:
            connection.close()

        stored = {
            (kind, instant(available_at), payload_hash.strip())
            for kind, available_at, payload_hash in receipt_rows
        }
        missing = [
            (kind, date.isoformat()) for kind, date, checksum in expected
            if (kind, date, checksum) not in stored
        ]
        if missing:
            raise AirflowFailException(
                "Ingestion incomplete for training cutoff: "
                f"missing_or_mismatched_releases={missing[:20]} "
                f"(total={len(missing)})"
            )

        summary = {"as_of": as_of, "verified_batch_count": len(expected)}
        LOGGER.info("Verified ingested feed snapshot: %s", json.dumps(summary, sort_keys=True))
        return summary

    @task(
        retries=2,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=2),
    )
    def submit_training(request: dict, ingestion: dict) -> dict:
        import sys

        import httpx
        from airflow.sdk.exceptions import AirflowFailException

        if ingestion.get("as_of") != request["as_of"]:
            raise AirflowFailException("Ingestion verification cutoff differs from training as_of")

        sys.path.insert(0, "/opt/airflow/dags/promiseops.zip")
        from promiseops.airflow_runtime import gateway_credentials

        base_url, token = gateway_credentials()
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        ) as client:
            response = client.post("/training", json=request)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise AirflowFailException(
                    f"Training gateway rejected the request: HTTP {response.status_code}"
                )
            response.raise_for_status()
            if response.status_code != 202:
                raise AirflowFailException(
                    f"Training gateway must return HTTP 202; received {response.status_code}"
                )
            payload = response.json()

        if not isinstance(payload, dict):
            raise AirflowFailException("HTTP 202 response must be a Job description object")
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
            or namespace != _JOBS_NAMESPACE
        ):
            raise AirflowFailException(
                "Gateway must return a valid Job name in the g02-jobs namespace"
            )

        result = {
            "job_name": job_name,
            "namespace": namespace,
            "request_id": request["request_id"],
            "as_of": request["as_of"],
        }
        LOGGER.info("Training Job accepted: %s", json.dumps(result, sort_keys=True))
        return result

    @task.sensor(poke_interval=30, timeout=6 * 60 * 60, mode="reschedule")
    def wait_for_training(job: dict):
        from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
        from airflow.sdk import BaseHook, PokeReturnValue
        from airflow.sdk.exceptions import AirflowFailException
        from pathlib import Path
        import os

        if job.get("namespace") != _JOBS_NAMESPACE:
            raise AirflowFailException("Training Job namespace is outside the team jobs namespace")

        # Prefer the dedicated team connection. Do not force in_cluster=True:
        # that overrides the connection and fails when the task pod has no
        # mounted service-account token (as reported by kube_access_probe).
        try:
            connection = BaseHook.get_connection("promiseops_jobs_kubernetes")
        except Exception as error:
            raise AirflowFailException(
                "Airflow connection 'promiseops_jobs_kubernetes' is unavailable; "
                "configure it for read-only access to g02-jobs"
            ) from error

        extras = connection.extra_dejson or {}
        in_cluster = extras.get("in_cluster") is True or str(
            extras.get("in_cluster", "")
        ).lower() == "true"
        has_connection_config = in_cluster or bool(
            extras.get("kube_config_path") or extras.get("kube_config")
        )
        service_token = Path(
            "/var/run/secrets/kubernetes.io/serviceaccount/token"
        )
        if in_cluster and not service_token.is_file():
            raise AirflowFailException(
                "Airflow Kubernetes connection requests in-cluster authentication, but the "
                "task pod has no mounted service-account token. Mount the approved read-only "
                "service account or configure kube_config_path/kube_config on the connection."
            )
        default_config = any(
            Path(path).expanduser().is_file()
            for path in os.environ.get("KUBECONFIG", "").split(os.pathsep)
            if path
        ) or (Path.home() / ".kube" / "config").is_file()
        if not has_connection_config and not default_config:
            raise AirflowFailException(
                "Airflow connection 'promiseops_jobs_kubernetes' has no Kubernetes auth "
                "configuration. Set its in_cluster, kube_config_path, or kube_config extra, "
                "or provide a kubeconfig to the Airflow task runtime. No default kubeconfig "
                "was found in KUBECONFIG or ~/.kube/config."
            )

        hook = KubernetesHook(kubernetes_conn_id="promiseops_jobs_kubernetes")
        try:
            kubernetes_job = hook.get_job(job["job_name"], job["namespace"])
        except Exception as error:
            if getattr(error, "status", None) == 404:
                return PokeReturnValue(is_done=False)
            raise  # e.g. 403: staff must grant read-only job/pod/log RBAC

        conditions = getattr(getattr(kubernetes_job, "status", None), "conditions", None) or []
        complete = next(
            (c for c in conditions if c.type == "Complete" and c.status == "True"), None
        )
        failed = next(
            (c for c in conditions if c.type == "Failed" and c.status == "True"), None
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
                    logs = hook.core_v1_client.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=job["namespace"],
                        container=container.name,
                        tail_lines=200,
                    )
                    LOGGER.info(
                        "Worker logs for pod %s container %s:\n%s",
                        pod.metadata.name,
                        container.name,
                        logs,
                    )
        except Exception:
            LOGGER.exception("Could not retrieve logs for training Job %s", job["job_name"])

        if failed:
            raise AirflowFailException(
                f"Training Job {job['job_name']} failed: {failed.reason or 'unknown reason'}; "
                f"{failed.message or 'see worker logs'}"
            )
        return PokeReturnValue(is_done=True, xcom_value={**job, "status": "succeeded"})

    request = validate_request()
    ingestion = verify_ingestion(request["as_of"])
    submitted_job = submit_training(request, ingestion)
    wait_for_training(submitted_job)


promiseops_training()
