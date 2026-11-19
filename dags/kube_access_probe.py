"""Read-only diagnostics for Kubernetes access from the *Airflow task runtime*."""

from __future__ import annotations

from datetime import datetime, timezone

from airflow.sdk import dag, task


@dag(
    dag_id="promiseops_k8s_access_probe",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    params={"job_name": ""},
    tags=["promiseops", "diagnostics"],
)
def promiseops_k8s_access_probe():
    @task(retries=0)
    def inspect():
        import os
        from pathlib import Path

        from airflow.sdk import BaseHook, get_current_context
        from airflow.providers.cncf.kubernetes.hooks.kubernetes import KubernetesHook
        from kubernetes.client.exceptions import ApiException

        env_names = (
            "KUBERNETES_SERVICE_HOST",
            "KUBERNETES_SERVICE_PORT",
            "KUBECONFIG",
            "AIRFLOW_CONN_KUBERNETES_DEFAULT",
            "AIRFLOW_CONN_PROMISEOPS_JOBS_KUBERNETES",
        )
        for name in env_names:
            print(f"K8S_ENV {name} defined={bool(os.environ.get(name))}")

        service_dir = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        for name in ("token", "ca.crt", "namespace"):
            print(f"K8S_FILE serviceaccount_{name} exists={(service_dir / name).is_file()}")

        kubeconfig_paths = [
            Path(value).expanduser()
            for value in os.environ.get("KUBECONFIG", "").split(os.pathsep)
            if value
        ]
        print(f"K8S_FILE kubeconfig_env_has_file={any(p.is_file() for p in kubeconfig_paths)}")
        print(f"K8S_FILE home_kubeconfig_exists={(Path.home() / '.kube' / 'config').is_file()}")

        job_name = str(get_current_context()["params"].get("job_name", "")).strip()
        if job_name:
            import re

            if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", job_name):
                raise ValueError("job_name must be a valid Kubernetes resource name")
        else:
            print("K8S_API skipped: provide job_name in DAG trigger parameters")

        for conn_id in ("promiseops_jobs_kubernetes", "kubernetes_default"):
            try:
                conn = BaseHook.get_connection(conn_id)
                extras = conn.extra_dejson or {}
                if not isinstance(extras, dict):
                    extras = {}
                print(f"K8S_CONN {conn_id} available=True type={conn.conn_type}")
                for field in ("in_cluster", "kube_config_path", "kube_config", "namespace"):
                    print(f"K8S_CONN {conn_id} extra_{field}_defined={field in extras and extras[field] is not None}")
            except Exception as exc:
                print(f"K8S_CONN {conn_id} available=False error_type={type(exc).__name__}")
                continue

            if job_name:
                try:
                    # Deliberately DO NOT pass in_cluster=True here.
                    hook = KubernetesHook(kubernetes_conn_id=conn_id)
                    job = hook.get_job(job_name=job_name, namespace="g02-jobs")
                    status = job.status
                    print(
                        f"K8S_GET_JOB {conn_id} OK "
                        f"succeeded={status.succeeded or 0} failed={status.failed or 0}"
                    )
                except ApiException as exc:
                    print(f"K8S_GET_JOB {conn_id} HTTP_STATUS={exc.status}")
                except Exception as exc:
                    # Exception messages can contain URLs and configuration values.
                    print(f"K8S_GET_JOB {conn_id} ERROR_TYPE={type(exc).__name__}")

        first_kubeconfig = next((p for p in kubeconfig_paths if p.is_file()), None)
        if job_name and first_kubeconfig:
            try:
                hook = KubernetesHook(
                    kubernetes_conn_id="kubernetes_default",
                    config_file=str(first_kubeconfig),
                    in_cluster=False,
                )
                job = hook.get_job(job_name=job_name, namespace="g02-jobs")
                print(f"K8S_GET_JOB explicit_kubeconfig OK succeeded={job.status.succeeded or 0}")
            except ApiException as exc:
                print(f"K8S_GET_JOB explicit_kubeconfig HTTP_STATUS={exc.status}")
            except Exception as exc:
                print(f"K8S_GET_JOB explicit_kubeconfig ERROR_TYPE={type(exc).__name__}")

    inspect()


promiseops_k8s_access_probe()
