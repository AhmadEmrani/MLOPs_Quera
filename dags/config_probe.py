
"""Manual read-only Airflow configuration probe.

Only environment variable names and connection availability
are logged. No credentials, URIs or tokens are printed.
"""

import pendulum
from airflow.sdk import dag, task


@dag(
    dag_id="promiseops_config_probe",
    description="Read-only PromiseOps connection inspection",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["promiseops", "diagnostic"],
)
def promiseops_config_probe():

    @task(retries=0)
    def check_config() -> dict:
        import logging
        import os
        from airflow.sdk import BaseHook

        log = logging.getLogger("promiseops.config_probe")

        target_ids = (
            "promiseops_feed",
            "promiseops_postgres",
            "promiseops_jobs_kubernetes",
        )

        hints = (
            "FEED",
            "GATEWAY",
            "POSTGRES",
            "DATABASE",
            "PROMISEOPS",
            "MLFLOW",
        )

        env_names = sorted(
            name
            for name in os.environ
            if name.startswith("AIRFLOW_CONN_")
            or any(hint in name.upper() for hint in hints)
        )

        log.warning(
            "CONFIG_PROBE relevant_env_var_names=%s",
            env_names,
        )

        summary = {}

        for conn_id in target_ids:
            try:
                conn = BaseHook.get_connection(conn_id)

            except Exception as exc:
                status = "UNAVAILABLE"

                log.warning(
                    "CONFIG_PROBE conn_id=%s "
                    "status=%s error_type=%s",
                    conn_id,
                    status,
                    type(exc).__name__,
                )

            else:
                status = "RESOLVED"

                log.warning(
                    "CONFIG_PROBE conn_id=%s "
                    "status=%s type=%s",
                    conn_id,
                    status,
                    conn.conn_type,
                )

            summary[conn_id] = status

        log.warning(
            "CONFIG_PROBE completed "
            "(no data or connection tests performed)"
        )

        return summary

    check_config()


promiseops_config_probe()
