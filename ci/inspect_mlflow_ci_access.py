#!/usr/bin/env python3
"""Read-only inspection of MLflow connectivity from a GitLab Runner.

Never print or persist the URI, host, credentials, variable values, or exception
messages returned by untrusted remote services. This code never changes MLflow.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit

AUTH_VARS = (
    "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN",
)
URI_VARS = ("MLFLOW_TRACKING_URI", "PROMISEOPS_MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URI_FILE")


def _presence(env: dict[str, str]) -> dict[str, bool]:
    return {key: bool(env.get(key)) for key in (*URI_VARS, *AUTH_VARS)}


def _uri(env: dict[str, str]) -> tuple[str | None, str, str]:
    """Return URI, source category, reason; never reveal the URI in the report."""
    raw = None
    source = "absent"
    for key in URI_VARS:
        if env.get(key):
            raw = env[key]
            source = key
            break
    if not raw:
        return None, source, "not_visible_to_ci_job"
    # GitLab File-type variables are paths to protected temporary files.
    if source == "MLFLOW_TRACKING_URI_FILE" or Path(raw).is_file():
        try:
            raw = Path(raw).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None, source + ":file", "unreadable_file_variable"
        source += ":file"
    try:
        parsed = urlsplit(raw)
        port = parsed.port  # Access also validates port syntax.
        valid = (parsed.scheme in ("http", "https") and bool(parsed.hostname)
                 and parsed.username is None and parsed.password is None
                 and not any(ch.isspace() for ch in raw)
                 and (port is None or port > 0))
    except ValueError:
        valid = False
    if not valid:
        return None, source, "invalid_http_endpoint_or_embedded_credentials"
    return raw, source, "ok"


def _context(env: dict[str, str]) -> dict[str, str]:
    # Only fixed, non-secret metadata. CI_ENVIRONMENT_NAME might contain a
    # project-specific label, so report its presence rather than its value.
    return {
        "branch_is_main": str(env.get("CI_COMMIT_BRANCH") == "main").lower(),
        "ref_protected": env.get("CI_COMMIT_REF_PROTECTED", "unknown")
            if env.get("CI_COMMIT_REF_PROTECTED") in ("true", "false") else "unknown",
        "job_has_environment": str(bool(env.get("CI_ENVIRONMENT_NAME"))).lower(),
    }


def _network_probe(uri: str) -> dict[str, str]:
    """DNS and direct TCP checks are diagnostic only; proxies may bypass them."""
    url = urlsplit(uri)
    host = url.hostname
    assert host is not None
    port = url.port or (443 if url.scheme == "https" else 80)
    result = {"dns": "failed", "direct_tcp": "not_checked"}
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        result["dns"] = "passed"
    except (OSError, ValueError):
        return result
    try:
        with socket.create_connection((host, port), timeout=4):
            result["direct_tcp"] = "passed"
    except OSError:
        result["direct_tcp"] = "failed"
    return result


def _read_only_mlflow(uri: str) -> tuple[str, str | None]:
    """Return result without leaking server exception strings or endpoints."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(uri)
        # Search only: do not call set_experiment(), create_run(), log_*(),
        # set_registered_model_alias(), or any registry mutation method.
        MlflowClient().search_experiments(max_results=1)
        return "passed", None
    except Exception as exc:  # diagnostic mode, protect connection secrets
        name = type(exc).__name__
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name):
            name = "UnknownError"
        return "failed", name


def inspect(phase: str, report_path: Path, *, env: dict[str, str] | None = None) -> int:
    variables = dict(os.environ if env is None else env)
    uri, source, reason = _uri(variables)
    report = {
        "schema_version": 1,
        "inspection": "gitlab_mlflow_read_only",
        "ci": _context(variables),
        "variable_presence": _presence(variables),
        "tracking_uri_source": source,
        "tracking_uri_status": reason,
        "network": {"dns": "not_checked", "direct_tcp": "not_checked"},
        "mlflow_read": {"status": "not_checked", "error_type": None},
        "result": "blocked",
    }
    if uri is not None:
        report["network"] = _network_probe(uri)
        report["result"] = "configured_unverified"
        if phase == "read":
            result, error = _read_only_mlflow(uri)
            report["mlflow_read"] = {"status": result, "error_type": error}
            report["result"] = "passed" if result == "passed" else "blocked"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MLflow CI inspect: phase={phase} result={report['result']} "
          f"uri={reason} ref_protected={report['ci']['ref_protected']} "
          f"dns={report['network']['dns']} tcp={report['network']['direct_tcp']} "
          f"mlflow_read={report['mlflow_read']['status']}")
    if reason == "not_visible_to_ci_job":
        print("MLFLOW_TRACKING_URI is not visible to this job. Review GitLab project/group "
              "CI/CD Variables, protected main branch, and environment scope '*'.")
        print("Gateway-injected worker MLflow settings are not inherited by GitLab CI.")
    elif reason != "ok":
        print("MLflow CI URI configuration is invalid or unreadable (value hidden).")
    elif phase == "read" and report["mlflow_read"]["status"] == "failed":
        print("MLflow read-only API failed; verify network/TLS/auth/permissions. "
              "Exception content is suppressed to avoid leaking credentials.")
    return 0 if report["result"] in ("configured_unverified", "passed") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "read"), required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/mlflow/ci-access.json"))
    args = parser.parse_args()
    return inspect(args.phase, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
