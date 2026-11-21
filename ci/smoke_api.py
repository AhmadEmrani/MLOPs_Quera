"""Authenticated API checks for the staging and production deployment jobs."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call(path: str, *, method: str = "GET", body=None, api_key: str | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key is not None:
        headers["X-API-Key"] = api_key
    request = Request(
        os.environ["PROMISEOPS_API_URL"].rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        return error.code, error.read()
    except URLError as error:
        raise RuntimeError(f"request to {path} failed: {type(error.reason).__name__}") from error
    with response:
        return response.status, response.read()


def expect(path: str, status: int, **kwargs):
    actual, payload = call(path, **kwargs)
    if actual != status:
        raise RuntimeError(f"{kwargs.get('method', 'GET')} {path}: expected {status}, got {actual}")
    return payload


def main() -> int:
    key = os.environ.get("PROMISEOPS_API_KEY", "")
    order_id = os.environ.get("PROMISEOPS_SMOKE_ORDER_ID", "")
    pipeline = os.environ.get("CI_PIPELINE_ID", "manual")
    job = os.environ.get("CI_JOB_ID", "manual")
    smoke_run = os.environ.get("PROMISEOPS_SMOKE_RUN_ID", job)
    if not key:
        raise RuntimeError("runtime Secret has no PROMISEOPS_API_KEY")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", order_id) is None:
        raise RuntimeError("set PROMISEOPS_SMOKE_ORDER_ID to an ingested order ID")

    expect("/healthz", 200)
    expect("/readyz", 200)
    expect("/metrics", 200, api_key=key)

    expect("/predict", 401, method="POST", body={"order_id": order_id})
    expect("/predict", 401, method="POST", body={"order_id": order_id}, api_key="invalid")
    expect("/predict", 422, method="POST", body={"not_order_id": "x"}, api_key=key)
    expect(
        "/predict",
        404,
        method="POST",
        body={"order_id": f"ci-unknown-{pipeline}-{job}"},
        api_key=key,
    )

    request_id = f"ci-{pipeline}-{smoke_run}"
    request_body = {"order_id": order_id, "request_id": request_id}
    first = expect("/predict", 200, method="POST", body=request_body, api_key=key)
    first_result = json.loads(first)
    second = expect("/predict", 200, method="POST", body=request_body, api_key=key)
    if json.loads(second) != first_result:
        raise RuntimeError("same request_id and order_id did not return the stored result")

    probability = first_result.get("risk_probability")
    if (
        first_result.get("order_id") != order_id
        or first_result.get("request_id") != request_id
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
        or not isinstance(first_result.get("model_version"), str)
        or not first_result["model_version"]
    ):
        raise RuntimeError("prediction response violates the public response contract")

    expect(
        "/predict",
        409,
        method="POST",
        body={"order_id": "ci-another-order", "request_id": request_id},
        api_key=key,
    )

    evidence_path = os.environ.get("PROMISEOPS_SMOKE_EVIDENCE_FILE", "").strip()
    if evidence_path:
        evidence = {
            "status": "passed",
            "model_version": first_result["model_version"],
            "checks": [
                "healthz",
                "readyz",
                "metrics_auth",
                "prediction",
                "missing_and_wrong_key",
                "invalid_input",
                "unknown_order",
                "request_idempotency",
                "request_id_conflict",
            ],
        }
        target = Path(evidence_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "API smoke checks passed: health, readiness, metrics auth, prediction, "
        "401/404/409/422 handling, response fields, and request idempotency."
    )
    print(f"model_version={first_result['model_version']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"API smoke checks failed: {error}", file=sys.stderr)
        raise SystemExit(1)
