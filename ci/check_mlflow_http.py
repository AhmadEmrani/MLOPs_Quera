#!/usr/bin/env python3
"""Read-only MLflow REST response check, using the shared CI URI resolver.

Never print the endpoint, Location header, server payload, or credentials.
This is a diagnostic; the MLflow SDK inspect is the authoritative read check.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main() -> int:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
    if not uri:
        print("MLFLOW_HTTP: missing URI (run shared CI resolver first)")
        return 1
    endpoint = uri + "/api/2.0/mlflow/experiments/search"
    request = Request(endpoint, headers={"Accept": "application/json"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=10) as response:
            content_type = response.headers.get_content_type()
            body = response.read(1000001)
            print(f"MLFLOW_HTTP: status={response.status} content_type={content_type}")
            if len(body) > 1000000:
                print("MLFLOW_HTTP: oversized response")
                return 1
            if content_type != "application/json":
                print("MLFLOW_HTTP: non-JSON response (possible UI, proxy or login page)")
                return 1
            try:
                obj = json.loads(body)
            except (UnicodeError, ValueError):
                print("MLFLOW_HTTP: invalid JSON")
                return 1
            if not isinstance(obj, dict) or not isinstance(obj.get("experiments", []), list):
                print("MLFLOW_HTTP: unexpected JSON response")
                return 1
            print("MLFLOW_HTTP: read API reachable")
            return 0
    except HTTPError as error:
        print(f"MLFLOW_HTTP: status={error.code} redirect={300 <= error.code < 400}")
        return 1
    except URLError as error:
        print(f"MLFLOW_HTTP: network_error={type(error.reason).__name__}")
        return 1
    except Exception as error:
        print(f"MLFLOW_HTTP: other_error={type(error).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
