"""Offline regression for the shared MLflow CI env resolver and read-only jobs."""
from __future__ import annotations

import base64
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "ci/require_mlflow_ci_env.sh"


def _fake_kubectl(tmp_path: Path) -> Path:
    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$KUBECTL_CALLS\"\n"
        "case \"$1\" in\n"
        "  auth) [ \"${KUBECTL_ALLOW:-yes}\" = yes ] ;;\n"
        "  get) printf '%s' \"${KUBECTL_SECRET_B64:-}\" ;;\n"
        "  *) exit 5 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return folder


def _probe(tmp_path: Path, *, uri: str | None = None, env_add: dict[str, str] | None = None):
    bin_path = _fake_kubectl(tmp_path)
    calls = tmp_path / "kubectl-calls.txt"
    calls.write_text("", encoding="utf-8")
    env = os.environ.copy()
    for key in (
        "MLFLOW_TRACKING_URI", "PROMISEOPS_MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_URI_FILE", "PROMISEOPS_MLFLOW_URI_SOURCE",
        "PROMISEOPS_MLFLOW_SECRET_NAMESPACE",
    ):
        env.pop(key, None)
    env["PATH"] = str(bin_path) + os.pathsep + env.get("PATH", "")
    env["KUBECTL_CALLS"] = str(calls)
    if uri is not None:
        env["KUBECTL_SECRET_B64"] = base64.b64encode(uri.encode()).decode()
    if env_add:
        env.update(env_add)
    script = (
        '. ci/require_mlflow_ci_env.sh && '
        'python -c \'import hashlib,os; print("HASH="+hashlib.sha256(os.environ["MLFLOW_TRACKING_URI"].encode()).hexdigest());'
        ' print("SOURCE="+os.environ["PROMISEOPS_MLFLOW_URI_SOURCE"])\''
    )
    completed = subprocess.run(
        ["sh", "-c", script], cwd=ROOT, env=env,
        capture_output=True, text=True, check=False,
    )
    return completed, calls.read_text(encoding="utf-8")


def test_kubernetes_secret_fallback_g02_prod(tmp_path):
    uri = "http://internal-mlflow.example.invalid:5000"
    out, calls = _probe(tmp_path, uri=uri)
    assert out.returncode == 0, out.stderr
    assert "HASH=" + hashlib.sha256(uri.encode()).hexdigest() in out.stdout
    assert "SOURCE=kubernetes-runtime-config" in out.stdout
    assert "secret/runtime-config" in calls
    assert "-n g02-prod" in calls
    assert uri not in (out.stdout + out.stderr)
    assert base64.b64encode(uri.encode()).decode() not in out.stdout + out.stderr


def test_staging_override_is_explicitly_allowed(tmp_path):
    out, calls = _probe(tmp_path, uri="https://mlflow.example.invalid", env_add={
        "PROMISEOPS_MLFLOW_SECRET_NAMESPACE": "g02-staging",
    })
    assert out.returncode == 0
    assert "-n g02-staging" in calls


def test_existing_ci_variable_takes_precedence_without_kubectl(tmp_path):
    primary = "https://ci.example.invalid"
    out, calls = _probe(tmp_path, uri="https://k8s.example.invalid", env_add={
        "MLFLOW_TRACKING_URI": primary,
        "PROMISEOPS_MLFLOW_TRACKING_URI": "https://alternative.example.invalid",
    })
    assert out.returncode == 0, out.stderr
    assert "SOURCE=standard" in out.stdout
    assert "HASH=" + hashlib.sha256(primary.encode()).hexdigest() in out.stdout
    assert calls == ""


def test_file_type_ci_variable_precedes_secret(tmp_path):
    value = "https://file.example.invalid/path"
    file = tmp_path / "uri-secret"
    file.write_text(value + "\n")
    out, calls = _probe(tmp_path, uri="https://k8s.example.invalid", env_add={
        "MLFLOW_TRACKING_URI": str(file),
    })
    assert out.returncode == 0, out.stderr
    assert "SOURCE=file" in out.stdout
    assert "HASH=" + hashlib.sha256(value.encode()).hexdigest() in out.stdout
    assert calls == ""


@pytest.mark.parametrize("bad_namespace", ["default", "kube-system", "g01-prod", "g02-ci"])
def test_other_secret_namespaces_fail_closed(tmp_path, bad_namespace):
    out, calls = _probe(tmp_path, uri="http://internal.example.invalid", env_add={
        "PROMISEOPS_MLFLOW_SECRET_NAMESPACE": bad_namespace,
    })
    assert out.returncode != 0
    assert calls == ""
    assert "namespace" in out.stderr


def test_k8s_rbac_denial_is_clear_and_no_secret_is_printed(tmp_path):
    uri = "https://mlflow.example.invalid"
    out, calls = _probe(tmp_path, uri=uri, env_add={"KUBECTL_ALLOW": "no"})
    assert out.returncode != 0
    assert "cannot read" in out.stderr.lower()
    assert not any(line.startswith("get ") for line in calls.splitlines())
    assert uri not in out.stdout + out.stderr


@pytest.mark.parametrize("uri", [
    "", "not-a-url", "http://", "https://user:pass@example.invalid",
    "https://mlflow.example.invalid/#/experiments/2",
    "https://mlflow.example.invalid/?token=value",
    "https://mlflow.example.invalid/with space",
])
def test_invalid_secret_uri_fails_closed(tmp_path, uri):
    out, _ = _probe(tmp_path, uri=uri)
    assert out.returncode != 0
    assert uri not in out.stdout + out.stderr or uri == ""


def test_main_mlflow_jobs_use_shared_resolver():
    golden = (ROOT / ".gitlab/ci/golden.yml").read_text()
    release = (ROOT / ".gitlab/ci/release.yml").read_text()
    inspect = (ROOT / ".gitlab/ci/inspect.yml").read_text()
    assert ". ci/require_mlflow_ci_env.sh" in golden
    assert release.count(". ci/require_mlflow_ci_env.sh") >= 4
    assert "sh ci/setup_registry_env.sh" in release
    assert "# no synthetic production deployment" not in release
    assert inspect.count(". ci/require_mlflow_ci_env.sh") >= 2
    assert "python ci/check_mlflow_http.py" in inspect
    assert "https://g02-mlflow.mlops.qbootcamp.ir" not in inspect


def test_patch_installer_idempotent():
    result = subprocess.run(
        ["python", "ci/apply_mlflow_k8s_fix.py", "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "already applied" in result.stdout


def test_http_probe_success_is_read_only(monkeypatch, capsys):
    from ci import check_mlflow_http
    class Response:
        status = 200
        headers = type("Headers", (), {"get_content_type": lambda self: "application/json"})()
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self, n): return b'{"experiments": [{"experiment_id": "2"}]}'
    class Opener:
        def open(self, req, timeout):
            assert req.get_method() == "GET"
            assert req.full_url.endswith("/api/2.0/mlflow/experiments/search")
            return Response()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://private-endpoint.example.invalid")
    monkeypatch.setattr(check_mlflow_http, "build_opener", lambda *a: Opener())
    assert check_mlflow_http.main() == 0
    printed = capsys.readouterr().out
    assert "read API reachable" in printed
    assert "private-endpoint.example.invalid" not in printed


def test_http_probe_rejects_login_html_without_printing_response(monkeypatch, capsys):
    from ci import check_mlflow_http
    class Response:
        status = 200
        headers = type("Headers", (), {"get_content_type": lambda self: "text/html"})()
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self, n): return b'<html>SECRET_LOGIN_FORM</html>'
    class Opener:
        def open(self, req, timeout): return Response()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://private-endpoint.example.invalid")
    monkeypatch.setattr(check_mlflow_http, "build_opener", lambda *a: Opener())
    assert check_mlflow_http.main() == 1
    printed = capsys.readouterr().out
    assert "non-JSON" in printed
    assert "SECRET_LOGIN_FORM" not in printed


def test_http_probe_requires_resolved_uri(monkeypatch, capsys):
    from ci import check_mlflow_http
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert check_mlflow_http.main() == 1
    assert "missing URI" in capsys.readouterr().out
