"""Isolated tests. These never connect to external MLflow or Kubernetes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_mlflow", ROOT / "ci/inspect_mlflow_ci_access.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_absent_variable_reports_environment_scope_issue(tmp_path):
    report = tmp_path / "report.json"
    assert mod.inspect("preflight", report, env={"CI_COMMIT_BRANCH": "main", "CI_COMMIT_REF_PROTECTED": "false"}) == 1
    item = json.loads(report.read_text())
    assert item["tracking_uri_status"] == "not_visible_to_ci_job"
    assert item["ci"]["ref_protected"] == "false"
    assert item["network"]["dns"] == "not_checked"


def test_uri_value_hidden_in_log_and_artifact(tmp_path, capsys):
    report = tmp_path / "report.json"
    uri = "https://example.invalid/supersecret123"
    with patch.object(mod, "_network_probe", return_value={"dns": "passed", "direct_tcp": "passed"}):
        assert mod.inspect("preflight", report, env={"MLFLOW_TRACKING_URI": uri}) == 0
    assert uri not in report.read_text() + capsys.readouterr().out
    assert "example.invalid" not in report.read_text()


def test_gitlab_file_type_variable_and_mixed_precedence(tmp_path):
    file = tmp_path / "mlflow_var"
    file.write_text("https://example.invalid/mlflow\n")
    uri, source, status = mod._uri({"MLFLOW_TRACKING_URI": str(file)})
    assert uri == "https://example.invalid/mlflow"
    assert status == "ok" and source.endswith(":file")
    uri, source, status = mod._uri({"MLFLOW_TRACKING_URI": "http://one.invalid", "PROMISEOPS_MLFLOW_TRACKING_URI": "http://two.invalid"})
    assert uri == "http://one.invalid" and source == "MLFLOW_TRACKING_URI"


def test_uri_validation_fails_closed():
    for bad in ["ftp://host", "https://user:password@host", "http://host:bad", "file:///var/mlruns", "https://host/a b"]:
        uri, _, status = mod._uri({"MLFLOW_TRACKING_URI": bad})
        assert uri is None
        assert status == "invalid_http_endpoint_or_embedded_credentials"


def test_successful_read_only_mlflow_probe(tmp_path):
    report = tmp_path / "report.json"
    with patch.object(mod, "_network_probe", return_value={"dns": "failed", "direct_tcp": "failed"}), \
         patch.object(mod, "_read_only_mlflow", return_value=("passed", None)):
        assert mod.inspect("read", report, env={"MLFLOW_TRACKING_URI": "https://host.invalid"}) == 0
    item = json.loads(report.read_text())
    assert item["result"] == "passed" and item["network"]["direct_tcp"] == "failed"


def test_mlflow_read_error_has_no_endpoint(tmp_path):
    report = tmp_path / "report.json"
    with patch.object(mod, "_network_probe", return_value={"dns": "passed", "direct_tcp": "passed"}), \
         patch.object(mod, "_read_only_mlflow", return_value=("failed", "MlflowException")):
        assert mod.inspect("read", report, env={"MLFLOW_TRACKING_URI": "https://secret.invalid"}) == 1
    item = json.loads(report.read_text())
    assert item["mlflow_read"] == {"status": "failed", "error_type": "MlflowException"}
    assert "secret.invalid" not in report.read_text()


def test_no_duplicate_job_installation_and_check(tmp_path, monkeypatch):
    path = ROOT / "ci/install_mlflow_ci_inspect.py"
    spec2 = importlib.util.spec_from_file_location("install_mlflow", path)
    installer = importlib.util.module_from_spec(spec2)
    assert spec2 and spec2.loader
    spec2.loader.exec_module(installer)
    base = tmp_path / ".gitlab"
    (base / "ci").mkdir(parents=True)
    target = base / "ci/inspect.yml"
    target.write_text("inspect-ci-config-presence:\n  script: []\ninspect-prod-resources:\n  script: []\n")
    root_ci = tmp_path / ".gitlab-ci.yml"
    root_ci.write_text('include:\n  - local: "/.gitlab/ci/inspect.yml"\n')
    monkeypatch.setattr(installer, "ROOT_CI", root_ci)
    monkeypatch.setattr(installer, "TARGET", target)
    assert installer.apply(check=True) == "ready"
    assert "inspect-mlflow-ci-access:" not in target.read_text()
    assert installer.apply(check=False) == "installed"
    assert installer.apply(check=False) == "already installed"
    assert target.read_text().count("inspect-mlflow-ci-access:") == 1
