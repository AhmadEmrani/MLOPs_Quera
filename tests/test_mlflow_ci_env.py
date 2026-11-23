"""MLflow CI resolver; no remote access and no credentials printed."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARS = ("MLFLOW_TRACKING_URI", "PROMISEOPS_MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URI_FILE")


def run_resolver(extra=None):
    env = {key: value for key, value in os.environ.items() if key not in VARS}
    env.update(extra or {})
    script = '. ci/require_mlflow_ci_env.sh && test "$MLFLOW_TRACKING_URI" = "$EXPECTED_URI"'
    return subprocess.run(["sh", "-c", script], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=False)


def test_missing_uri_fails_actionably_without_leaking():
    result = run_resolver({"CI_COMMIT_REF_PROTECTED": "false", "EXPECTED_URI": "unused"})
    assert result.returncode != 0
    assert "Settings > CI/CD > Variables" in result.stderr
    assert "CI_COMMIT_REF_PROTECTED=false" in result.stderr
    assert "MLflow CI URI present" not in result.stdout


def test_standard_uri_is_supported_without_logging_value():
    uri = "https://mlflow.example.invalid/tracking"
    result = run_resolver({"MLFLOW_TRACKING_URI": uri, "EXPECTED_URI": uri})
    assert result.returncode == 0, result.stderr
    assert uri not in result.stdout + result.stderr


def test_alternative_uri_is_supported():
    uri = "http://mlflow.example.invalid"
    result = run_resolver({"PROMISEOPS_MLFLOW_TRACKING_URI": uri, "EXPECTED_URI": uri})
    assert result.returncode == 0, result.stderr
    assert "source=alternative" in result.stdout


def test_primary_takes_precedence_over_alternative():
    uri = "https://first.example.invalid"
    result = run_resolver({"MLFLOW_TRACKING_URI": uri,
                           "PROMISEOPS_MLFLOW_TRACKING_URI": "http://other.example.invalid",
                           "EXPECTED_URI": uri})
    assert result.returncode == 0, result.stderr


def test_gitlab_file_type_variable(tmp_path):
    uri = "https://file.example.invalid/path"
    secret_file = tmp_path / "uri"
    secret_file.write_text(uri + "\n", encoding="utf-8")
    result = run_resolver({"MLFLOW_TRACKING_URI": str(secret_file), "EXPECTED_URI": uri})
    assert result.returncode == 0, result.stderr
    assert uri not in result.stdout + result.stderr


def test_explicit_file_path(tmp_path):
    uri = "https://file.example.invalid"
    secret_file = tmp_path / "uri"
    secret_file.write_text(uri, encoding="utf-8")
    result = run_resolver({"MLFLOW_TRACKING_URI_FILE": str(secret_file), "EXPECTED_URI": uri})
    assert result.returncode == 0, result.stderr


def test_invalid_or_blank_uri_fails_closed():
    for value in ("", "/tmp/not-an-endpoint", "file:///tmp/mlruns", "https://example.invalid/has space"):
        result = run_resolver({"MLFLOW_TRACKING_URI": value, "EXPECTED_URI": value})
        assert result.returncode != 0
        assert value not in result.stdout + result.stderr or value == ""
