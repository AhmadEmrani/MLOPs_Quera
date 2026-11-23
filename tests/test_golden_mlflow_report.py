"""Report publisher tests without network, MLflow installation or credentials."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ci.publish_golden_mlflow import publish


def report(path: Path, **changes):
    payload={
        "schema_version":1,"status":"passed","tests_passed":9,"tests_failed":0,
        "tests_errors":0,"tests_skipped":1,"tests_total":10,
        "fixture_sha256":"a"*64,"source_commit":"b"*40,
        "suite":"golden-and-temporal-safety", "ci_pipeline_id":"123","ci_job_id":"987",
        "postgres_integration_executed": False,
    }
    payload.update(changes)
    path.write_text(json.dumps(payload))


def test_publisher_rejects_failed_or_tampered_evidence(tmp_path):
    p=tmp_path/"report.json"
    report(p, status="failed")
    with pytest.raises(ValueError, match="failed golden evidence"):
        publish(p, tracking_uri="https://example.test")
    report(p,fixture_sha256="nope")
    with pytest.raises(ValueError,match="checksum"):
        publish(p,tracking_uri="https://example.test")
    report(p)
    with pytest.raises(ValueError,match="HTTP"):
        publish(p,tracking_uri="file:///tmp/mlruns")


def test_publisher_logs_metrics_artifacts_and_never_changes_alias(monkeypatch, tmp_path):
    p=tmp_path/"report.json"
    report(p)
    (tmp_path/"junit.xml").write_text("<testsuite/>")
    calls=[]
    class Run:
        info=SimpleNamespace(run_id="golden-test-run-id")
        def __enter__(self): return self
        def __exit__(self,*args):return False
    fake=SimpleNamespace(
        set_tracking_uri=lambda v:calls.append(("uri",v)),
        set_experiment=lambda v:calls.append(("experiment",v)),
        start_run=lambda **kw:Run(),
        log_params=lambda v:calls.append(("params",v)),
        log_metrics=lambda v:calls.append(("metrics",v)),
        set_tags=lambda v:calls.append(("tags",v)),
        log_artifact=lambda *a,**kw:calls.append(("artifact",a,kw)),
    )
    monkeypatch.setitem(sys.modules,"mlflow",fake)
    run_id=publish(p,tracking_uri="https://example.test")
    assert run_id=="golden-test-run-id"
    assert [x[0] for x in calls].count("artifact")==2
    assert next(x[1] for x in calls if x[0]=="metrics")["tests_passed"]==9
    assert next(x[1] for x in calls if x[0]=="tags")["postgres_integration_executed"]=="false"
    assert not any("alias" in c[0] for c in calls)
