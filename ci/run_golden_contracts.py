#!/usr/bin/env python3
"""Run deterministic golden/temporal regression checks and write MLflow-ready JSON."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def main() -> int:
    output = Path("reports/golden")
    output.mkdir(parents=True, exist_ok=True)
    tests = ["tests/test_golden_contract.py", "tests/test_temporal_safety.py", "tests/test_golden_postgres.py", "tests/test_golden_model_predictions.py"]
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", *tests,
           "--junitxml", str(output / "junit.xml")]
    status = subprocess.call(cmd)
    junit = output / "junit.xml"
    passed = failed = skipped = errors = total = 0
    integration_ran = model_golden_ran = False
    if junit.is_file():
        root = ET.parse(junit).getroot()
        for suite in root.iter("testsuite"):
            total += int(suite.get("tests", "0"))
            failed += int(suite.get("failures", "0"))
            errors += int(suite.get("errors", "0"))
            skipped += int(suite.get("skipped", "0"))
        passed = total - failed - errors - skipped
        for testcase in root.iter("testcase"):
            identity = (testcase.get("classname", "") + " " + testcase.get("name", ""))
            executed = testcase.find("skipped") is None
            if "test_golden_postgres" in identity:
                integration_ran = executed
            if "test_golden_model_predictions" in identity:
                model_golden_ran = executed
    fixture = Path("fixtures/golden_features_v1.json")
    report = {
        "schema_version": 1,
        "suite": "golden-and-temporal-safety",
        "status": "passed" if status == 0 and passed >= 8 and failed == 0 and errors == 0 else "failed",
        "tests_total": total,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
        "tests_skipped": skipped,
        "postgres_integration_executed": integration_ran,
        "model_output_golden_executed": model_golden_ran,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "source_commit": os.environ.get("CI_COMMIT_SHA", "local"),
        "branch": os.environ.get("CI_COMMIT_REF_NAME", "local"),
        "ci_pipeline_id": os.environ.get("CI_PIPELINE_ID", "local"),
        "ci_job_id": os.environ.get("CI_JOB_ID", "local"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
