#!/usr/bin/env python3
"""Add read-only MLflow inspection to the existing GitLab inspect include."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".gitlab/ci/inspect.yml"
ROOT_CI = ROOT / ".gitlab-ci.yml"
MARKER = "inspect-mlflow-ci-access:"
JOB = '''

# Read-only GitLab Runner diagnostic; does not access Production or mutate MLflow.
inspect-mlflow-ci-access:
  stage: inspect
  needs: []
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
  allow_failure: true
  script:
    - |
      set -eu
      set +x
      mkdir -p reports/mlflow
      python ci/inspect_mlflow_ci_access.py --phase preflight --report reports/mlflow/ci-access.json
      test -d /opt/promiseops/tracking-wheelhouse
      cp -a /opt/promiseops/tracking-wheelhouse ./tracking-wheelhouse
      uv venv --python /usr/local/bin/python .mlflow-inspect-venv
      uv pip sync --python .mlflow-inspect-venv/bin/python --no-index --find-links tracking-wheelhouse --require-hashes requirements-tracking.lock
      .mlflow-inspect-venv/bin/python ci/inspect_mlflow_ci_access.py --phase read --report reports/mlflow/ci-access.json
  artifacts:
    when: always
    expire_in: 14 days
    paths:
      - reports/mlflow/ci-access.json
'''


def apply(*, check: bool) -> str:
    ci = ROOT_CI.read_text(encoding="utf-8")
    if '- local: "/.gitlab/ci/inspect.yml"' not in ci:
        raise ValueError(".gitlab-ci.yml does not include the expected inspect.yml; refusing installation")
    source = TARGET.read_text(encoding="utf-8")
    if MARKER in source:
        if JOB.strip() not in source:
            raise ValueError("An inspect-mlflow-ci-access job already exists with different content")
        return "already installed"
    if "inspect-ci-config-presence:" not in source or "inspect-prod-resources:" not in source:
        raise ValueError("Unrecognized inspect job layout; refusing installation")
    if not check:
        TARGET.write_text(source.rstrip("\n") + JOB, encoding="utf-8")
    return "ready" if check else "installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(apply(check=args.check))


if __name__ == "__main__":
    main()
