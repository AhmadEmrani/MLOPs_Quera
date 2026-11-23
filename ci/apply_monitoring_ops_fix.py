#!/usr/bin/env python3
"""Idempotently consolidate obsolete CI inspections and add guarded migration.

No cloud/Kubernetes/DB actions; edits only two known GitLab YAML files.
Does not change training/release/production deployment jobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HEADER = re.compile(r"(?m)^([a-z][a-z0-9_-]+):[ \t]*$")
OLD_MONITOR = {"inspect-monitoring-preview", "inspect-monitoring-db-readonly"}
OLD_INSPECT = {
    "inspect-mlflow-direct", "inspect-mlflow-k8s-discovery", "inspect-ci-config-presence",
}

MONITOR_JOB = r'''inspect-monitoring:
  # One read-only entrypoint; MODE=preview loads exact MLflow reference.
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
      umask 077
      MONITOR_AS_OF="${MONITOR_AS_OF:-2018-06-29T23:59:59-03:00}"
      MONITOR_MODE="${MONITOR_MODE:-readiness}"
      case "$MONITOR_MODE" in readiness|preview) ;; *) echo 'MONITOR_MODE must be readiness or preview' >&2; exit 1;; esac
      kubectl auth can-i get secret/runtime-config -n g02-prod --quiet
      PG_B64="$(kubectl get secret runtime-config -n g02-prod -o jsonpath='{.data.PROMISEOPS_DATABASE_URL}')"
      test -n "$PG_B64" || { echo 'Authorized database URL is missing' >&2; exit 1; }
      PROMISEOPS_DATABASE_URL="$(printf '%s' "$PG_B64" | base64 -d)"
      unset PG_B64
      export PROMISEOPS_DATABASE_URL
      cp -a /opt/promiseops/wheelhouse ./wheelhouse
      uv venv --python /usr/local/bin/python .monitor-ops-venv
      uv pip sync --python .monitor-ops-venv/bin/python --no-index --find-links wheelhouse --require-hashes requirements.lock
      PYTHONPATH=src .monitor-ops-venv/bin/python ci/inspect_monitoring_db.py \
        --as-of "$MONITOR_AS_OF" --output reports/monitoring/db-readiness.json
      if [ "$MONITOR_MODE" = preview ]; then
        : "${MONITOR_MODEL_VERSION:?Provide exact immutable model sha256 identity}"
        : "${MONITOR_RUN_ID:?Provide its original MLflow training run ID}"
        : "${MONITOR_FEATURE_REFERENCE_SHA256:?Provide its verified reference SHA256}"
        PYTHONPATH=src .monitor-ops-venv/bin/python - <<'PY_READY'
      import json
      from pathlib import Path
      report = json.loads(Path('reports/monitoring/db-readiness.json').read_text())
      if report['status'] in ('migration_004_missing', 'no_monitoring_predictions'):
          raise SystemExit('Monitoring preview blocked: schema or predictions are not ready')
      PY_READY
        . ci/require_mlflow_ci_env.sh
        PRIVATE_DIR="$(mktemp -d)"
        trap 'rm -rf "$PRIVATE_DIR"' EXIT HUP INT TERM
        cp -a /opt/promiseops/tracking-wheelhouse ./tracking-wheelhouse
        uv venv --python /usr/local/bin/python .monitor-tracking-venv
        uv pip sync --python .monitor-tracking-venv/bin/python --no-index --find-links tracking-wheelhouse --require-hashes requirements-tracking.lock
        PYTHONPATH=src .monitor-tracking-venv/bin/python ci/download_monitoring_reference.py \
          --run-id "$MONITOR_RUN_ID" \
          --model-version "$MONITOR_MODEL_VERSION" \
          --reference-sha256 "$MONITOR_FEATURE_REFERENCE_SHA256" \
          --destination "$PRIVATE_DIR"
        PYTHONPATH=src .monitor-ops-venv/bin/python -m promiseops.monitoring.cli \
          --as-of "$MONITOR_AS_OF" --model-version "$MONITOR_MODEL_VERSION" \
          --reference-file "$PRIVATE_DIR/feature_reference.json" \
          --validation-metrics "$PRIVATE_DIR/validation_metrics.json" \
          --reference-sha256 "$MONITOR_FEATURE_REFERENCE_SHA256" \
          --output reports/monitoring/preview.json
      fi
      unset PROMISEOPS_DATABASE_URL
  artifacts:
    when: always
    expire_in: 14 days
    paths:
      - reports/monitoring/db-readiness.json
      - reports/monitoring/preview.json
'''

MIGRATION_JOB = r'''monitoring-migrate-schema:
  # Explicit migration; protected environment; never automatic.
  stage: inspect
  needs:
    - job: monitoring-contracts
      artifacts: false
  resource_group: promiseops-production
  environment:
    name: promiseops-production
    action: verify
  rules:
    - if: '$CI_COMMIT_BRANCH == "main" && $CI_COMMIT_REF_PROTECTED == "true"'
      when: manual
  allow_failure: true
  script:
    - |
      set -eu
      set +x
      umask 077
      kubectl auth can-i get secret/runtime-config -n g02-prod --quiet
      PG_B64="$(kubectl get secret runtime-config -n g02-prod -o jsonpath='{.data.PROMISEOPS_DATABASE_URL}')"
      test -n "$PG_B64" || { echo 'Authorized database URL is missing' >&2; exit 1; }
      PROMISEOPS_DATABASE_URL="$(printf '%s' "$PG_B64" | base64 -d)"
      unset PG_B64
      export PROMISEOPS_DATABASE_URL
      cp -a /opt/promiseops/wheelhouse ./wheelhouse
      uv venv --python /usr/local/bin/python .monitor-migration-venv
      uv pip sync --python .monitor-migration-venv/bin/python --no-index --find-links wheelhouse --require-hashes requirements.lock
      .monitor-migration-venv/bin/python ci/apply_monitoring_migration.py
      unset PROMISEOPS_DATABASE_URL
'''


def sections(source: str) -> tuple[str, list[tuple[str, str]]]:
    matches = list(HEADER.finditer(source))
    if not matches:
        raise ValueError('No top-level jobs detected')
    header = source[:matches[0].start()]
    records = []
    for i, m in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(source)
        records.append((m.group(1), source[m.start():end].rstrip()))
    if len(set(name for name, _ in records)) != len(records):
        raise ValueError('Duplicate top-level job name')
    return header, records


def clean_monitor_header(source: str) -> str:
    # Remove only the old, now inaccurate comment attached to the preceding job.
    return source.replace(
        '# Optional manual live *read-only* preview. Requires approved 004 schema migration,\n'
        '# baseline artifact from an exact MLflow run, and a company-time cutoff.\n',
        '',
    )


def render_monitoring(source: str) -> str:
    _, jobs = sections(clean_monitor_header(source))
    names = {name for name, _ in jobs}
    if 'monitoring-contracts' not in names or not ({'inspect-monitoring', *OLD_MONITOR} & names):
        raise ValueError('Unrecognized monitoring YAML: required anchors missing')
    prefix, jobs = sections(clean_monitor_header(source))
    preserved = [block for name, block in jobs
                 if name not in (OLD_MONITOR | {'inspect-monitoring', 'monitoring-migrate-schema'})]
    return (prefix.rstrip() + '\n' if prefix.strip() else '') + '\n\n'.join(
        [x.rstrip() for x in preserved] + [MONITOR_JOB.rstrip(), MIGRATION_JOB.rstrip()]
    ) + '\n'


def render_inspect(source: str) -> str:
    prefix, jobs = sections(source)
    names = {name for name, _ in jobs}
    if 'inspect-mlflow-ci-access' not in names or 'inspect-api-endpoint' not in names:
        raise ValueError('Unrecognized inspect YAML: required jobs missing')
    preserved = [block for name, block in jobs if name not in OLD_INSPECT]
    return (prefix.rstrip() + '\n' if prefix.strip() else '') + '\n\n'.join(
        [x.rstrip() for x in preserved]
    ) + '\n'


def apply(root: Path, *, check: bool) -> list[str]:
    paths = {
        root / '.gitlab/ci/monitoring.yml': render_monitoring,
        root / '.gitlab/ci/inspect.yml': render_inspect,
    }
    rendered: dict[Path, str] = {}
    for path, fn in paths.items():
        rendered[path] = fn(path.read_text(encoding='utf-8'))
    modified = []
    for path, value in rendered.items():
        if path.read_text(encoding='utf-8') != value:
            modified.append(str(path.relative_to(root)))
            if not check:
                path.write_text(value, encoding='utf-8')
    return modified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Only check compatibility with YAML files')
    args = parser.parse_args()
    changes = apply(ROOT, check=args.check)
    print('Monitoring operations patch: ' + ('compatible' if args.check else 'applied') +
          '; changed=' + (', '.join(changes) if changes else 'none'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
