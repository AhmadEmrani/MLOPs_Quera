#!/usr/bin/env python3
"""Append a safe manual database inspection job to existing CI monitoring config."""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = '''
# Manual, authorized read-only SQL probe. No schema changes, no row data exports.
inspect-monitoring-db-readonly:
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
      : "${MONITOR_AS_OF:?Set the authorized historical company-time cutoff}"
      kubectl auth can-i get secret/runtime-config -n g02-prod --quiet
      PG_B64="$(kubectl get secret runtime-config -n g02-prod -o jsonpath='{.data.PROMISEOPS_DATABASE_URL}')"
      test -n "$PG_B64"
      PROMISEOPS_DATABASE_URL="$(printf '%s' "$PG_B64" | base64 -d)"
      unset PG_B64
      export PROMISEOPS_DATABASE_URL
      cp -a /opt/promiseops/wheelhouse ./wheelhouse
      uv venv --python /usr/local/bin/python .monitor-db-inspect-venv
      uv pip sync --python .monitor-db-inspect-venv/bin/python --no-index --find-links wheelhouse --require-hashes requirements.lock
      PYTHONPATH=src .monitor-db-inspect-venv/bin/python ci/inspect_monitoring_db.py \\
        --as-of "$MONITOR_AS_OF" --output reports/monitoring/db-readiness.json
      unset PROMISEOPS_DATABASE_URL
  artifacts:
    when: always
    expire_in: 14 days
    paths:
      - reports/monitoring/db-readiness.json
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    filename = ROOT / '.gitlab/ci/monitoring.yml'
    old = filename.read_text(encoding='utf-8')
    if 'inspect-monitoring-db-readonly:' in old:
        print('Inspection job already installed')
        return 0
    if 'monitoring-contracts:' not in old or 'inspect-monitoring-preview:' not in old:
        raise SystemExit('Unknown monitoring CI structure; refusing to edit')
    if args.check:
        print('Ready to append read-only database inspection job')
        return 0
    filename.write_text(old.rstrip() + '\n' + JOB, encoding='utf-8')
    print('Added inspect-monitoring-db-readonly to .gitlab/ci/monitoring.yml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
