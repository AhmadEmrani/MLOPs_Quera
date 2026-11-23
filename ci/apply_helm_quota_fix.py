#!/usr/bin/env python3
"""Install safe Helm Secret quota preflight and remove duplicate inspect jobs.

This edits only known anchors in the observed feat/activeLearning layout,
validates every file first, and is idempotent. No remote operations.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path('.gitlab/ci/release.yml')
PROD = Path('ci/deploy_production.sh')
INSPECT = Path('.gitlab/ci/inspect.yml')
FAILURE = Path('.gitlab/ci/inspect-deploy-failure.yml')
PRE_FLIGHT = 'python ci/helm_secret_budget.py --namespace "$NAMESPACE" --quota-fallback "${PROMISEOPS_SECRET_QUOTA_FALLBACK:-20}"'


def replace_helm(body: str, *, production: bool) -> str:
    target = body
    old = '--atomic --wait --timeout 8m --history-max 20'
    new = '--atomic --wait --timeout 8m --history-max 5'
    count = target.count(old)
    if count == 2:
        target = target.replace(old, new)
    elif count != 0 or target.count(new) != 2:
        raise ValueError('Unrecognized Helm history configuration; refusing to overwrite')

    if production:
        # This script has two Helm upgrades; each must independently ensure
        # headroom because atomic rollbacks also consume a revision.
        prefix = ''
        pattern = r'(?m)^(helm upgrade(?: --install)? "\$RELEASE_NAME" "\$CHART" \\\n)'
    else:
        prefix = '      '
        pattern = r'(?m)^(      helm upgrade --install (?:"\$RELEASE_NAME"|promiseops) deploy/chart \\\n)'
    matches = list(re.finditer(pattern, target))
    if len(matches) != 2:
        raise ValueError('Expected exactly two Helm upgrade commands in ' + ('production' if production else 'staging'))

    for match in reversed(matches):
        start = match.start()
        preceding = target[:start]
        # Idempotent: inspect exact immediately-preceding guard, not a global count.
        if preceding.endswith(prefix + PRE_FLIGHT + '\n'):
            continue
        target = target[:start] + prefix + PRE_FLIGHT + '\n' + target[start:]
    if target.count(PRE_FLIGHT) != 2:
        raise ValueError('Helm guard must appear exactly twice')
    return target


HEADER = re.compile(r'(?m)^([a-z][a-z0-9-]*):[ \t]*$')


def delete_named_job(body: str, job: str, preserved: str) -> str:
    matches = list(HEADER.finditer(body))
    names = [m.group(1) for m in matches]
    if len(names) != len(set(names)) or preserved not in names:
        raise ValueError('Unrecognized CI layout; required inspect job missing or duplicate')
    if job not in names:
        return body
    index = names.index(job)
    start = matches[index].start()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
    new = body[:start].rstrip('\n') + '\n\n' + body[end:].lstrip('\n')
    return new.rstrip('\n') + '\n'


def rewrite_inspect(body: str) -> str:
    body = delete_named_job(body, 'inspect-own-namespaces', 'inspect-api-endpoint')
    required = ('inspect-mlflow-ci-access:', 'inspect-api-endpoint:', 'inspect-prod-resources:')
    if not all(item in body for item in required):
        raise ValueError('Required consolidated inspect jobs not found')
    # Add Helm quota diagnosis to existing resource inspection: no new CI job.
    line = '      python ci/helm_secret_budget.py --namespace "$NAMESPACE" --dry-run || true'
    anchor = '      echo "Inspection completed. No resources changed."'
    if line not in body:
        if body.count(anchor) != 1:
            raise ValueError('Cannot find production resource inspection anchor')
        body = body.replace(anchor, '      echo "=== Helm Secret quota budget (dry-run) ==="\n' + line + '\n\n' + anchor, 1)
    return body


def rewrite_failure(body: str) -> str:
    body = delete_named_job(body, 'inspect-deploy-rs', 'inspect-deploy-failure')
    if 'inspect-runtime-serviceaccount:' not in body:
        raise ValueError('Runtime service account inspection should be preserved')
    line = '      python ci/helm_secret_budget.py --namespace "$NAMESPACE" --dry-run || true'
    anchor = '      helm -n "$NAMESPACE" status promiseops || true'
    if line not in body:
        if body.count(anchor) != 1:
            raise ValueError('Cannot find deployment failure inspection anchor')
        body = body.replace(anchor, anchor + '\n      echo "=== Helm Secret quota budget (dry-run) ==="\n' + line, 1)
    return body


def planned(root: Path) -> dict[Path, str]:
    processors = {
        RELEASE: lambda data: replace_helm(data, production=False),
        PROD: lambda data: replace_helm(data, production=True),
        INSPECT: rewrite_inspect,
        FAILURE: rewrite_failure,
    }
    result = {}
    for relative, transform in processors.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f'Missing {relative}')
        result[path] = transform(path.read_text(encoding='utf-8'))
    return result


def apply(root: Path, *, check: bool) -> list[str]:
    proposed = planned(root)  # Validate ALL changes before any file is written.
    changed = [p for p, value in proposed.items() if p.read_text(encoding='utf-8') != value]
    if not check:
        for path in changed:
            path.write_text(proposed[path], encoding='utf-8')
    return [str(p.relative_to(root)) for p in changed]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        changed = apply(ROOT, check=args.check)
    except ValueError as error:
        parser.exit(1, f'Helm quota patch BLOCKED: {error}\n')
    print('Helm quota patch: ' + ('layout_valid' if args.check else 'applied') +
          '; files=' + (', '.join(changed) if changed else 'already_applied'))
    print('No Kubernetes, Helm, GitLab, database or Production operations performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
