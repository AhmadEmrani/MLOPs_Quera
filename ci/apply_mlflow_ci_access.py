#!/usr/bin/env python3
"""Safe, idempotent GitLab CI patch for shared MLflow environment resolution.

Intended for AhmadEmrani/MLOPs_Quera refactor/codesAndLifeCycle.
The installer refuses unknown layouts instead of silently editing wrong files.
No network or Production access. --check reports changes but does not write.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PATCHES = {
    '.gitlab/ci/golden.yml': [
        (
            '    - \'test -n "${MLFLOW_TRACKING_URI:-}" || { echo "Missing protected MLFLOW_TRACKING_URI"; exit 1; }\'',
            '    - . ci/require_mlflow_ci_env.sh',
        ),
    ],
    '.gitlab/ci/release.yml': [
        (
            '      : "${MLFLOW_TRACKING_URI:?Set protected MLFLOW_TRACKING_URI for model release review}"',
            '      . ci/require_mlflow_ci_env.sh',
        ),
        (
            '      : "${MLFLOW_TRACKING_URI:?Protected MLFLOW_TRACKING_URI is required}"',
            '      . ci/require_mlflow_ci_env.sh',
        ),
        (
            '      if [ -n "${ROLLBACK_MLFLOW_VERSION:-}" ]; then\n'
            '        sh ci/setup_registry_env.sh',
            '      if [ -n "${ROLLBACK_MLFLOW_VERSION:-}" ]; then\n'
            '        . ci/require_mlflow_ci_env.sh\n'
            '        sh ci/setup_registry_env.sh',
        ),
        (
            '      mkdir -p deployment-evidence\n'
            '      sh ci/setup_registry_env.sh\n'
            '      .registry-venv/bin/python ci/model_registry_lifecycle.py check-approved',
            '      mkdir -p deployment-evidence\n'
            '      . ci/require_mlflow_ci_env.sh\n'
            '      sh ci/setup_registry_env.sh\n'
            '      .registry-venv/bin/python ci/model_registry_lifecycle.py check-approved',
        ),
    ],
    'ci/setup_registry_env.sh': [
        (
            ': "${MLFLOW_TRACKING_URI:?The protected MLflow URI is required for champion lifecycle}"',
            '. ci/require_mlflow_ci_env.sh',
        ),
    ],
}


def transform(path: str, source: str) -> tuple[str, int]:
    changed = 0
    for old, new in PATCHES[path]:
        if old in source:
            if source.count(old) != 1:
                raise ValueError(f'{path}: ambiguous anchor for MLflow CI resolver')
            source = source.replace(old, new)
            changed += 1
        elif new in source:
            # A previous installer run already applied this exact change.
            continue
        else:
            raise ValueError(f'{path}: unrecognized GitLab CI layout; refusing overwrite')
    return source, changed


def apply(root: Path, *, check: bool = False) -> dict[str, int]:
    pending: dict[Path, str] = {}
    stats: dict[str, int] = {}
    # Validate ALL files before mutating any file.
    for name in PATCHES:
        target = root / name
        if not target.is_file():
            raise FileNotFoundError(f'Missing required CI file: {name}')
        original = target.read_text(encoding='utf-8')
        revised, count = transform(name, original)
        stats[name] = count
        if revised != original:
            pending[target] = revised
    if not check:
        for path, content in pending.items():
            path.write_text(content, encoding='utf-8')
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate compatibility; do not write')
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args(argv)
    try:
        stats = apply(args.root, check=args.check)
    except (OSError, ValueError) as error:
        print(f'MLflow CI patch BLOCKED: {error}')
        return 1
    for path, count in stats.items():
        print(f'{path}: {count} change(s) {"pending" if args.check else "applied"}')
    print('MLflow CI preflight patch: compatible; no Production changes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
