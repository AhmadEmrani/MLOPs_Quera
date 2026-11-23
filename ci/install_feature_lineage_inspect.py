#!/usr/bin/env python3
"""Add optional manual inspection job to main GitLab pipeline, idempotently."""
from pathlib import Path

path = Path('.gitlab-ci.yml')
original = path.read_text(encoding='utf-8')
name = '  - local: "/.gitlab/ci/feature-lineage.yml"\n'
if name in original:
    print('Feature lineage inspection already included')
else:
    anchor = '  - local: "/.gitlab/ci/inspect.yml"\n'
    if original.count(anchor) != 1:
        raise SystemExit('Expected exactly one inspect.yml include; aborting')
    path.write_text(original.replace(anchor, anchor + name, 1), encoding='utf-8')
    print('Included .gitlab/ci/feature-lineage.yml')
