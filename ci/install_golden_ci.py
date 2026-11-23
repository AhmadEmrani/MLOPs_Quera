#!/usr/bin/env python3
"""Install an additive CI include only; do not overwrite existing jobs."""
from pathlib import Path

root = Path('.gitlab-ci.yml')
if not root.is_file():
    raise SystemExit('Run from PromiseOps repository root')
text = root.read_text(encoding='utf-8')
new = '  - local: "/.gitlab/ci/golden.yml"\n'
if new in text:
    print('Golden CI include already installed')
else:
    anchor = 'include:\n'
    if text.count(anchor) != 1:
        raise SystemExit('Ambiguous include block; edit .gitlab-ci.yml manually')
    text = text.replace(anchor, anchor + new, 1)
    root.write_text(text,encoding='utf-8')
    print('Added .gitlab/ci/golden.yml include')
