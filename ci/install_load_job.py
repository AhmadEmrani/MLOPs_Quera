#!/usr/bin/env python3
"""Idempotently register the load-test CI include without replacing other CI edits."""

from __future__ import annotations

import re
from pathlib import Path

ENTRY = '  - local: "/.gitlab/ci/load.yml"'


def add_include(text: str) -> str:
    if re.search(r'^\s*-\s+local:\s+[\'"]?/?\.gitlab/ci/load\.yml[\'"]?\s*$', text, re.M):
        return text
    lines = text.splitlines(keepends=True)
    begin = next((i for i, line in enumerate(lines) if re.fullmatch(r'include:\s*\n?', line)), None)
    if begin is None:
        raise ValueError(".gitlab-ci.yml has no top-level include list")
    end = next(
        (i for i in range(begin + 1, len(lines))
         if lines[i].strip() and not lines[i].startswith((" ", "\t", "#"))),
        len(lines),
    )
    if not any(re.match(r'^\s+-\s+local:\s+', line) for line in lines[begin + 1:end]):
        raise ValueError("the include section is not the expected local-include list")
    while end > begin + 1 and lines[end - 1].strip() == "":
        end -= 1
    lines.insert(end, ENTRY + "\n")
    return "".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / ".gitlab-ci.yml"
    original = config.read_text(encoding="utf-8")
    updated = add_include(original)
    if updated == original:
        print("Load CI include already exists; no change needed.")
    else:
        config.write_text(updated, encoding="utf-8")
        print("Registered /.gitlab/ci/load.yml in .gitlab-ci.yml (other entries retained).")


if __name__ == "__main__":
    main()
