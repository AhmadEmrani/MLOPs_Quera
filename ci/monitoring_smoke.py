"""Offline contract runner (real report evaluation uses the read-only CLI)."""
from __future__ import annotations
import subprocess
import sys


def main() -> int:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q",
                             "tests/test_monitoring_phase1.py"], check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
