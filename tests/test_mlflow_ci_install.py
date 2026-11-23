"""The installer is strict, atomic across layout validation and idempotent."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mlflow_ci_patch', HERE / 'ci/apply_mlflow_ci_access.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_repo(root):
    for name, substitutions in MODULE.PATCHES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(old for old, _ in substitutions) + '\n', encoding='utf-8')


def test_check_does_not_write_and_apply_is_idempotent(tmp_path):
    fake_repo(tmp_path)
    prior = {name: (tmp_path / name).read_bytes() for name in MODULE.PATCHES}
    assert all(MODULE.apply(tmp_path, check=True).values())
    assert all((tmp_path / name).read_bytes() == content for name, content in prior.items())
    assert all(MODULE.apply(tmp_path).values())
    assert all(count == 0 for count in MODULE.apply(tmp_path).values())


def test_unknown_layout_refuses_all_file_writes(tmp_path):
    fake_repo(tmp_path)
    target = tmp_path / '.gitlab/ci/release.yml'
    target.write_text('changed by someone else\n', encoding='utf-8')
    original = {name: (tmp_path / name).read_bytes() for name in MODULE.PATCHES}
    with pytest.raises(ValueError, match='unrecognized'):
        MODULE.apply(tmp_path)
    assert all((tmp_path / name).read_bytes() == content for name, content in original.items())
