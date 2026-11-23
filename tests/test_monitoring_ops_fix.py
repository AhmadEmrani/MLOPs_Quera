"""Offline tests: no DB, Kubernetes, MLflow, CI secrets or Production changes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'ci' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load_script('apply_monitoring_ops_fix')
migration = load_script('apply_monitoring_migration')


def test_migration_source_verified_and_only_additive():
    source = (ROOT / 'db/migrations/004_monitoring_predictions.sql').read_bytes()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / '004.sql'
        p.write_bytes(source)
        commands = migration.migration_commands(p)
    assert len(commands) == 2
    assert commands[0].startswith('CREATE TABLE IF NOT EXISTS monitoring_predictions')
    assert commands[1].startswith('CREATE INDEX IF NOT EXISTS idx_monitoring_model_company_window')
    assert not any(word in ' '.join(commands).upper() for word in [' DROP ', ' TRUNCATE ', ' DELETE '])


def test_migration_rejects_tampering(tmp_path):
    p = tmp_path / '004.sql'
    p.write_text('DROP TABLE orders;')
    with pytest.raises(migration.MigrationBlocked, match='file_changed'):
        migration.migration_commands(p)


class FakeCursor:
    def __init__(self, *, existing=False, busy=False, malformed=False):
        self.existing = existing
        self.busy = busy
        self.malformed = malformed
        self.executed = []
        self.last = ''

    def execute(self, sql, params=None):
        self.last = str(sql)
        self.executed.append((self.last, params))
        if self.last.startswith('CREATE TABLE'):
            self.existing = True
        if self.last.startswith('CREATE INDEX'):
            self.index = True

    def fetchone(self):
        if 'pg_try_advisory_xact_lock' in self.last:
            return (not self.busy,)
        if 'to_regclass(\'monitoring_predictions\') IS NOT NULL' in self.last:
            return (self.existing,)
        if 'SELECT EXISTS' in self.last:
            return (self.existing and not self.malformed,)
        raise AssertionError('unexpected fetchone: ' + self.last)

    def fetchall(self):
        if 'FROM pg_attribute' in self.last:
            if not self.existing:
                return []
            return [(k, v[0], v[1]) for k, v in migration.REQUIRED_COLUMNS.items()
                    if not (self.malformed and k == 'features_jsonb')]
        if 'FROM pg_constraint' in self.last:
            return [(name,) for name in migration.EXPECTED_PK]
        raise AssertionError('unexpected fetchall: ' + self.last)

    def __enter__(self): return self
    def __exit__(self, *args): pass


class FakeConnection:
    def __init__(self, cursor): self.fake = cursor
    def cursor(self): return self.fake


def test_migration_atomic_first_application():
    cur = FakeCursor()
    ret = migration.migrate(FakeConnection(cur), ['CREATE TABLE t', 'CREATE INDEX x'])
    assert ret == 'applied'
    assert cur.executed[0][0] == 'BEGIN'
    assert cur.executed[-1][0] == 'COMMIT'
    assert not any(x[0] == 'ROLLBACK' for x in cur.executed)


def test_migration_idempotent():
    cur = FakeCursor(existing=True)
    assert migration.migrate(FakeConnection(cur), ['CREATE TABLE t', 'CREATE INDEX x']) == 'already_applied'
    assert not any(q.startswith('CREATE') for q, _ in cur.executed)


def test_migration_refuses_overlapping_scoring_and_rolls_back():
    cur = FakeCursor(busy=True)
    with pytest.raises(migration.MigrationBlocked, match='scheduled_scoring_in_progress'):
        migration.migrate(FakeConnection(cur), ['CREATE TABLE t', 'CREATE INDEX x'])
    assert cur.executed[-1][0] == 'ROLLBACK'
    assert not any(q.startswith('CREATE') for q, _ in cur.executed)


def test_migration_refuses_incorrect_schema_and_rolls_back():
    cur = FakeCursor(malformed=True)
    with pytest.raises(migration.MigrationBlocked, match='schema_verification_failed'):
        migration.migrate(FakeConnection(cur), ['CREATE TABLE t', 'CREATE INDEX x'])
    assert cur.executed[-1][0] == 'ROLLBACK'


def test_migration_without_protected_main_cannot_connect(monkeypatch):
    monkeypatch.setenv('CI_COMMIT_BRANCH', 'feat/activeLearning')
    monkeypatch.setenv('CI_COMMIT_REF_PROTECTED', 'false')
    monkeypatch.setenv('PROMISEOPS_DATABASE_URL', 'a-secret')
    assert migration.main([]) == 1


def test_installer_prunes_obsolete_and_preserves_other_jobs(tmp_path):
    ci = tmp_path / '.gitlab/ci'
    ci.mkdir(parents=True)
    (ci/'monitoring.yml').write_text('''monitoring-contracts:\n  stage: test\n\ninspect-monitoring-preview:\n  stage: inspect\n\ninspect-monitoring-db-readonly:\n  stage: inspect\n''')
    (ci/'inspect.yml').write_text('''inspect-api-endpoint:\n  stage: inspect\n\ninspect-ci-config-presence:\n  stage: inspect\n\ninspect-mlflow-ci-access:\n  stage: inspect\n\ninspect-mlflow-direct:\n  stage: inspect\n\ninspect-mlflow-k8s-discovery:\n  stage: inspect\n\ninspect-training-failure:\n  stage: inspect\n''')
    changed = installer.apply(tmp_path,check=True)
    assert len(changed) == 2
    assert 'inspect-monitoring-preview:' in (ci/'monitoring.yml').read_text()
    installer.apply(tmp_path,check=False)
    monitoring = (ci/'monitoring.yml').read_text()
    inspect = (ci/'inspect.yml').read_text()
    assert 'inspect-monitoring:' in monitoring
    assert 'monitoring-migrate-schema:' in monitoring
    assert 'inspect-monitoring-preview:' not in monitoring
    assert 'inspect-monitoring-db-readonly:' not in monitoring
    assert 'inspect-mlflow-direct:' not in inspect
    assert 'inspect-mlflow-k8s-discovery:' not in inspect
    assert 'inspect-ci-config-presence:' not in inspect
    assert 'inspect-mlflow-ci-access:' in inspect
    assert 'inspect-api-endpoint:' in inspect
    assert 'inspect-training-failure:' in inspect
    assert installer.apply(tmp_path,check=False) == []
    import yaml
    for file in ci.glob('*.yml'):
        jobs = yaml.safe_load(file.read_text())
        assert isinstance(jobs,dict)
    assert yaml.safe_load(monitoring)['monitoring-migrate-schema']['rules'][0]['when'] == 'manual'
    assert yaml.safe_load(monitoring)['monitoring-migrate-schema']['environment']['name'] == 'promiseops-production'


def test_reject_unknown_yaml(tmp_path):
    ci = tmp_path / '.gitlab/ci'
    ci.mkdir(parents=True)
    (ci/'monitoring.yml').write_text('unknown:\n  stage: test\n')
    (ci/'inspect.yml').write_text('inspect-api-endpoint:\n  stage: inspect\n')
    with pytest.raises(ValueError): installer.apply(tmp_path,check=False)
    assert (ci/'monitoring.yml').read_text() == 'unknown:\n  stage: test\n'


def test_one_manual_inspection_and_distinct_manual_migration():
    import yaml
    config = yaml.safe_load(installer.MONITOR_JOB)
    assert list(config) == ['inspect-monitoring']
    migration_config = yaml.safe_load(installer.MIGRATION_JOB)['monitoring-migrate-schema']
    assert migration_config['resource_group'] == 'promiseops-production'
    assert migration_config['rules'][0]['when'] == 'manual'
    assert migration_config['environment']['name'] == 'promiseops-production'
    assert 'CI_COMMIT_REF_PROTECTED == "true"' in migration_config['rules'][0]['if']
    assert 'MONITOR_MODE' in installer.MONITOR_JOB
    assert 'MODE=preview' not in installer.MIGRATION_JOB
