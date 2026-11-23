"""Offline contract tests for bounded Helm history cleanup and patch installation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'ci' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # dataclasses access sys.modules during class creation
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


budget = load('helm_secret_budget')
installer = load('apply_helm_quota_fix')


def helm(version, status='superseded'):
    return budget.Secret(
        f'sh.helm.release.v1.promiseops.v{version}',
        'helm.sh/release.v1', 'helm', 'promiseops', str(version), status,
    )


def runtime(name):
    return budget.Secret(name, 'Opaque', '', '', '', '')


def test_quota_full_prunes_only_old_superseded_and_keeps_active():
    items = [runtime('registry-pull'), runtime('runtime-config')]
    items += [helm(n, 'deployed' if n == 18 else 'superseded') for n in range(1, 19)]
    drop, used, free, protected = budget.plan(items, limit=20)
    assert used == 20
    assert [x.version for x in drop] == list(range(1, 14))
    assert free >= 3
    assert 18 in protected and 17 in protected and 16 in protected
    assert all(x.status == 'superseded' for x in drop)
    assert all(x.name not in ('registry-pull', 'runtime-config') for x in drop)


def test_non_helm_secrets_never_deleted_and_fails_closed_without_headroom():
    items = [runtime('non-helm-'+str(i)) for i in range(17)]
    items += [helm(10), helm(11), helm(12, 'deployed')]
    with pytest.raises(budget.BudgetBlocked, match='insufficient_secret_headroom'):
        budget.plan(items, limit=20)


def test_only_two_historical_rollback_versions_protected_under_pressure():
    items = [runtime('other-'+str(i)) for i in range(15)]
    items += [helm(i, 'deployed' if i == 10 else 'superseded') for i in range(6, 11)]
    with pytest.raises(budget.BudgetBlocked, match='insufficient_secret_headroom'):
        budget.plan(items, limit=20)


def test_detect_pending_or_unknown_release_before_deletion():
    for state, explanation in [('pending-upgrade', 'pending'), ('mystery', 'unknown')]:
        data = [helm(1), helm(2, 'deployed'), helm(3, state)]
        with pytest.raises(budget.BudgetBlocked, match=explanation):
            budget.plan(data, limit=20)


def test_mismatched_helm_metadata_is_rejected():
    bad = budget.Secret('sh.helm.release.v1.promiseops.v9', 'helm.sh/release.v1',
                        'helm', 'promiseops', '8', 'superseded')
    with pytest.raises(budget.BudgetBlocked, match='metadata_mismatch'):
        budget.plan([bad, helm(10, 'deployed')], limit=20)


def test_keep_other_release_secrets():
    other = budget.Secret('sh.helm.release.v1.unrelated.v6', 'helm.sh/release.v1',
                          'helm', 'unrelated', '6', 'superseded')
    items = [other, helm(10, 'deployed')]
    drop, _, _, _ = budget.plan(items, limit=20)
    assert other not in drop


def test_first_installation_with_no_helm_history_is_valid():
    drop, count, free, _ = budget.plan([runtime('runtime-config'), runtime('registry-pull')], limit=20)
    assert drop == [] and count == 2 and free == 18


def test_assumed_quota_used_more_conservative_than_listing():
    items = [runtime('other')] + [helm(x, 'deployed' if x == 10 else 'superseded') for x in range(1, 11)]
    drop, used, _, _ = budget.plan(items, limit=20, quota_used=20)
    assert used == 20 and len(drop) >= 3


def test_delete_requires_read_and_fresh_metadata(monkeypatch):
    records = [runtime('runtime-config'), runtime('registry-pull')]
    records += [helm(n, 'deployed' if n == 12 else 'superseded') for n in range(1, 13)]
    commands = []
    monkeypatch.setattr(budget, 'read_all', lambda _ns: records)
    monkeypatch.setattr(budget, 'quota', lambda _ns, fallback: (20, 20, 'resourcequota'))

    def fake_run(args):
        commands.append(args)
        if args[:3] == ['kubectl', 'auth', 'can-i']:
            return 'yes\n'
        if args[3:5] == ['get', 'secret']:
            item = next(x for x in records if x.name == args[5])
            return '\t'.join([item.name, item.kind, item.owner,
                              item.release, item.version_label, item.status])
        return 'secret deleted\n'
    monkeypatch.setattr(budget, '_run', fake_run)
    result = budget.execute('g02-staging', dry_run=True, limit=20)
    assert result['mode'] == 'dry_run'
    assert commands == []  # no delete/auth/metadata recheck in dry-run
    # Test validation: selected old records cannot silently change status.
    drop, _, _, _ = budget.plan(records, limit=20, quota_used=20)
    if drop:
        original = drop[0]
        def stale(args):
            return '\t'.join([original.name, original.kind, original.owner,
                              original.release, original.version_label, 'deployed'])
        monkeypatch.setattr(budget, '_run', stale)
        with pytest.raises(budget.BudgetBlocked, match='changed_during_cleanup'):
            budget._confirm_unchanged('g02-staging', original)


def test_reject_unapproved_namespace_before_cluster_access(monkeypatch):
    monkeypatch.setattr(budget, 'read_all', lambda _ns: pytest.fail('must not call cluster'))
    with pytest.raises(budget.BudgetBlocked, match='invalid_namespace'):
        budget.execute('kube-system', dry_run=False, limit=20)


def test_jsonpath_only_requests_metadata_not_secret_data():
    assert '.data' not in budget.ALL_SECRETS
    assert '.data' not in budget.ONE_SECRET


def fixture_files(root: Path):
    (root / '.gitlab/ci').mkdir(parents=True)
    (root / 'ci').mkdir(parents=True)
    (root / '.gitlab/ci/release.yml').write_text('''deploy-staging:
  stage: deploy_staging
  script:
    - |
      kubectl get secret registry-pull runtime-config -n "$NAMESPACE" -o name >/dev/null

      helm upgrade --install "$RELEASE_NAME" deploy/chart \\
        --atomic --wait --timeout 8m --history-max 20

deploy-model-staging:
  stage: deploy_model_staging
  script:
    - |
      kubectl get secret registry-pull runtime-config -n "$NAMESPACE" -o name >/dev/null
      helm upgrade --install promiseops deploy/chart \\
        --atomic --wait --timeout 8m --history-max 20
''')
    (root / 'ci/deploy_production.sh').write_text('''#!/bin/sh
set -eu
helm upgrade --install "$RELEASE_NAME" "$CHART" \\
  --atomic --wait --timeout 8m --history-max 20
helm upgrade "$RELEASE_NAME" "$CHART" \\
  --atomic --wait --timeout 8m --history-max 20
''')
    (root / '.gitlab/ci/inspect.yml').write_text('''inspect-own-namespaces:
  stage: inspect

inspect-api-endpoint:
  stage: inspect

inspect-training-failure:
  stage: inspect

inspect-prod-resources:
  script:
    - |
      echo "Inspection completed. No resources changed."

inspect-mlflow-ci-access:
  stage: inspect
''')
    (root / '.gitlab/ci/inspect-deploy-failure.yml').write_text('''inspect-deploy-failure:
  script:
    - |
      helm -n "$NAMESPACE" status promiseops || true

inspect-deploy-rs:
  stage: inspect

inspect-runtime-serviceaccount:
  stage: inspect
''')


def test_installer_fixes_both_staging_and_production_and_removes_only_duplicates(tmp_path):
    fixture_files(tmp_path)
    before = {p: p.read_bytes() for p in [tmp_path / f for f in
              ['.gitlab/ci/release.yml', 'ci/deploy_production.sh',
               '.gitlab/ci/inspect.yml', '.gitlab/ci/inspect-deploy-failure.yml']]}
    changed = installer.apply(tmp_path, check=True)
    assert len(changed) == 4
    assert all(p.read_bytes() == body for p, body in before.items())
    installer.apply(tmp_path, check=False)
    release = (tmp_path / '.gitlab/ci/release.yml').read_text()
    prod = (tmp_path / 'ci/deploy_production.sh').read_text()
    inspect = (tmp_path / '.gitlab/ci/inspect.yml').read_text()
    failure = (tmp_path / '.gitlab/ci/inspect-deploy-failure.yml').read_text()
    assert release.count('python ci/helm_secret_budget.py --namespace') == 2
    assert prod.count('python ci/helm_secret_budget.py --namespace') == 2
    assert release.count('--history-max 5') == 2
    assert prod.count('--history-max 5') == 2
    assert '--history-max 20' not in release + prod
    assert 'inspect-own-namespaces:' not in inspect
    assert 'inspect-deploy-rs:' not in failure
    assert 'inspect-api-endpoint:' in inspect
    assert 'inspect-runtime-serviceaccount:' in failure
    assert installer.apply(tmp_path, check=True) == []
    assert installer.apply(tmp_path, check=False) == []
    import yaml
    for file in (tmp_path / '.gitlab/ci').glob('*.yml'):
        assert isinstance(yaml.safe_load(file.read_text()), dict)


def test_installer_refuses_unknown_layout_without_partial_writes(tmp_path):
    fixture_files(tmp_path)
    target = tmp_path / 'ci/deploy_production.sh'
    target.write_text('helm upgrade a stranger\n')
    original = {path: path.read_bytes() for path in [tmp_path / '.gitlab/ci/release.yml', target,
                tmp_path / '.gitlab/ci/inspect.yml', tmp_path / '.gitlab/ci/inspect-deploy-failure.yml']}
    with pytest.raises(ValueError):
        installer.apply(tmp_path, check=False)
    assert all(p.read_bytes() == original[p] for p in original)


def test_apply_prunes_revisions_only_and_rechecks_every_secret(monkeypatch):
    records = [runtime('runtime-config'), runtime('registry-pull')]
    records += [helm(n, 'deployed' if n == 18 else 'superseded') for n in range(1, 19)]
    actions = []
    monkeypatch.setattr(budget, 'read_all', lambda _namespace: list(records))
    monkeypatch.setattr(budget, 'quota', lambda _namespace, fallback: (20, len(records), 'resourcequota'))

    def fake_run(args):
        if args[:3] == ['kubectl', 'auth', 'can-i']:
            return 'yes\n'
        if 'get' in args and 'secret' in args:
            candidate = next(s for s in records if s.name == args[5])
            return '\t'.join((candidate.name, candidate.kind, candidate.owner,
                              candidate.release, candidate.version_label, candidate.status))
        if 'delete' in args:
            name = args[5]
            selected = next(s for s in records if s.name == name)
            assert selected.status == 'superseded'
            records.remove(selected)
            actions.append(name)
            return 'deleted\n'
        pytest.fail('unexpected kubectl call: ' + repr(args))

    monkeypatch.setattr(budget, '_run', fake_run)
    result = budget.execute('g02-staging', dry_run=False, limit=20)
    assert result['planned_superseded_deletions'] == len(actions)
    assert len(records) <= 17
    assert set(actions).isdisjoint({'runtime-config', 'registry-pull', 'sh.helm.release.v1.promiseops.v18'})
    assert set(actions).isdisjoint({f'sh.helm.release.v1.promiseops.v{i}' for i in [16,17]})
