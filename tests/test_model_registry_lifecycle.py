"""No MLflow server or Kubernetes context required for these safety tests."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from ci import model_registry_lifecycle as lifecycle
from ci import apply_registry_lifecycle as patcher


SHA = 'a' * 64
IMG = 'registry.example.com/g02/promiseops/app@sha256:' + 'b' * 64


class NotFound(Exception):
    error_code = 'RESOURCE_DOES_NOT_EXIST'


class FakeClient:
    def __init__(self, champion=None, challenger='9'):
        self.aliases = {'challenger': challenger}
        if champion is not None:
            self.aliases['champion'] = champion
        self.tags = {
            'artifact_sha256': SHA, 'quality_gate': 'passed',
            'selection_status': 'selected', 'approval_status': 'pending',
            'algorithm': 'random_forest', 'source_commit': 'c' * 40,
            'dataset_manifest_sha256': 'd' * 64,
            'feature_contract_sha256': 'e' * 64,
            'validation_cohort_hash': 'f' * 64,
        }
        self.events = []

    def get_model_version(self, name, version):
        return SimpleNamespace(version=version, status='READY', tags=dict(self.tags), run_id='run-123')

    def get_model_version_by_alias(self, name, alias):
        value = self.aliases.get(alias)
        if value is None:
            raise NotFound('alias missing')
        return SimpleNamespace(version=value)

    def set_model_version_tag(self, name, version, key, value):
        self.tags[key] = value
        self.events.append(('tag', key, value))

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version
        self.events.append(('alias', alias, version))


def selection(champion='absent', challenger='9'):
    return {
        'selected_model_version': '9', 'artifact_sha256': SHA,
        'public_gates': 'passed', 'model_name': 'random_forest',
        'mlflow_run_id': 'run-123', 'source_commit': 'c' * 40,
        'cohort_hash': 'f' * 64,
        'file_sha256': {'model.joblib': SHA,
                        'dataset_manifest.json': 'd' * 64,
                        'feature_contract.json': 'e' * 64},
        'champion_at_release': champion,
        'challenger_at_release': challenger,
    }


def smoke(digest=SHA):
    return {'status': 'passed', 'model_version': 'sha256:' + digest,
            'checks': ['prediction', 'readyz', 'metrics_auth']}


def deployment(image=IMG, replicas=2):
    return {
        'metadata': {'name': 'promiseops', 'generation': 9},
        'spec': {'replicas': replicas, 'template': {'spec': {
            'containers': [{'name': 'api', 'image': image}]
        }}},
        'status': {'observedGeneration': 9, 'updatedReplicas': replicas,
                   'readyReplicas': replicas, 'availableReplicas': replicas},
    }


def test_first_promotion_requires_successful_smoke_and_pinned_deployment():
    client = FakeClient()
    result = lifecycle.promote(client, 'promiseops-late-delivery', selection(),
                               '9', SHA, smoke(), deployment(), IMG, ci_job_id='123')
    assert result['champion_after'] == '9'
    assert client.aliases['champion'] == '9'
    assert client.tags['approval_status'] == 'approved'
    assert client.tags['previous_champion_version'] == 'absent'
    assert client.events[-1] == ('alias', 'champion', '9')


@pytest.mark.parametrize('bad_smoke', [
    {'status': 'failed', 'model_version': 'sha256:' + SHA, 'checks': ['prediction']},
    smoke('b' * 64),
    {'status': 'passed', 'model_version': 'sha256:' + SHA, 'checks': []},
])
def test_smoke_failure_blocks_every_registry_mutation(bad_smoke):
    client = FakeClient()
    with pytest.raises(lifecycle.RegistrySafetyError):
        lifecycle.promote(client, 'm', selection(), '9', SHA,
                          bad_smoke, deployment(), IMG)
    assert not client.events


def test_promotion_rejects_wrong_image():
    client = FakeClient()
    with pytest.raises(lifecycle.RegistrySafetyError, match='image digest'):
        lifecycle.promote(client, 'm', selection(), '9', SHA, smoke(),
                          deployment(image=IMG.replace('b' * 64, 'c' * 64)), IMG)
    assert not client.events


def test_unavailable_or_outdated_replicas_reject_promotion():
    client = FakeClient()
    data = deployment(replicas=3)
    data['status']['availableReplicas'] = 2
    with pytest.raises(lifecycle.RegistrySafetyError, match='availableReplicas'):
        lifecycle.promote(client, 'm', selection(), '9', SHA, smoke(), data, IMG)
    assert not client.events


def test_changed_champion_since_candidate_review_blocks_preflight():
    client = FakeClient(champion='12')
    with pytest.raises(lifecycle.RegistrySafetyError, match='Champion moved'):
        lifecycle.preflight(client, 'm', selection(), '9', SHA)
    assert not client.events


def test_changed_challenger_since_review_blocks_preflight():
    client = FakeClient(challenger='11')
    with pytest.raises(lifecycle.RegistrySafetyError, match='Challenger moved'):
        lifecycle.preflight(client, 'm', selection(), '9', SHA)
    assert not client.events


def test_artifact_mismatch_blocks_all_registry_writes():
    client = FakeClient()
    client.tags['artifact_sha256'] = 'b' * 64
    with pytest.raises(lifecycle.RegistrySafetyError, match='artifact checksum'):
        lifecycle.preflight(client, 'm', selection(), '9', SHA)
    assert not client.events


def test_no_nonapproved_version_can_be_reconciled():
    client = FakeClient(champion='8')
    with pytest.raises(lifecycle.RegistrySafetyError, match='approval status'):
        lifecycle.reconcile(client, 'm', '9', smoke(), deployment(), IMG)
    assert not client.events
    assert client.aliases['champion'] == '8'


def test_reconcile_approved_prior_version_after_rollback():
    client = FakeClient(champion='12')
    client.tags['approval_status'] = 'approved'
    result = lifecycle.reconcile(client, 'm', '9', smoke(), deployment(), IMG)
    assert result['champion_before'] == '12'
    assert result['champion_after'] == '9'
    assert client.aliases['champion'] == '9'
    assert client.tags['production_image'] == IMG


def test_idempotent_promotion_preserves_original_prior_champion():
    client = FakeClient()
    lifecycle.promote(client, 'm', selection(), '9', SHA, smoke(), deployment(), IMG)
    original = client.tags['previous_champion_version']
    client.events.clear()
    result = lifecycle.promote(client, 'm', selection(), '9', SHA, smoke(), deployment(), IMG)
    assert result['champion_after'] == '9'
    assert client.tags['previous_champion_version'] == original
    assert client.events == []


def test_preflight_requires_selection_run_and_companion_hashes():
    client = FakeClient()
    bad = selection()
    bad['file_sha256']['feature_contract.json'] = '0' * 64
    with pytest.raises(lifecycle.RegistrySafetyError, match='feature_contract_sha256'):
        lifecycle.preflight(client, 'm', bad, '9', SHA)
    assert not client.events


def test_install_ci_fails_closed_on_unknown_layout():
    with pytest.raises(RuntimeError, match='Expected existing'):
        patcher.apply('not a CI release YAML')


def test_install_ci_idempotent_on_realistic_yaml_excerpt():
    original = '''rollback-production:
  script:
    - |
      sh ci/deploy_production.sh "$NAMESPACE" "$ROLLBACK_DIGEST_VALUE"
      PROMISEOPS_SMOKE_EVIDENCE_FILE="deployment-evidence/rollback-smoke.json" \\
        sh ci/run_api_smoke.sh "$NAMESPACE"
  artifacts:
    paths:
      - deployment-evidence/rollback-smoke.json

build-model-image:
  stage: model_image

promote-model-production:
  script:
    - |
      sh ci/deploy_production.sh "$NAMESPACE" "$DIGEST"
      set -a
      . dist/model-release/release.env
      set +a
      PROMISEOPS_SMOKE_EVIDENCE_FILE="deployment-evidence/model-production-smoke.json" \\
        sh ci/run_api_smoke.sh "$NAMESPACE"
  artifacts:
    paths:
      - deployment-evidence/model-production-smoke.json
'''
    result = patcher.apply(original)
    assert 'ci/model_registry_lifecycle.py promote' in result
    assert 'ci/model_registry_lifecycle.py preflight' in result
    assert 'ci/model_registry_lifecycle.py reconcile' in result
    assert 'reconcile-champion-production:' in result
    assert patcher.apply(result) == result


def test_alias_move_refuses_unexpected_concurrent_change():
    client = FakeClient(champion='12')
    with pytest.raises(lifecycle.RegistrySafetyError, match='changed during approval'):
        lifecycle.move_alias(client, 'm', '9', prior='8')
    assert client.aliases['champion'] == '12'
    assert not client.events


def test_reconciliation_requires_matching_runtime_model_sha():
    client = FakeClient(champion='12')
    client.tags['approval_status'] = 'approved'
    with pytest.raises(lifecycle.RegistrySafetyError, match='/predict model SHA'):
        lifecycle.reconcile(client, 'm', '9', smoke(digest='9'*64), deployment(), IMG)
    assert client.aliases['champion'] == '12'
    assert not client.events


def test_installer_result_is_valid_gitlab_yaml():
    import ast
    from pathlib import Path
    import yaml

    src = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    fixture = next(node for node in src.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == 'test_install_ci_idempotent_on_realistic_yaml_excerpt')
    original = next(ast.literal_eval(stmt.value) for stmt in fixture.body
                    if isinstance(stmt, ast.Assign)
                    and any(isinstance(v, ast.Name) and v.id == 'original'
                            for v in stmt.targets))
    updated = yaml.safe_load(patcher.apply(original))
    assert 'promote-model-production' in updated
    assert 'rollback-production' in updated
    assert 'reconcile-champion-production' in updated
    assert updated['reconcile-champion-production']['rules'][0]['when'] == 'manual'


def test_challenger_may_change_after_successful_predeployment_preflight():
    client = FakeClient()
    lifecycle.preflight(client, 'm', selection(), '9', SHA)  # Before deployment.
    client.aliases['challenger'] = '10'  # Training continued during rollout.
    result = lifecycle.promote(client, 'm', selection(), '9', SHA,
                               smoke(), deployment(), IMG)
    assert result['champion_after'] == '9'
    assert client.aliases['challenger'] == '10'


def test_other_champion_between_preflight_and_promotion_fails_closed():
    client = FakeClient()
    lifecycle.preflight(client, 'm', selection(), '9', SHA)
    client.aliases['champion'] = '10'
    with pytest.raises(lifecycle.RegistrySafetyError, match='Champion moved'):
        lifecycle.promote(client, 'm', selection(), '9', SHA,
                          smoke(), deployment(), IMG)
    assert not client.events
