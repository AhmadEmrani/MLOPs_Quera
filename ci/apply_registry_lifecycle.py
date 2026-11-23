#!/usr/bin/env python3
"""Apply narrowly scoped, idempotent CI changes for champion lifecycle.

Run this once in the *existing* PromiseOps repository after extracting patch.
Fails on unexpected YAML layout without rewriting the original file.
"""

from pathlib import Path

FILE = Path('.gitlab/ci/release.yml')


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f'CI layout changed ({description}); review release.yml before applying')
    return source.replace(old, new, 1)


def apply(source: str) -> str:
    if 'ci/model_registry_lifecycle.py promote' in source:
        print('Champion lifecycle CI integration already installed; no changes.')
        return source
    # Guard the known structural boundaries instead of replacing the whole YAML.
    if 'promote-model-production:\n' not in source or 'rollback-production:\n' not in source:
        raise RuntimeError('Expected existing promote-model-production and rollback-production jobs')
    preflight = '''      set -a
      . dist/model-release/release.env
      set +a
      : "${MLFLOW_TRACKING_URI:?Protected MLFLOW_TRACKING_URI is required}"
      sh ci/setup_registry_env.sh
      .registry-venv/bin/python ci/model_registry_lifecycle.py preflight \\
        --selection dist/model-release/selection.json \\
        --version "$MODEL_MLFLOW_VERSION" \\
        --sha256 "$MODEL_ARTIFACT_SHA256" \\
        --evidence deployment-evidence/model-registry-preflight.json
      sh ci/deploy_production.sh "$NAMESPACE" "$DIGEST"
'''
    old = '''      sh ci/deploy_production.sh "$NAMESPACE" "$DIGEST"
      set -a
      . dist/model-release/release.env
      set +a
'''
    # The preflight evidence directory must exist prior to invoking the CLI.
    preflight = preflight.replace('      .registry-venv/bin/python ci/model_registry_lifecycle.py preflight',
                                  '      mkdir -p deployment-evidence\n      .registry-venv/bin/python ci/model_registry_lifecycle.py preflight')
    promote_start = source.index('promote-model-production:\n')
    head, promote = source[:promote_start], source[promote_start:]
    promote = replace_once(promote, old, preflight, 'promotion preflight anchor')
    smoke = '''      PROMISEOPS_SMOKE_EVIDENCE_FILE="deployment-evidence/model-production-smoke.json" \\
        sh ci/run_api_smoke.sh "$NAMESPACE"
'''
    extra = '''      .registry-venv/bin/python ci/model_registry_lifecycle.py promote \\
        --selection dist/model-release/selection.json \\
        --version "$MODEL_MLFLOW_VERSION" \\
        --sha256 "$MODEL_ARTIFACT_SHA256" \\
        --smoke deployment-evidence/model-production-smoke.json \\
        --namespace "$NAMESPACE" \\
        --expected-image "${CI_REGISTRY_IMAGE}/app@${DIGEST}" \\
        --evidence deployment-evidence/model-registry-promotion.json
'''
    promote = replace_once(promote, smoke, smoke + extra, 'promotion successful smoke')
    artifact = '''      - deployment-evidence/model-production-smoke.json
'''
    promote = replace_once(promote, artifact,
                           artifact + '''      - deployment-evidence/model-registry-preflight.json
      - deployment-evidence/model-registry-promotion.json
''', 'promotion evidence')
    source = head + promote

    # Rollback: optional version-aware alias sync. No MLflow availability is
    # required for an emergency rollback unless the operator requests sync.
    rb_start = source.index('rollback-production:\n')
    rb_end = source.find('\nbuild-model-image:', rb_start)
    if rb_end < 0:
        raise RuntimeError('Expected build-model-image after rollback-production')
    before, rollback, after = source[:rb_start], source[rb_start:rb_end], source[rb_end:]
    old_rb_deploy = '''      sh ci/deploy_production.sh "$NAMESPACE" "$ROLLBACK_DIGEST_VALUE"
'''
    replacement_rb_deploy = '''      if [ -n "${ROLLBACK_MLFLOW_VERSION:-}" ]; then
        sh ci/setup_registry_env.sh
        .registry-venv/bin/python ci/model_registry_lifecycle.py check-approved \\
          --version "$ROLLBACK_MLFLOW_VERSION" \\
          --evidence deployment-evidence/rollback-registry-preflight.json
      fi
      sh ci/deploy_production.sh "$NAMESPACE" "$ROLLBACK_DIGEST_VALUE"
'''
    # mkdir before version-check but preserve rest of rollback path.
    replacement_rb_deploy = replacement_rb_deploy.replace(
        '      if [ -n "${ROLLBACK_MLFLOW_VERSION:-}" ]; then',
        '      mkdir -p deployment-evidence\n      if [ -n "${ROLLBACK_MLFLOW_VERSION:-}" ]; then', 1)
    rollback = replace_once(rollback, old_rb_deploy, replacement_rb_deploy, 'rollback deployment')
    rb_smoke = '''      PROMISEOPS_SMOKE_EVIDENCE_FILE="deployment-evidence/rollback-smoke.json" \\
        sh ci/run_api_smoke.sh "$NAMESPACE"
'''
    rb_extra = '''      if [ -n "${ROLLBACK_MLFLOW_VERSION:-}" ]; then
        .registry-venv/bin/python ci/model_registry_lifecycle.py reconcile \\
          --version "$ROLLBACK_MLFLOW_VERSION" \\
          --smoke deployment-evidence/rollback-smoke.json \\
          --namespace "$NAMESPACE" \\
          --expected-image "${CI_REGISTRY_IMAGE}/app@${ROLLBACK_DIGEST_VALUE}" \\
          --evidence deployment-evidence/rollback-registry-reconciliation.json
      else
        echo 'NOTE: Emergency rollback completed. No MLflow alias was moved because ROLLBACK_MLFLOW_VERSION was not provided.'
        echo 'If the model changed, run reconcile-champion-production after verifying the registered version.'
      fi
'''
    rollback = replace_once(rollback, rb_smoke, rb_smoke + rb_extra, 'rollback successful smoke')
    rb_artifact = '      - deployment-evidence/rollback-smoke.json\n'
    rollback = replace_once(rollback, rb_artifact, rb_artifact + '''      - deployment-evidence/rollback-registry-preflight.json
      - deployment-evidence/rollback-registry-reconciliation.json
''', 'rollback evidence')
    source = before + rollback + after

    # Recovery/manual reconciliation also covers existing legacy deployments,
    # but only if the exact live model matches a previously approved version.
    source += '''
# Manual, auditable alias recovery after rollback or an interrupted promotion.
# Cannot approve pending/unreviewed models or change the running image.
reconcile-champion-production:
  stage: inspect
  needs: []
  resource_group: promiseops-production
  environment:
    name: promiseops-production
    action: verify
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
  allow_failure: true
  script:
    - |
      set -eu
      set +x
      : "${REGISTRY_MODEL_VERSION:?Provide a previously approved MLflow Model Version}"
      [ "${REGISTRY_RECONCILE_APPROVED:-}" = yes ] || { echo 'Set REGISTRY_RECONCILE_APPROVED=yes' >&2; exit 1; }
      TEAM_NAME="${TEAM:-g02}"
      case "$TEAM_NAME" in g0[1-7]) ;; *) echo 'Invalid team' >&2; exit 1 ;; esac
      NAMESPACE="${TEAM_NAME}-prod"
      mkdir -p deployment-evidence
      sh ci/setup_registry_env.sh
      .registry-venv/bin/python ci/model_registry_lifecycle.py check-approved \\
        --version "$REGISTRY_MODEL_VERSION" \\
        --evidence deployment-evidence/manual-registry-preflight.json
      IMAGE="$(kubectl get deployment promiseops -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
      PROMISEOPS_SMOKE_EVIDENCE_FILE="deployment-evidence/manual-registry-smoke.json" \\
        sh ci/run_api_smoke.sh "$NAMESPACE"
      .registry-venv/bin/python ci/model_registry_lifecycle.py reconcile \\
        --version "$REGISTRY_MODEL_VERSION" \\
        --smoke deployment-evidence/manual-registry-smoke.json \\
        --namespace "$NAMESPACE" \\
        --expected-image "$IMAGE" \\
        --evidence deployment-evidence/manual-registry-reconciliation.json
  artifacts:
    when: always
    expire_in: 30 days
    paths:
      - deployment-evidence/manual-registry-preflight.json
      - deployment-evidence/manual-registry-smoke.json
      - deployment-evidence/manual-registry-reconciliation.json
'''
    return source


def main() -> None:
    if not FILE.is_file():
        raise SystemExit(f'Missing {FILE}; run from existing PromiseOps project root')
    before = FILE.read_text(encoding='utf-8')
    try:
        after = apply(before)
    except RuntimeError as error:
        raise SystemExit(f'No files modified: {error}') from error
    if before != after:
        FILE.write_text(after, encoding='utf-8')
        print(f'Patched {FILE}; review git diff before committing.')


if __name__ == '__main__':
    main()
