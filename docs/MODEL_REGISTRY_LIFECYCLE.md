# PromiseOps — MLflow Model Registry lifecycle

## Scope and safety contract

This patch changes **GitLab CI and MLflow Registry only**. It never edits the running API source, model, Docker image, Helm manifests, training splits, or PostgreSQL schema. Existing Production is not redeployed when these files are committed.

* Training stores Model Versions and can assign the `challenger` alias, not `champion`.
* A manual model-release approval can deploy a reviewed candidate to Production using a digest-pinned image.
* **Before deployment:** verify the selected model and evidence from `selection.json`, Registry version SHA/tags and the champion/challenger snapshots recorded during review.
* **After deployment and a passing authenticated smoke:** independently inspect Production Deployment image and ready/updated replicas, check `model_version=sha256:<reviewed model.joblib SHA>` in the smoke result, and only then record approval metadata and set `champion`.
* Re-read alias after the write. The app continues to use its bundled image, not a floating MLflow alias.
* MLflow does **not** provide an atomic alias compare-and-swap across Kubernetes and the Registry. A concurrent external edit or unavailable MLflow server can block synchronization after deployment. The job goes red and the explicit manual recovery workflow must be used. **A failed post-deploy alias step does not roll back the Production image.**

## How to apply

From the existing repository root, after extracting the ZIP:

```sh
python ci/apply_registry_lifecycle.py
PYTHONPATH=src python -m pytest -q tests/test_model_registry_lifecycle.py
git diff --check
git diff -- .gitlab/ci/release.yml
git status --short
```

The installer modifies *only* the `promote-model-production` and `rollback-production` jobs in `.gitlab/ci/release.yml`, and adds `reconcile-champion-production`. It refuses unrecognized CI layouts instead of overwriting the original file. It is idempotent.

Run the normal repository regression suite and validate the pipeline in GitLab's CI Lint before merging.

## Requirements

* Protected GitLab CI/CD variable `MLFLOW_TRACKING_URI` accessible on `main` (the existing `prepare-model-release` already requires it).
* Existing CI toolbox, `/opt/promiseops/tracking-wheelhouse`, `requirements-tracking.lock`, and offline `uv` installation.
* CI Kubernetes identity can `get deployment promiseops` in `g02-prod` (the existing resource inspection job already does so).
* CI has write permission for MLflow Registered Model aliases and Model Version tags.
* `resource_group: promiseops-production` must be kept on every Production change and reconciliation job.
* Protect the `promiseops-production` environment so only authorized reviewers can start manual Production actions.

## Model promotion

No new manual step: start the existing `promote-model-production` job (after the training gate, `prepare-model-release`, model image build and model staging verification). The job will now:

1. Prepare the offline tracking environment.
2. Run `model_registry_lifecycle.py preflight` (read-only).
3. Deploy the reviewed immutable image.
4. Execute `run_api_smoke.sh` with evidence.
5. Run `model_registry_lifecycle.py promote` (write to MLflow only after all checks).
6. Save `deployment-evidence/model-registry-preflight.json` and `deployment-evidence/model-registry-promotion.json` as artifacts.

The verified alias is `champion` on the same registered model (`PROMISEOPS_REGISTERED_MODEL_NAME` or default `promiseops-late-delivery`). Tags include `approval_status=approved`, pinned Production image, CI job ID, and previous champion version. Re-running a previously completed promotion is idempotent (it will not erase the original previous-version tag).

**Important:** `deploy-production` for code-only changes does not change the MLflow Champion. This is correct because the same bundled model may be deployed with a different image digest.

## Rollback

Existing `rollback-production` still supports urgent image rollback with `ROLLBACK_DIGEST` alone. This cannot safely infer an MLflow version from an image digest; if no MLflow version is supplied, the job prints a clear warning and **does not move `champion`**. You must then reconcile if the model has changed.

To sync the alias as part of rollback, additionally pass:

```text
ROLLBACK_MLFLOW_VERSION=<previously approved registered model version>
```

It will check the version's prior approval **before** deploying, and after rollback smoke require the actually deployed model SHA to match the Registry version. If they differ, alias reconciliation is blocked; no guessed version is ever assigned. The image rollback can have succeeded even if the alias action failed; consult the deployment evidence and take recovery action.

## Manual recovery / initial bootstrap

Run the added manual job `reconcile-champion-production` on `main` with:

```text
REGISTRY_MODEL_VERSION=<previously approved MLflow version>
REGISTRY_RECONCILE_APPROVED=yes
```

The job reads the currently running digest-pinned Deployment and performs Production smoke. It only moves Champion if the live model's immutable SHA matches the Registry tag and the Model Version was already selected and explicitly marked `approved`. It **cannot** approve a pending model.

If Production still serves the initial starter bundle but no exact approved Model Version exists for it, do **not** invent an MLflow number and do **not** assign `champion` yet. Use the normal reviewed-model promotion workflow when the first qualifying candidate exists. This is an intentional fail-closed state.

## Expected evidence and invariants

* A rejected/failed candidate, staging failure, failed deployment or failed smoke never moves `champion`.
* `champion` always refers to the exact reviewed, verified model artifact (when this workflow sets it).
* `challenger` is managed by Training; this patch never moves it.
* All post-deployment Registry mutations require the same release `selection.json`, version tags, checksum, successful `/predict` and currently ready Deployment.
* Evidence files contain no API keys, database credentials, raw order IDs or feature values.
* A manual Registry edit outside the CI's resource group can race alias updates; there is no MLflow atomic compare-and-swap API. Require approved operators to use the CI workflow rather than the MLflow UI for champion updates.
