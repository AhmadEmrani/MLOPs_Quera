# MLflow access from GitLab CI — Golden and Model Registry

## Why training may work while Golden MLflow job fails

The course gateway creates training worker Pods and injects `MLFLOW_TRACKING_URI`
into those Pods (see `docs/CONTRACTS.md`). GitLab's `g02-ci` Kubernetes executor is
a **different environment**: it does not inherit the worker's variables.

The following jobs need a tracking URI inside GitLab CI:
`golden-mlflow-evidence`, `prepare-model-release`, `promote-model-production`,
`reconcile-champion-production`, and optionally `rollback-production` when the
MLflow version is supplied.

`ci/require_mlflow_ci_env.sh` is sourced by these jobs. It supports:

1. `MLFLOW_TRACKING_URI` (standard GitLab variable).
2. `PROMISEOPS_MLFLOW_TRACKING_URI` (optional alternative name).
3. GitLab File-type variables (contents read from the GitLab-provided temp path),
   or `MLFLOW_TRACKING_URI_FILE` (explicit path to an already-mounted file).

It does not guess the endpoint, read Production Secrets or print endpoint values.
It cannot add protected GitLab variables through GitHub.

## One-time setup required in GitLab

Go to **Project → Settings → CI/CD → Variables**. Add/verify:

- **Key:** `MLFLOW_TRACKING_URI`.
- **Value:** the real team MLflow HTTP(S) tracking endpoint supplied by the
  platform owner or your existing MLflow connection details (never invent).
- **Environment scope:** `*` (the manual Golden job has no Production environment).
- **Protected:** use only if `main` is a protected GitLab branch.
- **Masked:** if GitLab masking rules accept the endpoint value.

If MLflow uses authentication, add its credentials separately in masked or
protected GitLab variables such as `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` according to the server's supported scheme.
Never put credentials in source code, CLI arguments, or console logs.

If the variable exists but is missing during a job, review branch protection,
environment scope, inherited group/project variables, and whether a manual job
variable overrides it with an empty value. The helper prints only whether the
ref is protected, **not** the URI.

## Execution

1. Apply `ci/apply_mlflow_ci_access.py` to `.gitlab/ci/golden.yml`,
   `.gitlab/ci/release.yml`, and `ci/setup_registry_env.sh`.
2. Run `PYTHONPATH=src python -m pytest -q tests/test_mlflow_ci_env.py tests/test_mlflow_ci_install.py`.
3. Commit and sync the branch to GitLab; merge to main and validate CI Lint.
4. After `golden-contracts` passes, manually run `golden-mlflow-evidence`.
5. Check the separate `promiseops-golden-tests` MLflow experiment and artifacts.
6. Model Registry jobs use the same URI resolver when an approved release is run.

This fix changes **CI environment resolution only**. It does not connect to
Production, deploy a model, change Champion/Challenger or modify the database.
