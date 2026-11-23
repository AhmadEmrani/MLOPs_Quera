# Read-only MLflow CI inspection

## Which existing job helps?

`inspect-ci-config-presence` reports whether `MLFLOW_TRACKING_URI` is visible
in the GitLab Runner (it does **not** check connectivity). In contrast,
`inspect-training-failure` inspects `g02-jobs` Worker Pods. Worker Pods are
created by the platform's training gateway and receive a separate set of
runtime variables; an MLflow URI there **does not imply** the same URI will be
injected into GitLab CI jobs.

`inspect-mlflow-ci-access` is a complementary, manually triggered `inspect`
stage job on `main`. It reads no Kubernetes Secrets or Production data,
creates no MLflow Run and does not mutate the Model Registry.

The job first inspects the presence of fixed CI variables, the branch protected
flag, and the presence of a GitLab environment (but not its name). It then
resolves the URI from `MLFLOW_TRACKING_URI`, `PROMISEOPS_MLFLOW_TRACKING_URI`,
or `MLFLOW_TRACKING_URI_FILE`, including GitLab File-type variables. It checks
DNS and direct TCP reachability as diagnostics and, if configuration is
available, installs the existing offline MLflow dependency wheelhouse and calls
`MlflowClient.search_experiments(max_results=1)` (read-only).

**Important:** TCP checks are direct and may fail behind a proxy even if the
MLflow API succeeds. Treat the MLflow API read result as the decisive check.
This job does not test write permissions required to create a golden evidence
Run; that is established by the separate manual `golden-mlflow-evidence` job.

## Apply to current branch

```bash
unzip -o promiseops-mlflow-inspect-job.zip -d .
python ci/install_mlflow_ci_inspect.py --check
python ci/install_mlflow_ci_inspect.py
python -m pytest -q tests/test_mlflow_ci_inspect.py
git diff --check
```

This installer adds the job to `.gitlab/ci/inspect.yml`. The root
`.gitlab-ci.yml` already includes it; no include change is required.

## Execute

After merge, open a `main` GitLab pipeline and manually run
`inspect-mlflow-ci-access`. For the current failure, also run
`inspect-ci-config-presence` to see which variables are visible. Each job's
environment and context can differ, particularly when an environment-scoped
variable is used, so **use the new job for the definitive CI scope check**.

Expected artifact: `reports/mlflow/ci-access.json`.

Interpretation:
- `tracking_uri_status=not_visible_to_ci_job`: inspect GitLab Settings > CI/CD
  > Variables. If Protected is enabled, `main` must be a protected branch;
  environment scope `*` is recommended for jobs without `environment:`.
- `invalid_http_endpoint_or_embedded_credentials`: fix variable formatting;
  use separately stored credentials rather than a user:password in the URI.
- `network.dns=failed` or `network.direct_tcp=failed`: inspect runner egress,
  DNS or proxy; the read-only MLflow request result takes precedence.
- `mlflow_read.status=failed`: inspect authentication/TLS/authorization or
  connectivity with the platform team.
- `mlflow_read.status=passed`: read-only MLflow access from CI was verified;
  this does not prove create-run/write permission.

No URI, hostname, credential, exception message, or raw tokens are written to
logs or artifacts. `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, and
`MLFLOW_TRACKING_TOKEN` are reported by presence only.
