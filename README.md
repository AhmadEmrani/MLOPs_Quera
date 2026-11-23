# PromiseOps — MLflow CI Read-Only Inspection

For branch `refactor/codesAndLifeCycle` (verified existing `.gitlab/ci/inspect.yml` layout).

The existing `inspect-ci-config-presence` already reports whether `MLFLOW_TRACKING_URI` is visible. Use that first; no code change is needed to run it on `main`.

This patch adds `inspect-mlflow-ci-access`, a separate manual GitLab `inspect` stage job. It diagnoses variable visibility, protected ref, DNS/direct TCP and read-only MLflow API access, without exposing the URI or credentials or changing the registry.

Apply at repository root:

```bash
unzip -o promiseops-mlflow-inspect-job.zip -d .
python ci/install_mlflow_ci_inspect.py --check
python ci/install_mlflow_ci_inspect.py
python -m pytest -q tests/test_mlflow_ci_inspect.py
git diff --check
```

After pushing/merging to `main`, manually run `inspect-mlflow-ci-access` and review `reports/mlflow/ci-access.json` in the job's artifacts. The check does not replace the `golden-mlflow-evidence` job, which is the actual MLflow write test.

**Security:** This ZIP contains no endpoint, secret or credential. The installer is idempotent, modifies only `.gitlab/ci/inspect.yml`, and refuses unrecognized file layouts.
