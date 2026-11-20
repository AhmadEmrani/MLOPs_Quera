Implement a Helm chart and CI pipeline. Staff provides runtime service account, registry-pull and runtime-config secrets, ingress, policies and namespace quotas. Your workloads need probes, bounded requests/limits, two production replicas, max-three HPA, an intentional batch CronJob, image digests and rollback. Use ClusterIP services; do not create namespace, RBAC, policies, ingress, hostPath, NodePort or privileged workloads.

The production namespace has a 3 CPU / 3 GiB budget. A 750m CPU / 768Mi API limit
allows three HPA replicas plus one rolling-update surge pod within that budget.
Keep two available replicas during model promotion. Drain requests before process
shutdown (10-second preStop and 45-second termination grace). Separate readiness
from liveness, and never send traffic to a model that failed validation.

A 750m CPU / 768Mi scoring job fits alongside three steady API replicas. At that
scale, wait for an active scoring job to finish before a manual rollout so its
surge pod has room. Keep scoring jobs bounded, prevent overlapping CronJob runs,
and keep expensive training in the admitted training queue.

## GitLab deployment flow

The CI toolbox must contain `helm`, `kubectl`, `curl`, Python, and the BuildKit
client. GitLab CI uses the pre-authorized deploy identity; deployments target
the already-created `${TEAM}-staging` and `${TEAM}-prod` namespaces and never
create namespaces or cluster policy resources.

Set `PROMISEOPS_SMOKE_ORDER_ID` as a GitLab CI/CD variable to an order already
ingested into the team's database. It is an identifier, not a credential. The
smoke job reads `PROMISEOPS_API_KEY` from the existing `runtime-config` Secret
without printing it. `PROMISEOPS_RUNTIME_SERVICE_ACCOUNT` is optional and
defaults to `promiseops-api`; set it if staff provisioned a different runtime
account. The observed registry pull Secret name is `registry-pull`.

On `main`, CI validates the chart, builds and pushes the API image, and saves
the BuildKit digest in `build.json`. `deploy-staging` passes that exact digest
and `$CI_REGISTRY_IMAGE/app` to Helm. `verify-staging-api` calls the internal
Service and checks health, readiness, metrics authentication, prediction,
authentication and validation errors, unknown orders, request-ID replay, and
conflict handling. The Service is internal at
`http://promiseops.${TEAM}-staging.svc:80`; the runner's network policy must
allow CI-to-staging traffic. The smoke evidence and deployed image digest are
kept as job artifacts.

`deploy-production` is a manual GitLab deployment available only after the
staging checks pass. Protect the `promiseops-production` environment in GitLab
to restrict who can approve and start it. Before changing the release, CI
suspends the scoring CronJob, waits for all nonterminal scoring Jobs, deploys
the digest with Helm's atomic rollback, verifies two available API replicas,
then resumes the schedule. `rollback-production` is a separate manual job; set
`ROLLBACK_DIGEST` to a previously recorded image digest. It uses the same
resource and readiness checks. Helm's `--atomic` also restores the previous
release if the rollout itself fails.

## Model bundle promotion

For a model candidate, start a `main` pipeline with `RUN_MODEL_RELEASE=true`.
After staging verification, run the manual `prepare-model-release` job. It
checks the selected MLflow challenger against the recorded same-cohort gates,
packages the exact bundle, and publishes an immutable Generic Package. This job
needs the protected `MLFLOW_TRACKING_URI` variable. CI then builds a separate
digest-pinned image containing that bundle, deploys it to staging, and runs the
same API checks. `promote-model-production` is the manual production approval
for that candidate. Its evidence records the MLflow model version, artifact
checksum, and image digest. The API loads the bundle from the image; serving
does not resolve MLflow aliases at startup.

For later code-only image builds, set `PROMISEOPS_RELEASE_PACKAGE_VERSION` to
the latest approved package version after a model promotion. It defaults to the
starter package version `cf9f672b`; leaving that default would put the starter
bundle into a later code-only build.

## Scheduled scoring

Production enables `promiseops-batch` at 02:00 cluster time. It runs the
`promiseops.scheduled_batch` command rather than the API image's Uvicorn entry
point. A PostgreSQL cursor advances through successive 24-hour source
availability windows and saves each JSON priority result, so retries and later
runs do not treat old orders as newly released. `concurrencyPolicy: Forbid`, a
database advisory lock, the one-hour deadline, and bounded resources prevent
overlapping or unbounded scoring. Scoring uses the database credential only;
the API key is not exposed to the CronJob.
