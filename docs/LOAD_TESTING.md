# G02 production API load testing

Scope: this is an API **performance** test only, not a model-quality evaluation, not an HPA demonstration, and not a production model release.

## Source and cohort

`ci/select_load_orders.py` uses the existing `runtime-config` production DB credential to query PostgreSQL in read-only mode. A *post-validation* historical serving cohort is selected using:

- `orders.approved_at > validation_end`
- `orders.approved_at <= as_of`
- `orders.batch_available_at <= as_of`
- `EXISTS` at least one item and one payment
- no `outcomes`/labels are queried and no train/validation IDs are intentionally selected
- deterministic `ORDER BY md5(order_id)` with 200 unique IDs maximum

Defaults use `validation_end=2018-04-30T23:59:59-03:00` and `as_of=2018-06-29T23:59:59-03:00`, matching the team's previous Airflow cutoffs. Input timestamps must be timezone-aware, and are normalized to UTC for the SQL query. The database may already contain later published data: cutoff checks remain in force.

**Important:** API serving evaluates features using its own current serving-time logic, not the historical `as_of` passed to the selector. This test validates request performance and identity/availability, **not** historical replay fidelity or future-label accuracy. Unique request IDs intentionally persist new predictions. Reusing the same order ID with a new request ID is legitimate load traffic. Do not interpret these events as 1,800 *unique orders* in monitoring, or use them unfiltered for drift/evaluation cohort counts.

## CI execution

The job `load-test-production` is in `.gitlab/ci/load.yml`. Register its include without replacing existing CI changes using `python ci/install_load_job.py`:

1. Run only on `main` as a **manual** job, after the desired image has been deployed.
2. Confirm there is an authorized testing window. Production load must be permitted by the shared platform team.
3. First run with the default `PROMISEOPS_LOAD_PROFILE=pilot`: 1 RPS for 10 seconds (plus 5 warm-ups).
4. For the acceptance run, manually set `PROMISEOPS_LOAD_PROFILE=acceptance` **and** `PROMISEOPS_LOAD_WINDOW_APPROVED=yes` after approval: exactly 10 RPS for 180 seconds, 1,800 requests plus 5 warm-ups.
5. Download `reports/load/selection-*.json`, `reports/load/context-*.txt`, and `reports/load/load-*.json` from GitLab artifacts.
6. Record dashboard screenshots and the start/end time, image digest, cohort SHA256, traffic, p95, HTTP status counts, server/transport failures, and run result in the evidence index.

The runner must be able to read the existing `runtime-config` Secret in `g02-prod`, reach the PostgreSQL connection specified by it, and reach `http://promiseops.g02-prod.svc:80`. **If CI-to-DB or CI-to-Service network policies forbid access, the job fails safely. Do not bypass NetworkPolicies or grant cluster-wide credentials; request the platform-approved test route.**

Only report, cohort metadata and deployment context become artifacts. Private order-ID JSON lives in an ephemeral directory and is erased when the job finishes. `PROMISEOPS_DATABASE_URL` and `PROMISEOPS_API_KEY` never appear in an argument, artifact, or log. Disable shell trace before reading secrets.

Acceptance checks are implemented by existing `acceptance/load.py`: p95 end-to-end wall time measured from scheduled due time <=0.5 s, server/transport error fraction <0.01, and all HTTP responses 200. The open-loop generator captures dispatch delay and request duration separately, limits concurrency to 32, uses a unique request ID per call, and has a 5-second request timeout. A nonzero exit code signals failure; reports are uploaded `when: always` if produced.

The separate HPA test must demonstrate independent scale-up and scale-down. Do not infer HPA acceptance from this load test.
