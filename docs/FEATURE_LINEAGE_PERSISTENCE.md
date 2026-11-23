# PromiseOps — Feature Lineage & Persistence (Refactoring Phase 2)

## Purpose and scope

This patch **does not implement drift detection or retraining**. It creates the
reproducible feature evidence necessary to do so later. All changes remain
compatible with the existing `champion`/`challenger` lifecycle: no alias changes,
no model promotion, and no change to the `/predict` response contract.

## Serving feature snapshots (PostgreSQL)

One new table `prediction_feature_observations` contains:

| Column | Meaning |
|---|---|
| `first_request_id` | FK to the prediction that first persisted the snapshot |
| `order_id` | Order identity; never a Prometheus label |
| `model_version` | Immutable SHA256 model identity, copied from the successful prediction |
| `feature_as_of` | UTC instant passed to `build_features` during this request |
| `feature_contract_sha256` | Bundle feature contract digest, if present |
| `contract_identity_source` | `bundle_feature_contract_sha256` or explicitly marked `fallback_names_only` |
| `feature_hash` | SHA256 of canonical 15-feature JSON values |
| `features_jsonb` | Exact pre-imputation feature values used to score this request (15 fields only) |
| `observed_at` | DB insertion time of the *first* distinct snapshot |

- The snapshot is constructed only after a **fresh** model calculation.
- `get_or_save_prediction` inserts a prediction and the first distinct snapshot
  in the **same transaction**; either both commit or both roll back.
- Retries of the same `request_id` do not recompute or write an observation.
- Fresh request IDs for the **same order, model, contract, and feature hash**
  write a new prediction but *not* a duplicate observation.
- Changed feature values for an order create another snapshot. A drift monitor
  must still deduplicate by order ID *within each window*.
- Feature snapshots never include labels, outcomes, or request IDs in the JSON
  payload. `first_request_id` and `order_id` are metadata in the DB only.
- The API response continues to expose only the public prediction fields.

### Example: unique orders in a nonoverlapping window

Use the following as a pattern for a later drift job, with bound query parameters.
Do not publish the result rows in CI artifacts or logs.

```sql
SELECT DISTINCT ON (order_id)
       order_id, model_version, feature_contract_sha256,
       contract_identity_source, features_jsonb,
       feature_as_of, observed_at
FROM prediction_feature_observations
WHERE model_version = %(model_version)s
  AND observed_at >= %(start)s
  AND observed_at < %(end)s
ORDER BY order_id, observed_at ASC, id ASC;
```

**Warning:** A historical 2018 order rescored in November 2026 is an observation
in **2026 serving time**, not a genuine 2018 online prediction. `feature_as_of`
reflects the request's actual scoring cutoff; it must not be used to fabricate
historical prediction-time evidence. A future replay mode must explicitly
separate order event time, source availability time, scoring time, and label
maturity.

### Database migration

`src/promiseops/data.py:SCHEMA_SQL` contains `CREATE TABLE IF NOT EXISTS` for
fresh installs. AppState's `startup()`/`/readyz` already call `ensure_schema()`.
This patch removes **per-request DDL** from both `get_or_save_prediction`
and the serving call to `build_features(..., ensure_schema=False)` to avoid
replaying `CREATE TABLE/INDEX` on each inference, a performance regression at
10 RPS. `AppState.startup()` and `/readyz` still ensure the schema.

For production, **prefer applying**
`db/migrations/002_prediction_feature_observations.sql` with an approved
migration role **before rolling out the new API**. This avoids concurrent DDL
races when the two production Pods start. It is required if the application
identity does not have CREATE privileges. The DDL is intentionally
repeated in the migration and `SCHEMA_SQL` for this purpose. This patch does not
request new Kubernetes permissions or run automatic privileged migrations.

Existing `predictions` cannot be backfilled truthfully: their exact feature
snapshots were not persisted. Do not guess historical values from today's DB.

## Training feature reference (MLflow)

For every **new** training run, `src/promiseops/model.py` constructs a
`feature_reference.json` from `split.train_rows`, not validation or holdout.

- All 15 features are represented, separately by type.
- Numeric features store missing/nonmissing counts, min/max/mean, deterministic
  quantile bin edges (up to 10 bins) and counts.
- Categorical features store bounded top-20 category counts, other count, and
  missing count; stable tie-break is lexical.
- No raw orders, labels, delivery outcomes, or PII rows are exported.
- The reference identifies train cohort SHA256, feature contract SHA256,
  dataset identity, **train_end** and `feature_aggregation_as_of`.
- The file is published to each candidate MLflow run as
  `monitoring/feature_reference.json`.
- SHA256 of the file appears in MLflow Run tags, Model Version tags,
  `dataset_manifest.json`, and candidate `release.json`.

The reference is deliberately calculated on the **training cohort only**.
Its feature extraction currently inherits the existing `build_features(...,
 as_of=evaluation_time)` behavior; the explicit
`feature_aggregation_as_of` field prevents mislabeling the reference as a
strict approval-time snapshot. A separate point-in-time audit remains in the
next refactoring phase.

**Existing MLflow runs and currently deployed images do not magically gain a
reference artifact.** Generate it with the next approved training job. No
Model Registry alias is changed by this patch.

A future monitor can load the artifact via the immutable Registry Model
Version's `run_id` and verify `feature_reference_sha256` before PSI/TVD.

## Deployment and verification

1. Apply patch on feature branch; run offline unit tests and entire existing
   suite; review `git diff` and GitLab CI Lint.
2. Confirm application migration permissions (or run explicit migration).
3. Deploy a **code-only image** with the *same approved model bundle*; do not
   accidentally revert the model via the default package version.
4. Run an authorized small Pilot (1 RPS, 10s) using distinct request IDs.
5. Execute manual `inspect-feature-lineage` CI job and download aggregate
   `reports/feature-lineage.json` artifact. It must report observations > 0.
6. Validate no change to `/predict` payload, readiness or baseline latency.
7. Re-run 10 RPS/180s Acceptance **only in the permitted test window**;
   changing the request path may affect P95.
8. After the next approved training run, verify the MLflow artifact and tags.

Manual inspection is available by running
`python ci/install_feature_lineage_inspect.py` once before merging. It adds an
optional, manual, read-only `inspect-feature-lineage` job. The tool uses the
same toolbox/wheelhouse and existing `runtime-config` secret as Load Testing;
its output is counts only. It fails if no observations exist in the last 24h,
so run it **after** a new scoring request reaches the deployed code.

## Limitations / next steps

- Drift PSI/TVD, 7-day windows, label maturity and monitoring alerts are **not**
  part of this patch.
- The reference file is in MLflow; it is not duplicated into the model API
  image. The future monitoring worker must have approved MLflow read access.
- `feature_hash` is a value-identity checksum; it does not by itself measure
  which feature drifted. Compare stored JSON and training distributions.
- Window queries must use **unique orders**, ignore repeat rescores, require
  a minimum count, and use nonoverlapping seven-day windows.
- Keep database retention/access policy aligned with the project's monitoring
  period. Raw feature JSON is not appropriate for unprotected CI artifacts.
