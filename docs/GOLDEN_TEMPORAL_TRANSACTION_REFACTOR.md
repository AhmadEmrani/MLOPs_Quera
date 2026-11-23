# Golden Tests + Temporal Safety + Transaction Refactor

Target branch: `refactor/codesAndLifeCycle`. This patch extends the existing
Feature Lineage & Persistence changes and is not a production rollout.

## Design

1. Golden fixture `fixtures/golden_features_v1.json` freezes 15 actual feature
   values, labels and canonical SHA256 for two orders from the project's
   `fixtures/synthetic-orders.json` and `synthetic-outcomes.json`. The two cases
   are deliberately tiny regression fixtures, NOT a statistical holdout and NOT
   an accuracy estimate. Use a separate untouched holdout for final evaluation.
2. Training feature extraction runs twice in the *same REPEATABLE READ snapshot*:
   train features bound by `train_end`, validation features by `validation_end`.
   The separate `label_as_of=as_of` permits validation label maturation without
   using any features released after the validation cutoff. The train labels
   still require `label_available_at <= train_end` in the existing TemporalSplit.
3. The outcome ingestion transaction saves the **envelope-level**
   `batch_available_at`. Effective `label_available_at` is `max(outcome.available_at,
   outcome.batch_available_at)` for new data. Legacy outcomes have NULL batch
   provenance; the maximum observed outcome receipt timestamp provides a
   conservative bound for them. If receipt history is incomplete, historical
   point-in-time claims must NOT be made. No guessed per-order timestamp is
   backfilled. This can reduce eligible historical cohort sizes.
4. Serving's feature SQL no longer joins `outcomes`; it uses only order/item,
   payment and catalog sources. This reduces read work and eliminates an
   unnecessary outcome lookup from every prediction.
5. A successful **new** prediction `INSERT` uses `RETURNING`, avoiding the
   second SELECT. `INSERT` and `prediction_feature_observations` remain in
   the same transaction. If the feature snapshot insert fails, both roll back.
   A racing `ON CONFLICT` loser still SELECTs the committed winner and retains
   request-ID idempotency. The public API response is unchanged.
6. Training's feature reference is labeled `feature_aggregation_as_of=train_end`
   for accuracy. Existing MLflow bundles retain their previous baseline until
   a new reviewed training/release; never rewrite old MLflow run metadata.
7. `golden-contracts` is a normal CI test-stage job and publishes JUnit/JSON
   artifacts. The PostgreSQL integration test is opt-in and requires
   `PROMISEOPS_TEST_DATABASE_URL` pointing to an isolated database with `test`
   in its name, and CREATE SCHEMA permission. It creates and drops its own
   randomly named schema; it never truncates shared tables.
8. `golden-mlflow-evidence` is manual on `main`, uses existing protected
   `MLFLOW_TRACKING_URI` and offline `requirements-tracking.lock`. It logs a
   **separate** experiment run with JUnit/JSON, checksums, counts, commit and
   optional `PROMISEOPS_GOLDEN_MODEL_VERSION`. It NEVER alters champion or
   challenger aliases. A missing DB integration result is clearly tagged.

## Installation (NO PRODUCTION EXECUTION)

```sh
# On refactor/codesAndLifeCycle, with clean working tree:
unzip -o promiseops-golden-temporal-transactions.zip -d .
python ci/apply_golden_temporal_refactor.py --check
python ci/apply_golden_temporal_refactor.py
python ci/install_golden_ci.py
python -m compileall -q src/promiseops ci tests
PYTHONPATH=src python -m pytest -q -rs \
  tests/test_golden_contract.py \
  tests/test_temporal_safety.py \
  tests/test_feature_query_refactor.py \
  tests/test_feature_persistence.py \
  tests/test_transaction_regression.py
PYTHONPATH=src python -m pytest -q -rs

git diff --check
git diff --stat
```

The `golden-contracts` CI job runs `ci/run_golden_contracts.py` and retains
`reports/golden/report.json` and JUnit XML. If no isolated test database is
configured, the DB integration check will be SKIPPED, and the JSON/MLflow tag
explicitly says so. This is not equivalent to verified PostgreSQL behavior.

## DB migration and final end-to-end checklist

Apply `db/migrations/003_outcome_batch_available_at.sql` using approved DB
migration permissions **before deploying code**. The source's idempotent
`SCHEMA_SQL` includes the same `ADD COLUMN IF NOT EXISTS` for new startup paths.
No migration is executed by the patch installer. During the final integration
window verify new ingested outcome batches have non-NULL provenance, that
historical/legacy unverified rows are treated conservatively, and that the
cohorts still have adequate mature samples. Run the isolated PostgreSQL golden
case, existing DB tests, Airflow training with recorded cutoffs, MLflow Golden
Evidence job, staging smoke, load test and later production promotion only
through approved CI. Golden evidence must not be used to replace normal
model quality gates.

## Known limitations

- Without an order-to-receipt mapping for old outcome batches, their exact
  individual label availability cannot be reconstructed retrospectively.
- `approved_at` is the business event time; source batch availability is
  separately enforced. This does not prove that upstream prices/payment entries
  were accessible at the approval instant. The batch boundary is the earliest
  defensible ingestion-time feature availability.
- Neither a two-row golden fixture nor this CI step proves model generalization.
- The patch installer checks exact anchors for the observed branch and stops
  if upstream files diverge; never force-replace modified code.

## Optional, reviewed model-output Golden fixture

If a matching immutable approved bundle is available, freeze and review its
expected probabilities manually (never overwrite an existing frozen file):

```sh
PYTHONPATH=src python ci/freeze_golden_predictions.py \
  --bundle /approved/read-only/model-bundle/ \
  --output fixtures/golden_model_approved_v1.json
# Review the diff and commit the fixture as a separate approval action.
PROMISEOPS_GOLDEN_BUNDLE_DIR=/approved/read-only/model-bundle \
PROMISEOPS_GOLDEN_MODEL_CASES_FILE=fixtures/golden_model_approved_v1.json \
PYTHONPATH=src python -m pytest -q tests/test_golden_model_predictions.py
```

The CI job defaults to SKIPPED for this optional artifact-bound test until
those variables/artifacts are supplied; its skip is visible in JUnit, JSON and
MLflow tags. The two-row fixed test guards numerical parity, not real model
quality.
