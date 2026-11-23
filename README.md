# PromiseOps stages 3 & 4: Golden + Temporal + Transaction Refactoring

**Target:** `refactor/codesAndLifeCycle` (GitHub/GitLab). This is a source patch
prepared against the checked GitHub branch. It has not been applied to the
remote branch or deployed to Production.

**Installation from repository root:**

```bash
unzip -o ~/Downloads/promiseops-golden-temporal-transactions.zip -d .
python ci/apply_golden_temporal_refactor.py --check
python ci/apply_golden_temporal_refactor.py
python ci/install_golden_ci.py
PYTHONPATH=src python -m pytest -q -rs \
  tests/test_golden_contract.py tests/test_temporal_safety.py \
  tests/test_feature_query_refactor.py tests/test_feature_persistence.py \
  tests/test_transaction_regression.py tests/test_golden_mlflow_report.py \
  tests/test_golden_model_predictions.py
PYTHONPATH=src python -m pytest -q -rs
git diff --check
git status --short
```

**Important:** This ZIP includes an installer for the 3 edited existing files.
Files `src/promiseops/data.py`, `src/promiseops/model.py`, and
`tests/test_feature_persistence.py` are not overwritten directly. If source
anchors differ, the installer fails closed, rather than replacing your work.

Manual migration (separate approved DB action) is in
`db/migrations/003_outcome_batch_available_at.sql`. Do **NOT** apply it to
production until your planned final integration stage. See
`docs/GOLDEN_TEMPORAL_TRANSACTION_REFACTOR.md` for design, limitations and
end-to-end test procedure.
