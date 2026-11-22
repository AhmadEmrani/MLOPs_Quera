# PromiseOps — Feature Lineage & Persistence (Phase 2)

Adds **atomic, deduplicated PostgreSQL feature observations** for freshly served
predictions, plus **training-only reference distributions in MLflow**.

Built against the reviewed `src/promiseops/data.py`, `api/services.py`,
`model.py`, and existing API/observability test fixtures. Run the complete
project suite after applying, as the live PostgreSQL/MLflow environment has
not been exercised here. The installer refuses
unknown source layouts rather than overwriting entire source files.

## Installation on a feature branch

```bash
cd /path/to/promiseops
unzip -o promiseops-feature-lineage-persistence.zip -d .
python ci/apply_feature_lineage.py --check
python ci/apply_feature_lineage.py
# Optional manual, read-only GitLab inspection job:
python ci/install_feature_lineage_inspect.py

PYTHONPATH=src python -m pytest -q \
  tests/test_feature_lineage.py \
  tests/test_feature_persistence.py \
  tests/test_feature_lineage_install.py \
  tests/test_feature_inspect.py
PYTHONPATH=src python -m pytest -q -rs
git diff --check
git diff --stat
git status --short
```

**Do not run the database migration by copying a SQL command into production
without checking the database role and change procedure.** See
`docs/FEATURE_LINEAGE_PERSISTENCE.md` for requirements, SQL semantics,
regression checks, code-only deployment, and safe inspection.

No deployments, MLflow writes, or database changes happen from unzipping the
package or running the source patcher.
