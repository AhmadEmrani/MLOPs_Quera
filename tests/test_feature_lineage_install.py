"""Patch extraction is safe: strict anchors, syntax checks, and idempotency."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "apply_feature_lineage.py"
spec = spec_from_file_location("lineage_installer", SCRIPT)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_unknown_source_fails_closed():
    with pytest.raises(RuntimeError, match="Unsupported source layout"):
        module.patch_data("def unrelated(): pass")
    with pytest.raises(RuntimeError, match="Unsupported source layout"):
        module.patch_service("def unrelated(): pass")
    with pytest.raises(RuntimeError, match="Unsupported source layout"):
        module.patch_training("def unrelated(): pass")


def test_schema_patch_has_unique_key_and_transaction_capture():
    source = (
        'SCHEMA_SQL = """\n'
        'CREATE TABLE IF NOT EXISTS predictions (\n'
        '    request_id text,\n'
        '    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n'
        ');\n"""\n'
    )
    source += '''def sample(compute_fn, request_id, order_id, connection):
    raw_conn = _get_raw_conn(connection)
    init_db(raw_conn)

    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT order_id, risk_probability, model_version FROM predictions WHERE request_id = %s",
            (request_id,),
        )
    result = compute_fn()
    risk_prob = float(result["risk_probability"])
    model_ver = str(result["model_version"])
    try:
        with raw_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (request_id, order_id, risk_probability, model_version)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (request_id) DO NOTHING
                """,
                (request_id, order_id, risk_prob, model_ver),
            )
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
'''
    patched = module.patch_data(source)
    assert patched == module.patch_data(patched)
    assert 'UNIQUE (order_id, model_version, feature_contract_sha256, feature_hash)' in patched
    assert 'inserted = cur.rowcount == 1' in patched
    assert 'ON CONFLICT (order_id, model_version,' in patched
    compile(patched, '<synthetic>', 'exec')


def test_safe_multi_file_application_refuses_partial_updates(tmp_path):
    for relative in (module.DB, module.SERVICE, module.TRAIN):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('invalid upstream layout', encoding='utf-8')
    before = [(tmp_path / p).read_text() for p in (module.DB, module.SERVICE, module.TRAIN)]
    with pytest.raises(RuntimeError):
        module.apply(tmp_path)
    assert before == [(tmp_path / p).read_text() for p in (module.DB, module.SERVICE, module.TRAIN)]


def test_service_patch_preserves_scoring_contract():
    src = '''from datetime import datetime, UTC
from promiseops.data import FEATURE_COLUMNS, build_features, get_or_save_prediction

def score(conn, order_id, release, features):
    frame = build_features(conn, [order_id], as_of=datetime.now(UTC))
    return {
        "risk_probability": 0.1,
        "model_version": str(release["model_version"]),
    }
'''
    changed = module.patch_service(src)
    assert changed == module.patch_service(changed)
    assert "feature_as_of = datetime.now(UTC)" in changed
    assert '"_feature_observation": observation(' in changed
    compile(changed, "<synthetic_service>", "exec")


def test_training_patch_writes_reference_and_logs_mlflow():
    src = '''from promiseops.data import build_features, training_snapshot, training_source_lineage

def train_candidate():
    train_stats = _cohort_stats(split.train_labels, split.train_order_ids)
    validation_stats = _cohort_stats(split.validation_labels, split.validation_order_ids)
    manifest_data = {
        "feature_contract_sha256": feature_contract_sha256,
        "source_commit": commit,
    }
    manifest = save_bundle(model, out, {
                "feature_contract_sha256": feature_contract_sha256,
                "dataset_manifest_file": "dataset_manifest.json",
    })
    version_tags = {
            "feature_contract_sha256": feature_contract_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    log_run(tags={
                "feature_contract_sha256": feature_contract_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
    }, artifacts={
                f"candidates/{name}/feature_contract.json": bundle_dir / "feature_contract.json",
    })
'''
    patched = module.patch_training(src)
    assert patched == module.patch_training(patched)
    assert patched.count('"feature_reference_sha256": reference_sha256') == 4
    assert '"monitoring/feature_reference.json": reference_path' in patched
    assert 'feature_aggregation_as_of=evaluation_time' in patched
    compile(patched, "<synthetic_training>", "exec")


def test_explicit_migration_matches_deployed_schema():
    migration = (Path(__file__).resolve().parents[1] / "db/migrations" /
                 "002_prediction_feature_observations.sql").read_text(encoding="utf-8")
    assert "prediction_feature_observations" in migration
    assert "UNIQUE (order_id, model_version, feature_contract_sha256, feature_hash)" in migration
    assert "first_request_id VARCHAR(128) NOT NULL REFERENCES predictions(request_id)" in migration
    assert "CREATE INDEX IF NOT EXISTS idx_feature_observations_window" in migration


def test_api_fixture_compatibility_patch():
    src = '''    def fake_features(_conn, order_ids, *, as_of, labeled=False):
        row = {name: 1 for name in FEATURE_COLUMNS}
        row["primary_category"] = None
        result = compute_fn()
        store[request_id] = {"order_id": order_id, "request_id": request_id, **result}
'''
    updated = module.patch_api_test(src)
    assert updated == module.patch_api_test(updated)
    assert 'ensure_schema=True' in updated
    assert 'result.pop("_feature_observation", None)' in updated


def test_observability_fixture_compatibility_patch():
    src = '''    row = {name: 1 for name in FEATURE_COLUMNS}
    row["primary_category"] = None
    if request_id not in store:
            store[request_id] = {"order_id": order_id, "request_id": request_id, **compute_fn()}
'''
    updated = module.patch_observability_test(src)
    assert updated == module.patch_observability_test(updated)
    assert 'result.pop("_feature_observation", None)' in updated
    assert 'row["customer_state"] = "SP"' in updated
