"""T1-T6 offline regressions: no database, MLflow, deployment or training used."""
from __future__ import annotations
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from promiseops.feature_lineage import training_reference, canonical_json
from promiseops.monitoring.contracts import Prediction, monitoring_windows, unique_predictions
from promiseops.monitoring.decisions import decide
from promiseops.monitoring.drift import drift_report, psi_bins, tvd_bins, reference_contract
from promiseops.monitoring.labels import evaluate_mature
from promiseops.monitoring.reference import validate_reference
from promiseops.monitoring.selection import select_for_review
from promiseops.monitoring.service import build_report
from promiseops.monitoring.repository import READ_MONITORING, fetch_predictions
from promiseops.model_pipeline.validation import FEATURE_COLUMNS

NOW = datetime(2018, 7, 1, tzinfo=UTC)
SHA = "a" * 64
VERSION = "sha256:" + "b" * 64


def features(i=0, shifted=False):
    row = {
        "item_count": (i % 5) + (200 if shifted else 1),
        "seller_count": 1,
        "total_price": float(100 + i % 10),
        "total_freight": float(10 + i % 3),
        "total_weight_g": 350.0,
        "payment_total": float(110 + i % 10),
        "max_installments": 1,
        "promise_days": 8,
        "approval_hour": 12,
        "approval_weekday": i % 7,
        "interstate_fraction": 0.25,
        "customer_state": "SP",
        "primary_seller_state": "RJ",
        "primary_category": "tools",
        "primary_payment_type": "credit_card",
    }
    assert set(row) == set(FEATURE_COLUMNS)
    return row


def reference():
    frame = pd.DataFrame([features(i) for i in range(200)])
    return training_reference(frame, feature_contract_sha256=SHA,
        train_end=NOW - timedelta(days=100),
        feature_aggregation_as_of=NOW - timedelta(days=100),
        train_cohort_hash="c" * 64,
        dataset_identity="sha256:" + "d" * 64)


def rows(shifted=False, count=110, label=None):
    result = []
    for i in range(count):
        for days in [10, 3]:
            result.append(Prediction(order_id=f"o-{days}-{i}", model_version=VERSION,
                  risk_probability=0.8 if i % 4 == 0 else 0.2,
                  scored_as_of=NOW - timedelta(days=days),
                  feature_contract_sha256=SHA, features=features(i, shifted),
                  label=label, batch_id=f"batch-{days}",
                  label_available_at=NOW - timedelta(days=1) if label is not None else None))
    return result


def test_t1_unique_order_dedup_and_company_time():
    r = rows(count=1)
    first = r[0]
    again = Prediction(first.order_id, first.model_version, first.risk_probability,
              first.scored_as_of + timedelta(minutes=1), SHA, first.features,
              batch_id="retry")
    assert len(unique_predictions(r + [again], NOW)) == 2
    assert monitoring_windows(NOW)[0][1] == monitoring_windows(NOW)[1][0]
    assert first in unique_predictions(r + [again], NOW)


def test_t2_reference_validation_no_missing_features():
    original = reference()
    contract, distributions = reference_contract(original)
    assert contract == SHA and set(distributions) == set(FEATURE_COLUMNS)
    bad = dict(original)
    bad["feature_names"] = ["secret_column"]
    with pytest.raises(ValueError, match="contract order"):
        reference_contract(bad)


def test_t2_reference_checksum_rejects_mutation():
    raw = (canonical_json(reference()) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    assert validate_reference(raw, expected_sha256=digest)["train_cohort_hash"] == "c" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_reference(raw + b" ", expected_sha256=digest)


def test_t3_no_drift_same_distribution():
    report = drift_report(rows(), reference(), as_of=NOW)
    assert report["sufficient_windows"]
    assert not report["persistent_drift"]


def test_t3_persistent_numeric_drift_two_nonoverlapping_windows():
    report = drift_report(rows(shifted=True), reference(), as_of=NOW)
    assert report["persistent_drift"]
    assert "item_count" in report["persistent_features"]
    assert report["windows"][0]["end_exclusive"] == report["windows"][1]["start"]


def test_t3_one_window_not_persistent():
    cohort = rows()
    for p in cohort:
        if p.scored_as_of == NOW - timedelta(days=3):
            p.features["item_count"] += 500
    report = drift_report(cohort, reference(), as_of=NOW)
    assert not report["persistent_drift"]


def test_t3_insufficient_window_suppresses_drift():
    report = drift_report(rows(shifted=True, count=99), reference(), as_of=NOW)
    assert not report["persistent_drift"]
    assert not report["sufficient_windows"]


def test_t3_numeric_and_categorical_metrics():
    assert psi_bins([50, 50], [50, 50]) == pytest.approx(0)
    assert psi_bins([50, 50], [100, 0]) > 0.2
    assert tvd_bins({"a": 5, "b": 5}, {"a": 5, "b": 5}) == pytest.approx(0)
    assert tvd_bins({"a": 5, "b": 5}, {"a": 10, "b": 0}) == pytest.approx(0.5)


def test_t3_incompatible_feature_contract_is_rejected():
    p = rows(count=1)[0]
    bad = Prediction(p.order_id, VERSION, p.risk_probability, p.scored_as_of,
                     "f"*64, p.features)
    with pytest.raises(ValueError, match="feature contract"):
        drift_report([bad], reference(), as_of=NOW)


def test_t4_delayed_labels_need_100_and_two_classes():
    report = evaluate_mature(rows(count=49, label=0), as_of=NOW)
    assert not report["sufficient"] and report["reason"] == "insufficient_mature_labels"
    report = evaluate_mature(rows(count=110, label=0), as_of=NOW)
    assert report["reason"] == "single_class_labels"


def test_t4_mature_metrics_match_contract():
    cohort = rows(count=110)
    with_labels = [Prediction(p.order_id, VERSION, p.risk_probability,
        p.scored_as_of, SHA, p.features, label=int(i % 3 == 0), batch_id=p.batch_id,
        label_available_at=NOW - timedelta(days=1))
        for i, p in enumerate(cohort)]
    report = evaluate_mature(with_labels, as_of=NOW)
    assert report["sufficient"]
    assert report["metrics"]["n"] == 220
    assert 0 <= report["metrics"]["brier"] <= 1
    assert report["metrics"]["k"] == 22


def test_t4_company_time_retention_not_wall_clock():
    old = Prediction("old", VERSION, 0.4, NOW - timedelta(days=61), SHA, features(), label=1,
                     label_available_at=NOW - timedelta(days=50))
    assert evaluate_mature([old], as_of=NOW)["n_unique_predictions"] == 0


def test_t4_query_requires_both_source_and_receipt_cutoffs():
    assert "out.available_at <= %(as_of)s" in READ_MONITORING
    assert "COALESCE(out.batch_available_at" in READ_MONITORING
    assert "ord.batch_available_at <= p.scored_as_of" in READ_MONITORING
    assert "(out.delivered_at AT TIME ZONE 'America/Sao_Paulo')" in READ_MONITORING
    assert "DISTINCT ON (p.order_id)" in READ_MONITORING
    assert "AS label_released_at" in READ_MONITORING


def test_t5_insufficient_labels_never_authorize_retraining():
    result = decide({"persistent_drift": True, "sufficient_windows": True},
                    {"sufficient": False, "reason": "insufficient_mature_labels"}, None)
    assert not result["retraining_recommended"]
    assert result["manual_review"]
    assert not result["automatic_training_permitted"]


def test_t5_degradation_recommends_not_promotes():
    metrics = {"recall_at_10pct": .05, "average_precision": .2, "brier": .24}
    ref = {"recall_at_10pct": .5, "average_precision": .4, "brier": .1}
    result = decide({"persistent_drift": False}, {"sufficient": True, "metrics": metrics}, ref)
    assert result["retraining_recommended"]
    assert "brier_degraded_vs_validation" in result["reasons"]
    assert not result["automatic_training_permitted"]


def test_t6_active_learning_private_deterministic_and_unlabeled_only():
    cohort = rows(count=8)
    selected = select_for_review(cohort, as_of=NOW, limit=3, salt="private-test-key")
    again = select_for_review(list(reversed(cohort)), as_of=NOW, limit=3, salt="private-test-key")
    assert selected == again
    assert selected["selected_count"] == 3
    assert "o-" not in json.dumps(selected)
    assert all(len(x["order_hash"]) == 64 for x in selected["review"])
    with pytest.raises(ValueError, match="salt"):
        select_for_review(cohort, as_of=NOW, limit=3)


def test_t6_already_labeled_samples_are_not_requested():
    result = select_for_review(rows(count=2, label=0), as_of=NOW, salt="private-test-key")
    assert result["selected_count"] == 0


def test_t1_t6_end_to_end_report_is_read_only():
    r = build_report(rows(shifted=True), None, reference(), as_of=NOW,
                     review_salt="private-test-key")
    assert r["schema_version"] == 1
    assert r["drift"]["persistent_drift"]
    assert r["decision"]["status"] == "insufficient_labels"
    assert not r["decision"]["retraining_recommended"]
    assert r["active_learning"]["selected_count"] == 20
    assert "o-10-" not in json.dumps(r)


def test_t1_installer_is_idempotent_without_network():
    from ci.apply_active_learning_phase1 import generate
    edited = generate()
    assert len(edited) == 4
    # In the installed tree the patcher yields identical file content.
    assert all(path.read_text(encoding="utf-8") == content for path, content in edited.items())


def test_t4_future_labels_never_enter_replay():
    p = rows(count=1)[0]
    with pytest.raises(ValueError, match="timestamp"):
        Prediction(p.order_id, VERSION, p.risk_probability, p.scored_as_of,
                   SHA, p.features, label=1)
    report = build_report([p], {p.order_id: (1, NOW + timedelta(days=1))},
                          reference(), as_of=NOW, min_window_orders=1, min_mature_labels=1)
    assert report["delayed_labels"]["n_mature"] == 0
    with pytest.raises(ValueError, match="future or unverified"):
        future = Prediction(p.order_id, VERSION, p.risk_probability, p.scored_as_of,
                            SHA, p.features, label=1, label_available_at=NOW + timedelta(days=1))
        build_report([future], None, reference(), as_of=NOW)


def test_t1_repository_is_read_only_and_parameterized():
    assert "%(model_version)s" in READ_MONITORING
    assert "%(as_of)s" in READ_MONITORING
    assert all(word not in READ_MONITORING.upper() for word in
               ("INSERT INTO", "UPDATE ", "DELETE FROM", "CREATE TABLE"))
    records = [("id-1", VERSION, .25, NOW-timedelta(days=2), SHA,
                features(), 1, "batch-a", NOW-timedelta(days=1))]
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql, args):
            assert sql == READ_MONITORING
            assert args["model_version"] == VERSION
            assert args["row_limit"] == 11
            assert args["earliest"] == NOW - timedelta(days=60)
        def fetchall(self): return records
    class Connection:
        def cursor(self): return Cursor()
    from promiseops.monitoring.repository import fetch_predictions
    rows_result = fetch_predictions(Connection(), model_version=VERSION, as_of=NOW, max_rows=10)
    assert rows_result[0].label == 1
    assert rows_result[0].label_available_at == NOW-timedelta(days=1)


def test_t3_constant_training_feature_detects_out_of_range():
    from promiseops.monitoring.drift import feature_drift
    ref = {"type": "numeric", "bin_edges": [], "bin_counts": [100],
           "non_missing_count": 100, "missing_count": 0, "min": 1.0, "max": 1.0}
    assert feature_drift(ref, [1.0] * 100)["value"] == pytest.approx(0)
    assert feature_drift(ref, [200.0] * 100)["value"] >= .2
    assert feature_drift(ref, [None] * 100)["value"] >= .2


def test_t2_mlflow_reference_downloader_preserves_exact_bytes(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from ci import download_monitoring_reference as cmd
    ref_raw = (canonical_json(reference()) + "\n").encode()
    ref_sha = hashlib.sha256(ref_raw).hexdigest()
    fake_run = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED"),
        data=SimpleNamespace(tags={"model_version": VERSION, "feature_reference_sha256": ref_sha},
                             metrics={"recall_at_10pct": .45, "average_precision": .34, "brier": .15}),
    )
    class Client:
        def get_run(self, run_id):
            assert run_id == "abcdefghi123"
            return fake_run
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(set_tracking_uri=lambda uri: None))
    monkeypatch.setitem(sys.modules, "mlflow.tracking", SimpleNamespace(MlflowClient=Client))
    monkeypatch.setattr(cmd, "download_mlflow_reference_bytes", lambda **kw: ref_raw)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid")
    monkeypatch.setattr(sys, "argv", ["cmd", "--run-id", "abcdefghi123", "--model-version", VERSION,
                "--reference-sha256", ref_sha, "--destination", str(tmp_path)])
    assert cmd.main() == 0
    assert (tmp_path / "feature_reference.json").read_bytes() == ref_raw
    assert json.loads((tmp_path / "validation_metrics.json").read_text())["brier"] == .15


def test_t1_staged_integration_writes_full_predictions_not_just_top10():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    scorer = (root / "src/promiseops/cli.py").read_text()
    batch = (root / "src/promiseops/scheduled_batch.py").read_text()
    assert 'monitoring_rows.append({' in scorer
    assert 'monitoring_rows=monitoring_rows' in batch
    assert 'INSERT INTO monitoring_predictions' in batch
    assert batch.index('INSERT INTO monitoring_predictions') < batch.index('INSERT INTO promiseops_scoring_runs')
    if 'INSERT INTO promiseops_scoring_cursor' in batch:
        assert batch.index('INSERT INTO monitoring_predictions') < batch.index('INSERT INTO promiseops_scoring_cursor')
