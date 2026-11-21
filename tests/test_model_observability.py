"""Serving instrumentation regressions, including lazy child initialization."""
from types import SimpleNamespace
import math

import pandas as pd
import pytest
from prometheus_client.parser import text_string_to_metric_families

from promiseops.api.metrics import Metrics
from promiseops.api import services
from promiseops.data import FEATURE_COLUMNS

VERSION = "sha256:" + "a" * 64


def sample(metrics, name, labels=None):
    return metrics.registry.get_sample_value(name, labels or {})


def test_all_contract_features_exist_at_zero_before_any_prediction():
    metrics = Metrics()
    children = [
        s for family in text_string_to_metric_families(metrics.render().decode())
        for s in family.samples if s.name == "promiseops_feature_missing_total"
    ]
    assert {s.labels["feature"] for s in children} == set(FEATURE_COLUMNS)
    assert all(s.value == 0 for s in children)
    assert sample(metrics, "promiseops_feature_rows_total") == 0
    assert sample(metrics, "promiseops_model_inferences_total") == 0
    assert sample(metrics, "promiseops_model_loaded") == 0
    assert math.isnan(sample(metrics, "promiseops_validation_brier"))


def test_release_reference_metrics_are_from_bundle_and_reset_when_unavailable():
    metrics = Metrics()
    metrics.set_release({"model_version": VERSION, "metrics": {
        "recall_at_10pct": 0.45, "precision_at_10pct": 0.32,
        "average_precision": 0.38, "brier": 0.12,
        "n": 1000, "prevalence": 0.08, "late_count": 80, "k": 100,
        "private_order_id": 1,
    }})
    assert sample(metrics, "promiseops_model_info", {"model_version": VERSION}) == 1
    assert sample(metrics, "promiseops_model_loaded") == 1
    assert sample(metrics, "promiseops_validation_reference_available") == 1
    assert sample(metrics, "promiseops_validation_recall_at_10pct") == 0.45
    assert sample(metrics, "promiseops_validation_n") == 1000
    assert "private_order_id" not in metrics.render().decode()
    metrics.set_release(None)
    assert sample(metrics, "promiseops_model_info", {"model_version": VERSION}) is None
    assert sample(metrics, "promiseops_model_loaded") == 0
    assert sample(metrics, "promiseops_validation_reference_available") == 0
    assert math.isnan(sample(metrics, "promiseops_validation_brier"))


def test_only_valid_artifact_version_is_exposed_and_missing_reference_is_not_zero():
    metrics = Metrics()
    metrics.set_release({"model_version": "secret-untrusted-version", "metrics": {"brier": -1}})
    assert "secret-untrusted-version" not in metrics.render().decode()
    assert sample(metrics, "promiseops_validation_reference_available") == 0
    assert math.isnan(sample(metrics, "promiseops_validation_brier"))


def test_arbitrary_feature_labels_are_rejected():
    metrics = Metrics()
    with pytest.raises(ValueError, match="fixed contract"):
        metrics.observe_features({"secret_order_id": 1}, 1)
    assert "secret_order_id" not in metrics.render().decode()
    assert sample(metrics, "promiseops_feature_rows_total") == 0


class FakeModel:
    classes_ = (0, 1)

    def predict_proba(self, frame):
        return [[0.25, 0.75]] * len(frame)


def test_prediction_metrics_are_fresh_computations_not_cached_replies(monkeypatch):
    metrics = Metrics()
    row = {name: 1 for name in FEATURE_COLUMNS}
    row["primary_category"] = None
    monkeypatch.setattr(services, "build_features", lambda *a, **k: pd.DataFrame([row]))
    store = {}

    def save(_conn, *, request_id, order_id, compute_fn):
        if request_id not in store:
            store[request_id] = {"order_id": order_id, "request_id": request_id, **compute_fn()}
        return store[request_id]

    monkeypatch.setattr(services, "get_or_save_prediction", save)
    state = SimpleNamespace(
        model=FakeModel(), release={"model_version": VERSION}, metrics=metrics,
        open_connection=lambda: SimpleNamespace(close=lambda: None),
    )
    first = services.predict_order(state, order_id="private-order-1", request_id="private-request-1")
    second = services.predict_order(state, order_id="private-order-1", request_id="private-request-1")
    assert first == second and first["risk_probability"] == 0.75
    assert sample(metrics, "promiseops_model_inferences_total") == 1
    assert sample(metrics, "promiseops_feature_rows_total") == 1
    assert sample(metrics, "promiseops_feature_missing_total", {"feature": "primary_category"}) == 1
    assert sample(metrics, "promiseops_feature_missing_total", {"feature": "item_count"}) == 0
    assert sample(metrics, "promiseops_prediction_risk_count") == 1
    assert sample(metrics, "promiseops_prediction_risk_sum") == 0.75
    assert sample(metrics, "promiseops_model_inference_seconds_count") == 1
    text = metrics.render().decode()
    assert "private-order-1" not in text and "private-request-1" not in text


def test_failed_computation_does_not_observe_a_successful_risk(monkeypatch):
    metrics = Metrics()
    row = {name: 1 for name in FEATURE_COLUMNS}
    monkeypatch.setattr(services, "build_features", lambda *a, **k: pd.DataFrame([row]))

    class BrokenModel(FakeModel):
        def predict_proba(self, frame):
            return [[0.5, float("nan")]]

    state = SimpleNamespace(model=BrokenModel(), release={"model_version": VERSION}, metrics=metrics)
    with pytest.raises(ValueError, match="non-finite"):
        services.score(state, None, "private-order")
    assert sample(metrics, "promiseops_model_inference_errors_total") == 1
    assert sample(metrics, "promiseops_model_inferences_total") == 0
    assert sample(metrics, "promiseops_prediction_risk_count") == 0
    assert sample(metrics, "promiseops_model_inference_seconds_count") == 1
    assert sample(metrics, "promiseops_feature_rows_total") == 1


def test_registries_are_independent_between_process_instances():
    a, b = Metrics(), Metrics()
    a.inferences.inc()
    assert sample(a, "promiseops_model_inferences_total") == 1
    assert sample(b, "promiseops_model_inferences_total") == 0


def test_startup_publishes_loaded_release_and_failed_startup_clears_it(monkeypatch):
    from pathlib import Path
    from promiseops.api import state as api_state

    settings = SimpleNamespace(release_path=Path('/opt/promiseops/release/release.json'))
    loaded = {"model_version": VERSION, "metrics": {"brier": 0.17}}
    monkeypatch.setattr(api_state, "init_db", lambda conn: None)
    state = api_state.AppState(
        settings, bundle_loader=lambda path: (FakeModel(), loaded),
        connection_factory=lambda: SimpleNamespace(close=lambda: None),
    )
    state.startup()
    assert sample(state.metrics, "promiseops_model_loaded") == 1
    assert sample(state.metrics, "promiseops_model_info", {"model_version": VERSION}) == 1
    assert sample(state.metrics, "promiseops_validation_brier") == 0.17

    def broken_loader(path):
        raise ValueError("invalid bundle")

    state.bundle_loader = broken_loader
    state.startup()
    assert state.model is None
    assert sample(state.metrics, "promiseops_model_loaded") == 0
    assert sample(state.metrics, "promiseops_model_info", {"model_version": VERSION}) is None
    assert math.isnan(sample(state.metrics, "promiseops_validation_brier"))
