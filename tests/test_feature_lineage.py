from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promiseops.feature_lineage import (
    CATEGORICAL,
    NUMERIC,
    canonical_features,
    canonical_json,
    digest_json,
    observation,
    training_reference,
)
from promiseops.model_pipeline.validation import FEATURE_COLUMNS


SHA = "a" * 64
COHORT = "b" * 64
NOW = datetime(2018, 1, 31, 23, tzinfo=UTC)


def feature_record(**kwargs):
    row = {
        "item_count": 2, "seller_count": 1, "total_price": 120.0,
        "total_freight": 20.0, "total_weight_g": 350,
        "payment_total": 140.0, "max_installments": 1,
        "promise_days": 8, "approval_hour": 17, "approval_weekday": 2,
        "interstate_fraction": 0.5, "customer_state": "SP",
        "primary_seller_state": "RJ", "primary_category": "tools",
        "primary_payment_type": "credit_card",
    }
    row.update(kwargs)
    return row


def sample_frame(n=5):
    return pd.DataFrame([feature_record(total_price=100 + i, item_count=i + 1) for i in range(n)])


def reference(frame=None, **overrides):
    opts = {
        "feature_contract_sha256": SHA,
        "train_end": NOW,
        "feature_aggregation_as_of": NOW + timedelta(days=90),
        "train_cohort_hash": COHORT,
        "dataset_identity": "sha256:" + "c" * 64,
    }
    opts.update(overrides)
    return training_reference(sample_frame() if frame is None else frame, **opts)


def test_snapshot_has_exact_15_features_no_label_or_identifiers():
    row = feature_record()
    row.update(order_id="secret_order", late=1, delivered_at="2018-02-10")
    snap = observation(pd.DataFrame([row]), feature_as_of=NOW, release={"feature_contract_sha256": SHA})
    assert list(snap["features"]) == list(FEATURE_COLUMNS)
    assert set(snap["features"]).isdisjoint({"order_id", "late", "delivered_at"})
    assert snap["feature_contract_sha256"] == SHA
    assert snap["contract_identity_source"] == "bundle_feature_contract_sha256"
    assert snap["feature_hash"] == digest_json(snap["features"])
    assert snap["feature_as_of"] == "2018-01-31T23:00:00Z"


def test_snapshot_identical_across_column_order_numeric_scalar_dtypes():
    first = pd.DataFrame([feature_record()])
    second = pd.DataFrame([feature_record()])[list(reversed(FEATURE_COLUMNS))]
    assert canonical_features(first) == canonical_features(second)
    assert observation(first, feature_as_of=NOW, release={})["feature_hash"] == observation(
        second, feature_as_of=NOW, release={}
    )["feature_hash"]


def test_snapshot_change_affects_hash_but_not_category_order():
    original = observation(pd.DataFrame([feature_record()]), feature_as_of=NOW, release={})
    changed = observation(pd.DataFrame([feature_record(total_price=500)]), feature_as_of=NOW, release={})
    assert original["feature_hash"] != changed["feature_hash"]
    assert original["contract_identity_source"] == "fallback_names_only"
    assert original["feature_contract_sha256"] != SHA


def test_missing_categorical_and_numeric_are_json_null():
    row = feature_record(total_price=float("nan"), customer_state=None)
    values = canonical_features(pd.DataFrame([row]))
    assert values["total_price"] is None
    assert values["customer_state"] is None
    assert '"total_price":null' in canonical_json(values)


@pytest.mark.parametrize("feature,value", [("total_price", float("inf")), ("item_count", True), ("customer_state", 3)])
def test_invalid_scalar_values_are_rejected(feature, value):
    with pytest.raises(ValueError):
        canonical_features(pd.DataFrame([feature_record(**{feature: value})]))


def test_validation_requires_timezone_one_row_and_consistent_contract():
    with pytest.raises(ValueError):
        observation(sample_frame(2), feature_as_of=NOW, release={})
    with pytest.raises(ValueError):
        observation(sample_frame(1), feature_as_of=datetime(2018, 1, 1), release={})
    with pytest.raises(ValueError):
        observation(sample_frame(1), feature_as_of=NOW, release={"feature_contract_sha256": "bad"})


def test_reference_uses_train_only_and_preserves_mass():
    result = reference()
    assert result["n_unique_orders"] == 5
    assert set(result["distributions"]) == set(FEATURE_COLUMNS)
    for name, dist in result["distributions"].items():
        assert dist["missing_count"] + dist["non_missing_count"] == 5
        if name in NUMERIC:
            assert sum(dist["bin_counts"]) == dist["non_missing_count"]
            assert len(dist["bin_counts"]) == len(dist["bin_edges"]) + 1
            assert dist["bin_edges"] == sorted(set(dist["bin_edges"]))
        elif name in CATEGORICAL:
            assert sum(dist["category_counts"].values()) + dist["other_count"] == dist["non_missing_count"]
    assert "late" not in canonical_json(result)
    assert "order_id" not in canonical_json(result)


def test_reference_deterministic_hash_and_order_independent():
    frame = sample_frame()
    frame["order_id"] = [f"o{i}" for i in range(len(frame))]
    frame["late"] = [i % 2 for i in range(len(frame))]
    a = reference(frame)
    b = reference(frame.sample(frac=1, random_state=12))
    assert digest_json(a) == digest_json(b)


def test_reference_missing_all_and_constant_quantile_edges():
    frame = pd.DataFrame([feature_record(total_price=4.0, customer_state=None) for _ in range(9)])
    result = reference(frame)
    assert result["distributions"]["total_price"]["bin_counts"] == [0, 9]
    assert result["distributions"]["customer_state"]["missing_count"] == 9
    assert result["distributions"]["customer_state"]["category_counts"] == {}


def test_reference_categorical_other_limit_and_tie_stability():
    frame = pd.DataFrame([feature_record(primary_category=x) for x in ["c", "b", "a", "c", "a"]])
    result = reference(frame, max_categories=2)
    dist = result["distributions"]["primary_category"]
    assert dist["category_counts"] == {"a": 2, "c": 2}
    assert dist["other_count"] == 1


def test_reference_rejects_invalid_boundaries_and_contract():
    with pytest.raises(ValueError):
        reference(feature_contract_sha256="invalid")
    with pytest.raises(ValueError):
        reference(feature_aggregation_as_of=NOW - timedelta(days=1))
    with pytest.raises(ValueError):
        reference(quantile_bins=1)


def test_canonical_hash_stable():
    assert digest_json({"x": 1, "y": 2}) == digest_json({"y": 2, "x": 1})
