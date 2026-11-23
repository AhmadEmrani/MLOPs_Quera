"""Frozen, reviewable golden contracts; no database, cluster or MLflow needed."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from promiseops.feature_lineage import canonical_features, digest_json, observation
from promiseops.model_pipeline.candidates import positive_class_probability
from promiseops.model_pipeline.preprocessing import build_pipeline, feature_matrix
from promiseops.model_pipeline.validation import FEATURE_COLUMNS, FeatureRow

ROOT = Path(__file__).resolve().parents[1]
GOLD = json.loads((ROOT / "fixtures/golden_features_v1.json").read_text(encoding="utf-8"))


def test_golden_examples_are_static_and_cover_all_features():
    assert GOLD["schema_version"] == 1
    assert len(GOLD["cases"]) == 2
    assert [case["late"] for case in GOLD["cases"]] == [1, 0]
    for case in GOLD["cases"]:
        assert set(case["features"]) == set(FEATURE_COLUMNS)
        FeatureRow.model_validate({"order_id": case["order_id"], **case["features"]})
        assert digest_json(case["features"]) == case["feature_hash"]
        assert len(case["feature_hash"]) == 64


def test_training_serving_feature_parity_and_order():
    rows = [case["features"] for case in GOLD["cases"]]
    source = pd.DataFrame(rows)
    serving = feature_matrix(source)
    assert list(serving.columns) == list(FEATURE_COLUMNS)
    for i, row in enumerate(rows):
        assert canonical_features(serving.iloc[[i]]) == row
        assert digest_json(canonical_features(serving.iloc[[i]])) == GOLD["cases"][i]["feature_hash"]

    model = build_pipeline(LogisticRegression(solver="lbfgs", random_state=0))
    model.fit(serving, [case["late"] for case in GOLD["cases"]])
    before = positive_class_probability(model, serving)
    # A client may supply columns in any order; the canonical serving path
    # must reproduce exactly the same preprocessor output and prediction.
    reordered = source.loc[:, list(reversed(FEATURE_COLUMNS))]
    after = positive_class_probability(model, reordered)
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
    assert np.isfinite(before).all()


def test_golden_snapshot_excludes_labels_and_identifiers():
    row = dict(GOLD["cases"][0]["features"], late=1, delivered_at="2017-01-11", order_id="private")
    snap = observation(
        pd.DataFrame([row]),
        feature_as_of=datetime.fromisoformat(GOLD["feature_cutoff"].replace("Z", "+00:00")),
        release={},
    )
    assert snap["feature_hash"] == GOLD["cases"][0]["feature_hash"]
    assert set(snap["features"]) == set(FEATURE_COLUMNS)
    assert not set(snap["features"]) & {"late", "order_id", "delivered_at"}


def test_brazil_dst_approval_hour_is_locked():
    # 2017-01-01 was UTC-02 in Sao Paulo; assuming -03 would change the hour.
    local = datetime.fromisoformat("2017-01-01T16:00:00+00:00").astimezone(ZoneInfo("America/Sao_Paulo"))
    assert (local.hour, local.weekday(), local.utcoffset().total_seconds()) == (14, 6, -7200)
    assert GOLD["cases"][0]["features"]["approval_hour"] == local.hour


def test_golden_fixture_source_is_known_and_not_a_validation_or_holdout_slice():
    order_batch = json.loads((ROOT / "fixtures/synthetic-orders.json").read_text())
    outcome_batch = json.loads((ROOT / "fixtures/synthetic-outcomes.json").read_text())
    assert order_batch["available_at"].replace("+00:00", "Z") == GOLD["order_batch_available_at"]
    assert outcome_batch["available_at"].replace("+00:00", "Z") == GOLD["outcome_batch_available_at"]
    assert {case["order_id"] for case in GOLD["cases"]} <= {item["order_id"] for item in order_batch["orders"]}
