"""Optional model-specific golden output comparison against reviewed bundle.

Neither a missing bundle nor missing signed-off output makes a fake PASS:
pytest marks this check SKIPPED, and the MLflow evidence exposes that.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promiseops.model_pipeline.bundle import load_bundle
from promiseops.model_pipeline.candidates import positive_class_probability
from promiseops.model_pipeline.preprocessing import feature_matrix

ROOT=Path(__file__).resolve().parents[1]


def test_version_pinned_golden_model_outputs():
    bundle=os.environ.get("PROMISEOPS_GOLDEN_BUNDLE_DIR")
    expected=os.environ.get("PROMISEOPS_GOLDEN_MODEL_CASES_FILE")
    if not bundle or not expected:
        pytest.skip("Set both PROMISEOPS_GOLDEN_BUNDLE_DIR and PROMISEOPS_GOLDEN_MODEL_CASES_FILE")
    model,release=load_bundle(Path(bundle))
    fixture=ROOT/"fixtures/golden_features_v1.json"
    locked=json.loads(Path(expected).read_text(encoding="utf-8"))
    assert locked["model_version"]==release["model_version"]
    assert locked["model_artifact_sha256"]==release["artifact_sha256"]
    assert locked["feature_fixture_sha256"]==hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert locked["schema_version"]==1
    golden=json.loads(fixture.read_text(encoding="utf-8"))
    assert [r["case_id"] for r in golden["cases"]] == [r["case_id"] for r in locked["cases"]]
    features=feature_matrix(pd.DataFrame([c["features"] for c in golden["cases"]]))
    observed=positive_class_probability(model,features)
    expected_scores=[c["risk_probability"] for c in locked["cases"]]
    np.testing.assert_allclose(observed,expected_scores,rtol=0,
                               atol=float(locked["probability_tolerance"]))
    reversed_columns=features.loc[:,list(reversed(features.columns))]
    np.testing.assert_allclose(observed,positive_class_probability(model,reversed_columns),rtol=0,atol=1e-12)
