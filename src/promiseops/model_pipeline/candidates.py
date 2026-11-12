"""Candidate classifiers used by the modular training pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from promiseops.model_pipeline.preprocessing import build_pipeline, feature_matrix


def build_candidates(
    *,
    random_state: int = 42,
    n_estimators: int = 300,
) -> dict[str, BaseEstimator]:
    """Build the prevalence baseline, Logistic Regression, and Random Forest.

    The two learned classifiers share the same preprocessing pipeline. Fixed
    settings and an explicit random seed keep candidate runs reproducible.
    """
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")

    return {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": build_pipeline(
            LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight=None,
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
                class_weight=None,
            )
        ),
    }


def fit_candidates(
    candidates: Mapping[str, BaseEstimator],
    train_features: pd.DataFrame,
    train_labels: Iterable[int],
) -> dict[str, BaseEstimator]:
    """Fit every candidate on the same training rows and binary labels."""
    features = feature_matrix(train_features)
    raw_labels = list(train_labels)
    if not raw_labels or len(features) != len(raw_labels):
        raise ValueError("training features and labels must have the same non-zero length")
    if any(label not in (0, 1, False, True) for label in raw_labels):
        raise ValueError("training labels must be binary values 0 or 1")
    labels = [int(label) for label in raw_labels]
    if set(labels) != {0, 1}:
        raise ValueError("training labels must contain both classes")
    if not candidates:
        raise ValueError("at least one candidate is required")

    fitted: dict[str, BaseEstimator] = {}
    for name, candidate in candidates.items():
        if not name:
            raise ValueError("candidate names must be non-empty")
        fitted[name] = candidate.fit(features, labels)
    return fitted


def positive_class_probability(estimator: BaseEstimator, features: pd.DataFrame) -> np.ndarray:
    """Return validated probabilities for class ``1`` (late delivery)."""
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("candidate estimator must implement predict_proba")

    matrix = feature_matrix(features)
    classes = np.asarray(estimator.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("fitted estimator must contain exactly one positive class labeled 1")

    probability_matrix = np.asarray(estimator.predict_proba(matrix), dtype=float)
    if probability_matrix.ndim != 2 or probability_matrix.shape[0] != len(matrix):
        raise ValueError("predict_proba returned an invalid probability matrix")
    if probability_matrix.shape[1] != len(classes):
        raise ValueError("predict_proba columns do not match estimator classes")

    probabilities = probability_matrix[:, positive_indices[0]]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("candidate returned probabilities outside the finite [0, 1] range")
    return probabilities


def score_candidates(
    candidates: Mapping[str, BaseEstimator],
    features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Score one shared cohort with every fitted candidate."""
    if not candidates:
        raise ValueError("at least one fitted candidate is required")
    matrix = feature_matrix(features)
    return {
        name: positive_class_probability(candidate, matrix)
        for name, candidate in candidates.items()
    }
