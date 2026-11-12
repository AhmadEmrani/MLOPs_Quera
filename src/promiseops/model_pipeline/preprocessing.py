"""Shared scikit-learn preprocessing for training and serving."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from promiseops.model_pipeline.validation import FEATURE_COLUMNS


NUMERIC_FEATURES = (
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "total_weight_g",
    "payment_total",
    "max_installments",
    "promise_days",
    "approval_hour",
    "approval_weekday",
    "interstate_fraction",
)

CATEGORICAL_FEATURES = (
    "customer_state",
    "primary_seller_state",
    "primary_category",
    "primary_payment_type",
)


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the canonical predictors, excluding IDs and label metadata."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("model features must be provided as a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("feature DataFrame must not contain duplicate columns")

    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"feature DataFrame is missing columns: {', '.join(sorted(missing))}")
    return frame.loc[:, list(FEATURE_COLUMNS)].copy()


def build_preprocessor() -> ColumnTransformer:
    """Create the unfitted preprocessing transformer shared by all candidates."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Attach a classifier to the shared preprocessing in one fitted pipeline.

    Fit the returned pipeline on training rows only. The fitted preprocessing
    and estimator then serialize together for identical serving transformations.
    """
    if not isinstance(estimator, BaseEstimator):
        raise TypeError("estimator must be a scikit-learn estimator")
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("estimator", estimator),
        ]
    )
