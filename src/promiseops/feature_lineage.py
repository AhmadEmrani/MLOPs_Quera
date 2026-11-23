"""Deterministic serving feature snapshots and training-only reference distributions.

Do not put outcomes, labels, request IDs, or customer IDs into a feature snapshot.
This module does not query databases or mutate MLflow; callers own those actions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from promiseops.model_pipeline.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from promiseops.model_pipeline.validation import FEATURE_COLUMNS

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NUMERIC = frozenset(NUMERIC_FEATURES)
CATEGORICAL = frozenset(CATEGORICAL_FEATURES)
assert NUMERIC | CATEGORICAL == set(FEATURE_COLUMNS)
assert not NUMERIC & CATEGORICAL


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _value(value: Any, feature: str) -> float | str | None:
    """Make pandas / NumPy scalars JSON-stable without silently accepting infinities."""
    if value is None or pd.isna(value):
        return None
    if feature in NUMERIC:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Boolean is not a numeric feature: {feature}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric feature: {feature}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Non-finite numeric feature: {feature}")
        return 0.0 if number == 0 else number
    if feature in CATEGORICAL:
        if not isinstance(value, str):
            raise ValueError(f"Non-string categorical feature: {feature}")
        return value
    raise ValueError(f"Unknown feature: {feature}")


def canonical_features(frame: pd.DataFrame) -> dict[str, float | str | None]:
    if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
        raise ValueError("Serving snapshot requires exactly one feature row")
    if frame.columns.duplicated().any() or not set(FEATURE_COLUMNS).issubset(frame.columns):
        raise ValueError("Serving snapshot is missing canonical features or duplicates columns")
    row = frame.iloc[0]
    return {name: _value(row[name], name) for name in FEATURE_COLUMNS}


def observation(
    feature_frame: pd.DataFrame,
    *,
    feature_as_of: datetime,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a private, immutable snapshot to persist alongside a fresh prediction."""
    if feature_as_of.tzinfo is None or feature_as_of.utcoffset() is None:
        raise ValueError("feature_as_of must be timezone-aware")
    features = canonical_features(feature_frame)
    tagged = release.get("feature_contract_sha256")
    if tagged is None:
        # The legacy starter bundle may not carry the contract's file hash.
        # Mark fallback explicitly: this is NOT interchangeable with the real
        # feature_contract.json checksum, even if both are valid SHA256 values.
        contract = digest_json({"feature_names": list(FEATURE_COLUMNS), "type": "fallback_names_only"})
        identity_source = "fallback_names_only"
    elif isinstance(tagged, str) and HEX64.fullmatch(tagged):
        contract = tagged
        identity_source = "bundle_feature_contract_sha256"
    else:
        raise ValueError("release feature_contract_sha256 is invalid")
    return {
        "features": features,
        "feature_hash": digest_json(features),
        "feature_as_of": feature_as_of.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "feature_contract_sha256": contract,
        "contract_identity_source": identity_source,
    }


def training_reference(
    train_rows: pd.DataFrame,
    *,
    feature_contract_sha256: str,
    train_end: datetime,
    feature_aggregation_as_of: datetime,
    train_cohort_hash: str,
    dataset_identity: str,
    max_categories: int = 20,
    quantile_bins: int = 10,
) -> dict[str, Any]:
    """Create a deterministic compact reference without outcomes or individual rows.

    Numeric edges are exclusive upper bounds for the preceding bucket; current
    feature observations can be assigned via bisect_right/searchsorted(right).
    Missing values are counted separately, never imputed here.
    """
    if not isinstance(train_rows, pd.DataFrame) or train_rows.empty:
        raise ValueError("Reference requires a non-empty train cohort")
    if train_rows.columns.duplicated().any() or not set(FEATURE_COLUMNS).issubset(train_rows.columns):
        raise ValueError("Train rows are missing features or have duplicate columns")
    if not isinstance(feature_contract_sha256, str) or not HEX64.fullmatch(feature_contract_sha256):
        raise ValueError("Reference requires an exact contract file SHA256")
    if not isinstance(train_cohort_hash, str) or not HEX64.fullmatch(train_cohort_hash):
        raise ValueError("Reference requires a train cohort SHA256")
    if not 2 <= quantile_bins <= 100 or not 1 <= max_categories <= 100:
        raise ValueError("Invalid reference bin/category limits")
    for t in (train_end, feature_aggregation_as_of):
        if t.tzinfo is None or t.utcoffset() is None:
            raise ValueError("Reference cutoffs must include timezone")
    if feature_aggregation_as_of < train_end:
        raise ValueError("Feature aggregation cutoff precedes train cutoff")

    n = len(train_rows)
    distributions: dict[str, Any] = {}
    for feature in FEATURE_COLUMNS:
        samples = [_value(raw, feature) for raw in train_rows[feature]]
        missing_count = sum(x is None for x in samples)
        present = [x for x in samples if x is not None]
        if feature in NUMERIC:
            arr = np.asarray(present, dtype=np.float64)
            if len(arr):
                # Drop duplicate quantiles (constant or low-cardinality features).
                edges = sorted({float(x) for x in np.quantile(
                    arr, np.arange(1, quantile_bins) / quantile_bins
                )})
                idx = np.searchsorted(np.asarray(edges), arr, side="right")
                counts = np.bincount(idx, minlength=len(edges) + 1).astype(int).tolist()
                minimum, maximum, mean = float(arr.min()), float(arr.max()), float(arr.mean())
            else:
                edges, counts, minimum, maximum, mean = [], [0], None, None, None
            distributions[feature] = {
                "type": "numeric", "count": n, "non_missing_count": len(present),
                "missing_count": missing_count, "bin_edges": edges, "bin_counts": counts,
                "min": minimum, "max": maximum, "mean": mean,
            }
        else:
            frequencies = Counter(str(x) for x in present)
            top = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:max_categories]
            category_counts = {k: int(v) for k, v in top}
            distributions[feature] = {
                "type": "categorical", "count": n,
                "non_missing_count": len(present), "missing_count": missing_count,
                "category_counts": category_counts,
                "other_count": len(present) - sum(category_counts.values()),
            }
    return {
        "schema_version": 1,
        "reference_kind": "training_cohort_features",
        "feature_names": list(FEATURE_COLUMNS),
        "feature_contract_sha256": feature_contract_sha256,
        "train_cohort_hash": train_cohort_hash,
        "dataset_identity": dataset_identity,
        "train_end": train_end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "feature_aggregation_as_of": feature_aggregation_as_of.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "n_unique_orders": n,
        "numeric_bin_semantics": "searchsorted(edges, value, side=right); missing separate",
        "distributions": distributions,
    }
