"""Exact feature-bin comparison to an immutable MLflow training reference."""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter
from math import isfinite, log
from typing import Any
from promiseops.monitoring.contracts import (
    Prediction, DRIFT_THRESHOLD, MIN_WINDOW_ORDERS, iso, monitoring_windows,
    unique_predictions,
)


def psi_bins(expected: list[int], actual: list[int], *, epsilon: float = 1e-6) -> float:
    if len(expected) != len(actual) or not expected or min(expected + actual) < 0:
        raise ValueError("incompatible numeric histograms")
    ne, na = sum(expected), sum(actual)
    if not ne or not na:
        raise ValueError("both numeric cohorts must be nonempty")
    # Additive smoothing only to avoid infinities; same bins for both distributions.
    k = len(expected)
    return sum(((a + epsilon)/(na + k*epsilon) - (e + epsilon)/(ne + k*epsilon)) *
               log(((a + epsilon)/(na + k*epsilon)) / ((e + epsilon)/(ne + k*epsilon)))
               for e, a in zip(expected, actual))


def tvd_bins(expected: dict[str, int], actual: dict[str, int]) -> float:
    if not expected or not actual or min([*expected.values(), *actual.values()]) < 0:
        raise ValueError("invalid categorical histograms")
    ne, na = sum(expected.values()), sum(actual.values())
    if not ne or not na:
        raise ValueError("empty categorical population")
    return 0.5 * sum(abs(expected.get(key, 0)/ne - actual.get(key, 0)/na)
                     for key in expected.keys() | actual.keys())


def reference_contract(reference: dict[str, Any]) -> tuple[str, dict]:
    from promiseops.model_pipeline.validation import FEATURE_COLUMNS
    if reference.get("schema_version") != 1 or reference.get("reference_kind") != "training_cohort_features":
        raise ValueError("unrecognized feature reference")
    contract = reference.get("feature_contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64 or any(c not in "0123456789abcdef" for c in contract):
        raise ValueError("invalid reference contract SHA256")
    distributions = reference.get("distributions")
    if not isinstance(distributions, dict) or set(distributions) != set(FEATURE_COLUMNS):
        raise ValueError("reference has incomplete feature distributions")
    if reference.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError("feature contract order mismatches baseline")
    return contract, distributions


def feature_drift(distribution: dict, samples: list[Any]) -> dict[str, Any]:
    present = [value for value in samples if value is not None]
    missing = len(samples) - len(present)
    if distribution.get("type") == "numeric":
        edges = distribution.get("bin_edges", [])
        expected = distribution.get("bin_counts", [])
        if (not isinstance(edges, list) or not isinstance(expected, list) or
                len(expected) != len(edges) + 1 or sorted(set(edges)) != edges):
            raise ValueError("invalid numeric reference edges")
        expected_nonmissing = distribution.get("non_missing_count", 0)
        expected_missing = distribution.get("missing_count", 0)
        if expected_nonmissing != sum(expected) or expected_missing < 0:
            raise ValueError("invalid numeric reference bin masses")
        if not expected_nonmissing:
            return {"metric": "psi", "value": None, "reason": "reference_has_no_nonmissing_values",
                    "missing_fraction": missing / len(samples)}
        minimum = distribution.get("min")
        maximum = distribution.get("max")
        if minimum is None or maximum is None or minimum > maximum:
            raise ValueError("invalid numeric baseline range")
        counts = [0] * (len(edges) + 3)
        for value in present:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
                raise ValueError("invalid numeric feature")
            if value < minimum:
                counts[0] += 1  # New low-tail population: out of reference support.
            elif value > maximum:
                counts[-1] += 1  # New high-tail population.
            else:
                counts[bisect_right(edges, value) + 1] += 1
        score = psi_bins([0, *expected, 0, expected_missing], [*counts, missing])
        return {"metric": "psi", "value": score, "missing_fraction": missing / len(samples)}
    if distribution.get("type") == "categorical":
        known = distribution.get("category_counts", {})
        if not isinstance(known, dict) or distribution.get("other_count", -1) < 0:
            raise ValueError("invalid categorical reference")
        for value in present:
            if not isinstance(value, str):
                raise ValueError("invalid categorical feature")
        seen = Counter(present)
        # Use integer-bin keys rather than reserved category names. Real source
        # values may literally equal '__other__' or '__missing__'.
        categories = sorted(known)
        expected = {f"bin{i}": int(known[name]) for i, name in enumerate(categories)}
        actual = {f"bin{i}": int(seen.get(name, 0)) for i, name in enumerate(categories)}
        expected["other"] = int(distribution["other_count"])
        actual["other"] = sum(v for k, v in seen.items() if k not in known)
        expected["missing"] = int(distribution["missing_count"])
        actual["missing"] = missing
        score = tvd_bins(expected, actual)
        return {"metric": "tvd", "value": score, "missing_fraction": missing / len(samples)}
    raise ValueError("unknown feature distribution type")


def drift_report(rows: list[Prediction], reference: dict, *, as_of: object,
                 min_orders: int = MIN_WINDOW_ORDERS) -> dict:
    if min_orders < 1:
        raise ValueError("min_orders must be positive")
    contract, distributions = reference_contract(reference)
    bounds = monitoring_windows(as_of)
    by_window: list[dict] = []
    for start, end in bounds:
        cohort = unique_predictions([p for p in rows if start <= p.scored_as_of < end], end)
        if any(p.feature_contract_sha256 != contract for p in cohort):
            raise ValueError("feature contract differs from immutable MLflow reference")
        values: dict[str, dict] = {}
        if len(cohort) >= min_orders:
            for feature, dist in distributions.items():
                if any(feature not in row.features for row in cohort):
                    raise ValueError("missing canonical feature in snapshot")
                values[feature] = feature_drift(dist, [row.features[feature] for row in cohort])
        by_window.append({"start": iso(start), "end_exclusive": iso(end),
                          "n_unique_orders": len(cohort), "sufficient": len(cohort) >= min_orders,
                          "features": values})
    detected = sorted(set(by_window[0]["features"]) & set(by_window[1]["features"]))
    persistent = [feature for feature in detected if all(
        window["features"][feature]["value"] is not None and
        window["features"][feature]["value"] >= DRIFT_THRESHOLD for window in by_window)]
    return {"threshold": DRIFT_THRESHOLD, "windows": by_window,
            "persistent_features": persistent, "persistent_drift": bool(persistent),
            "sufficient_windows": all(w["sufficient"] for w in by_window)}
