import math

import pytest

from promiseops.metrics import (
    RANDOM_REVIEW_RECALL,
    average_precision,
    brier,
    cohort_hash,
    constant_scores,
    cutoff,
    evaluate,
    is_late,
    needs_review,
    precision_at_k,
    prevalence,
    psi,
    recall_at_k,
    top_k,
    tvd,
)


def test_cutoff_is_ceiling_of_one_tenth_with_a_floor_of_one():
    assert cutoff(1) == 1
    assert cutoff(9) == 1
    assert cutoff(10) == 1
    assert cutoff(11) == 2
    assert cutoff(20) == 2
    assert cutoff(21) == 3
    # math.ceil(0.1 * 30) is 4 through binary floating point; the contract means 3.
    assert cutoff(30) == 3
    assert cutoff(100) == 10


def test_cutoff_rejects_empty_cohort():
    with pytest.raises(ValueError):
        cutoff(0)


def test_top_k_breaks_ties_by_ascending_order_id():
    # Equal risk everywhere: the deterministic order is the order ID itself.
    ids = ["c", "a", "b"]
    scores = [0.5, 0.5, 0.5]
    assert top_k(scores, ids, k=2) == [("a", 0.5), ("b", 0.5)]


def test_top_k_sorts_by_risk_descending_then_id():
    ids = ["b", "a", "c"]
    scores = [0.9, 0.9, 0.1]
    assert [order for order, _ in top_k(scores, ids, k=2)] == ["a", "b"]


def test_recall_and_precision_on_a_hand_computed_cohort():
    # K = 2 for N = 20. The top two are o00 (late) and o01 (on time), so one of the
    # two late orders is reviewed: recall 1/2, precision 1/2.
    labels = [1, 0, 1, 0] + [0] * 16
    scores = [0.9, 0.8, 0.7, 0.6] + [0.1] * 16
    ids = [f"o{index:02d}" for index in range(20)]
    assert cutoff(20) == 2
    assert recall_at_k(scores, labels, ids) == 0.5
    assert precision_at_k(scores, labels, ids) == 0.5


def test_recall_is_one_when_every_late_order_is_reviewed():
    labels = [1, 0, 1, 0] + [0] * 16
    scores = [0.9, 0.1, 0.8, 0.1] + [0.05] * 16
    ids = [f"o{index:02d}" for index in range(20)]
    assert recall_at_k(scores, labels, ids) == 1.0
    assert precision_at_k(scores, labels, ids) == 1.0


def test_recall_penalises_a_late_order_left_outside_the_list():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    ids = ["a", "b", "c", "d"]
    # K = 1, so only the highest-risk order is reviewed and neither late order makes it.
    assert cutoff(4) == 1
    assert recall_at_k(scores, labels, ids) == 0.0
    assert precision_at_k(scores, labels, ids) == 0.0


def test_recall_matches_the_random_review_baseline_definition():
    # A perfect ranking cannot do better than 1.0; a uniform ranking lands near 10%.
    n = 100
    ids = [f"o{index:03d}" for index in range(n)]
    labels = [1 if index < 10 else 0 for index in range(n)]
    perfect = [1.0 - index / n for index in range(n)]
    assert math.isclose(recall_at_k(perfect, labels, ids), 1.0)
    assert math.isclose(RANDOM_REVIEW_RECALL, 0.1)


def test_average_precision_matches_a_worked_example():
    labels = [1, 0, 1]
    scores = [0.9, 0.8, 0.7]
    ids = ["a", "b", "c"]
    # Hits at ranks 1 and 3: (1/1 + 2/3) / 2 late orders.
    assert math.isclose(average_precision(scores, labels, ids), (1.0 + 2 / 3) / 2)


def test_average_precision_is_one_for_a_perfect_ranking():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    ids = ["a", "b", "c", "d"]
    assert math.isclose(average_precision(scores, labels, ids), 1.0)


def test_brier_is_the_mean_squared_probability_error():
    scores = [0.9, 0.1, 0.4]
    labels = [1, 0, 1]
    expected = ((0.9 - 1) ** 2 + (0.1 - 0) ** 2 + (0.4 - 1) ** 2) / 3
    assert math.isclose(brier(scores, labels), expected)


def test_constant_baseline_uses_training_prevalence():
    labels = [1, 0, 0, 0]
    assert prevalence(labels) == 0.25
    assert constant_scores(labels) == [0.25] * 4


def test_evaluate_reports_cohort_size_and_prevalence():
    labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    ids = [f"o{index}" for index in range(10)]
    report = evaluate(scores, labels, ids)
    assert report["n"] == 10
    assert report["late_count"] == 1
    assert report["prevalence"] == 0.1
    assert report["k"] == 1
    assert report["recall_at_10pct"] == 1.0
    assert 0.0 <= report["brier"] <= 1.0


def test_metrics_reject_undefined_cohorts():
    with pytest.raises(ValueError):
        recall_at_k([0.5], [0], ["a"])  # no late orders
    with pytest.raises(ValueError):
        precision_at_k([0.5], [1], ["a", "b"])  # length mismatch
    with pytest.raises(ValueError):
        recall_at_k([0.5, 0.5], [1, 0], ["a", "a"])  # duplicate IDs
    with pytest.raises(ValueError):
        brier([1.5, 0.5], [1, 0])  # probability outside [0, 1]
    with pytest.raises(ValueError):
        brier([float("nan"), 0.5], [1, 0])


def test_late_uses_local_calendar_date_not_utc():
    from datetime import UTC, date, datetime

    # 2018-01-10 01:00 UTC is still 2018-01-09 23:00 in Sao Paulo (UTC-2 in January).
    later_same_local_day = datetime(2018, 1, 10, 1, 0, tzinfo=UTC)
    assert is_late(later_same_local_day, date(2018, 1, 9)) is False
    # 2018-01-10 03:00 UTC is 2018-01-10 01:00 local, the next promised day.
    next_local_day = datetime(2018, 1, 10, 3, 0, tzinfo=UTC)
    assert is_late(next_local_day, date(2018, 1, 9)) is True
    # A delivery before the promise is never late.
    assert is_late(datetime(2018, 1, 1, tzinfo=UTC), date(2018, 1, 9)) is False


def test_late_rejects_a_naive_timestamp():
    from datetime import date, datetime

    with pytest.raises(ValueError):
        is_late(datetime(2018, 1, 1), date(2018, 1, 2))


def test_cohort_hash_is_order_independent_and_distinguishes_cohorts():
    first = cohort_hash(["b", "a", "c"])
    assert first == cohort_hash(["a", "b", "c"])
    assert first != cohort_hash(["a", "b", "d"])
    assert len(first) == 64


def test_cohort_hash_rejects_duplicates():
    with pytest.raises(ValueError):
        cohort_hash(["a", "a"])


def test_psi_is_zero_for_identical_samples_and_large_for_a_shift():
    reference = [float(index) for index in range(100)]
    assert math.isclose(psi(reference, reference), 0.0, abs_tol=1e-9)
    shifted = [value + 50.0 for value in reference]
    assert psi(reference, shifted) > 0.2
    assert needs_review(psi(reference, shifted))


def test_tvd_is_bounded_and_detects_a_category_shift():
    assert tvd(["a", "b"], ["a", "b"]) == 0.0
    assert tvd(["a", "a"], ["b", "b"]) == 1.0
    distance = tvd(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert 0.0 < distance < 1.0
    assert not needs_review(tvd(["a", "a", "a", "b"], ["a", "a", "a", "b"]))
