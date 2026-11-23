"""Temporal edge tests: cutoffs and maturity independently of wall-clock time."""
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from promiseops.model_pipeline.split import select_temporal_split

TRAIN = datetime.fromisoformat("2018-01-31T23:59:59-03:00")
VAL = datetime.fromisoformat("2018-04-30T23:59:59-03:00")
NOW = datetime.fromisoformat("2018-06-29T23:59:59-03:00")


def record(i, approved, available, late):
    return {"order_id": f"ord{i}", "approved_at": approved, "label_available_at": available, "late": late}


def cases():
    return pd.DataFrame([
        record(1, TRAIN - timedelta(seconds=1), TRAIN, 0),
        record(2, TRAIN, TRAIN, 1),
        record(3, TRAIN + timedelta(seconds=1), NOW, 0),
        record(4, VAL, NOW, 1),
        record(5, TRAIN - timedelta(days=1), TRAIN + timedelta(seconds=1), 0),
        record(6, VAL + timedelta(seconds=1), NOW, 0),
        record(7, TRAIN + timedelta(days=1), NOW + timedelta(seconds=1), 1),
    ])


def test_boundary_inclusive_and_exclusions():
    split = select_temporal_split(cases(), train_end=TRAIN, validation_end=VAL,
                                  as_of=NOW, min_train_size=2, min_validation_size=2)
    assert set(split.train_order_ids) == {"ord1", "ord2"}
    assert set(split.validation_order_ids) == {"ord3", "ord4"}
    assert set(split.excluded["order_id"]) == {"ord5", "ord6", "ord7"}
    reasons = dict(zip(split.excluded["order_id"], split.excluded["exclusion_reason"]))
    assert reasons["ord5"] == "training_label_unavailable_by_train_end"
    assert reasons["ord7"] == "validation_label_unavailable_by_as_of"


def test_equivalent_utc_cutoffs_return_same_orders():
    local = select_temporal_split(cases(), train_end=TRAIN, validation_end=VAL,
                                  as_of=NOW, min_train_size=2, min_validation_size=2)
    utc = select_temporal_split(cases(), train_end=TRAIN.astimezone(UTC),
                                validation_end=VAL.astimezone(UTC), as_of=NOW.astimezone(UTC),
                                min_train_size=2, min_validation_size=2)
    assert utc.train_order_ids == local.train_order_ids
    assert utc.validation_order_ids == local.validation_order_ids


def test_cutoff_order_must_hold():
    with pytest.raises(ValueError, match="cutoffs"):
        select_temporal_split(cases(), train_end=VAL, validation_end=TRAIN,
                              as_of=NOW, min_train_size=2, min_validation_size=2)


def test_label_cannot_precede_order_approval():
    sample=cases()
    sample.loc[0,"label_available_at"] = TRAIN - timedelta(days=5)
    with pytest.raises(ValueError, match="label availability"):
        select_temporal_split(sample, train_end=TRAIN, validation_end=VAL,
                              as_of=NOW, min_train_size=2, min_validation_size=2)
