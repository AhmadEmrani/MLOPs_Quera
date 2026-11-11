"""Pandas-based chronological cohort selection for model training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Training, validation, and excluded rows with their cutoff metadata."""

    train_rows: pd.DataFrame
    validation_rows: pd.DataFrame
    excluded: pd.DataFrame
    train_end: datetime
    validation_end: datetime
    as_of: datetime

    @property
    def train_order_ids(self) -> tuple[str, ...]:
        return tuple(self.train_rows["order_id"].tolist())

    @property
    def validation_order_ids(self) -> tuple[str, ...]:
        return tuple(self.validation_rows["order_id"].tolist())

    @property
    def train_labels(self) -> tuple[int, ...]:
        return tuple(self.train_rows["late"].tolist())

    @property
    def validation_labels(self) -> tuple[int, ...]:
        return tuple(self.validation_rows["late"].tolist())


def _instant(value: datetime | str, *, field: str) -> datetime:
    """Parse a timezone-aware timestamp and normalize it to UTC."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_series(values: pd.Series, *, field: str) -> pd.Series:
    """Validate aware timestamps and return a UTC-normalized pandas series."""
    if values.isna().any():
        raise ValueError(f"{field} must not contain missing timestamps")
    try:
        aware = values.map(
            lambda value: pd.Timestamp(value).tzinfo is not None
            and pd.Timestamp(value).utcoffset() is not None
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain valid timestamps") from error
    if not aware.all():
        raise ValueError(f"{field} values must be timezone-aware")
    try:
        return pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain valid timestamps") from error


def _prepare_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate the labeled feature frame used to choose cohorts."""
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")

    frame = rows.copy()
    if "order_id" not in frame.columns and frame.index.name == "order_id":
        frame = frame.reset_index()

    required = {"order_id", "approved_at", "label_available_at", "late"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing required columns: {', '.join(sorted(missing))}")
    if frame["order_id"].isna().any():
        raise ValueError("order_id must not be missing")

    frame["order_id"] = frame["order_id"].astype(str)
    if frame["order_id"].eq("").any():
        raise ValueError("order_id must be non-empty")
    if frame["order_id"].duplicated().any():
        raise ValueError("order_id values must be unique in split input")

    valid_labels = frame["late"].notna() & frame["late"].isin((0, 1, False, True))
    if not valid_labels.all():
        raise ValueError("late labels must be non-missing binary values 0 or 1")
    frame["late"] = frame["late"].astype("int8")
    frame["approved_at"] = _utc_series(frame["approved_at"], field="approved_at")
    frame["label_available_at"] = _utc_series(
        frame["label_available_at"], field="label_available_at"
    )
    return frame


def _require_both_classes(rows: pd.DataFrame, *, cohort: str) -> None:
    if set(rows["late"].unique().tolist()) != {0, 1}:
        raise ValueError(f"{cohort} cohort must contain both on-time and late orders")


def select_temporal_split(
    rows: pd.DataFrame,
    *,
    train_end: datetime | str,
    validation_end: datetime | str,
    as_of: datetime | str,
    min_train_size: int = 2,
    min_validation_size: int = 2,
) -> TemporalSplit:
    """Select mature training and validation cohorts from labeled feature rows.

    Training rows must be approved by ``train_end`` and have labels available
    by ``train_end``. Validation rows must be approved after ``train_end`` and
    by ``validation_end``, with labels available by ``as_of``. Other rows are
    returned in ``excluded`` with an ``exclusion_reason`` column.

    The default cohort size is the minimum capable of containing both classes.
    Callers should pass larger limits when their evaluation policy requires
    them; the public project contract does not prescribe a numeric minimum.
    """
    train_cutoff = _instant(train_end, field="train_end")
    validation_cutoff = _instant(validation_end, field="validation_end")
    evaluation_time = _instant(as_of, field="as_of")
    if not train_cutoff < validation_cutoff <= evaluation_time:
        raise ValueError("cutoffs must satisfy train_end < validation_end <= as_of")
    if min_train_size < 2 or min_validation_size < 2:
        raise ValueError("minimum cohort sizes must be at least 2 to contain both classes")

    frame = _prepare_frame(rows)
    train_window = frame["approved_at"] <= train_cutoff
    validation_window = frame["approved_at"].gt(train_cutoff) & frame["approved_at"].le(
        validation_cutoff
    )
    train_mature = frame["label_available_at"] <= train_cutoff
    validation_mature = frame["label_available_at"] <= evaluation_time

    train_mask = train_window & train_mature
    validation_mask = validation_window & validation_mature
    excluded_mask = ~(train_mask | validation_mask)

    train_rows = frame.loc[train_mask].copy()
    validation_rows = frame.loc[validation_mask].copy()
    excluded = frame.loc[excluded_mask].copy()
    excluded["exclusion_reason"] = "outside_train_and_validation_windows"
    excluded.loc[train_window.loc[excluded.index] & ~train_mature.loc[excluded.index], "exclusion_reason"] = (
        "training_label_unavailable_by_train_end"
    )
    excluded.loc[
        validation_window.loc[excluded.index] & ~validation_mature.loc[excluded.index],
        "exclusion_reason",
    ] = "validation_label_unavailable_by_as_of"

    if len(train_rows) < min_train_size:
        raise ValueError(f"training cohort has {len(train_rows)} rows; minimum is {min_train_size}")
    if len(validation_rows) < min_validation_size:
        raise ValueError(
            f"validation cohort has {len(validation_rows)} rows; minimum is {min_validation_size}"
        )
    _require_both_classes(train_rows, cohort="training")
    _require_both_classes(validation_rows, cohort="validation")

    return TemporalSplit(
        train_rows=train_rows,
        validation_rows=validation_rows,
        excluded=excluded,
        train_end=train_cutoff,
        validation_end=validation_cutoff,
        as_of=evaluation_time,
    )
