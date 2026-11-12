"""Pydantic validation for canonical model feature rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from promiseops.contracts import SAFE_ID


FEATURE_COLUMNS = (
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
    "customer_state",
    "primary_seller_state",
    "primary_category",
    "primary_payment_type",
)


class FeatureRow(BaseModel):
    """One prediction-time feature record, excluding labels and outcome data."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    order_id: str = Field(pattern=SAFE_ID)
    item_count: int = Field(ge=1)
    seller_count: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    total_weight_g: float = Field(ge=0)
    payment_total: float = Field(ge=0)
    max_installments: int = Field(ge=0, le=100)
    promise_days: int = Field(ge=0)
    approval_hour: int = Field(ge=0, le=23)
    approval_weekday: int = Field(ge=0, le=6)
    interstate_fraction: float = Field(ge=0, le=1)
    customer_state: str = Field(pattern=r"^[A-Z]{2}$")
    primary_seller_state: str = Field(pattern=r"^[A-Z]{2}$")
    primary_category: str = Field(min_length=1, max_length=100)
    primary_payment_type: str = Field(min_length=1, max_length=40)


class LabeledFeatureRow(FeatureRow):
    """Training/evaluation row with temporal metadata and a binary target."""

    approved_at: datetime
    label_available_at: datetime
    late: Literal[0, 1]

    @field_validator("approved_at", "label_available_at")
    @classmethod
    def require_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(UTC)


def _records(rows: Any) -> list[Mapping[str, Any]]:
    """Convert a DataFrame or iterable of mappings to Pydantic input rows."""
    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        frame = rows
        if "order_id" not in frame.columns and getattr(frame.index, "name", None) == "order_id":
            frame = frame.reset_index()
        if "order_id" not in frame.columns:
            raise ValueError("feature DataFrame must have an order_id column or index")
        return frame.to_dict(orient="records")

    if isinstance(rows, Mapping):
        raise TypeError("rows must be a collection of feature records, not one mapping")
    records = list(rows) if isinstance(rows, Iterable) else None
    if records is None:
        raise TypeError("rows must be a pandas DataFrame or an iterable of mappings")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError("every feature record must be a mapping")
    return records


def validate_feature_rows(
    rows: Any,
    *,
    labeled: bool = False,
) -> list[FeatureRow] | list[LabeledFeatureRow]:
    """Validate feature records and reject duplicate IDs or malformed cohorts.

    Unlabeled rows may contain only ``order_id`` and the canonical feature
    columns. Labeled rows additionally require approval time, label availability
    time, and ``late``; all three remain metadata and are never predictors.
    """
    records = _records(rows)
    if not records:
        raise ValueError("feature cohort must not be empty")

    model_type = LabeledFeatureRow if labeled else FeatureRow
    validated = TypeAdapter(list[model_type]).validate_python(records)
    order_ids = [record.order_id for record in validated]
    if len(set(order_ids)) != len(order_ids):
        raise ValueError("order_id values must be unique in the feature cohort")
    return validated
