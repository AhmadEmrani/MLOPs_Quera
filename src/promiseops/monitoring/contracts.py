"""Immutable monitoring boundaries based on company replay time."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from typing import Any

MIN_WINDOW_ORDERS = 100
MIN_MATURE_LABELS = 100
WINDOW_DAYS = 7
PREDICTION_RETENTION_DAYS = 60
DRIFT_THRESHOLD = 0.2


def instant(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of must be an offset-aware timestamp")
    return value.astimezone(UTC)


def iso(value: datetime) -> str:
    return instant(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Prediction:
    order_id: str
    model_version: str
    risk_probability: float
    scored_as_of: datetime
    feature_contract_sha256: str
    features: dict[str, Any]
    label: int | None = None
    batch_id: str = ""
    label_available_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id:
            raise ValueError("order identity is required")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError("model identity is required")
        if not math.isfinite(self.risk_probability) or not 0 <= self.risk_probability <= 1:
            raise ValueError("risk probability must be finite in [0,1]")
        if self.label is not None and (type(self.label) is not int or self.label not in (0, 1)):
            raise ValueError("label must be 0, 1 or None")
        instant(self.scored_as_of)
        if self.label is not None and self.label_available_at is None:
            raise ValueError("mature labels require a verified publication timestamp")
        if self.label_available_at is not None:
            instant(self.label_available_at)
        if not isinstance(self.features, dict):
            raise ValueError("features must be an object")


def unique_predictions(rows: list[Prediction], as_of: datetime) -> list[Prediction]:
    """First prediction per order, never let retries inflate cohorts."""
    limit = instant(as_of)
    selected: dict[str, Prediction] = {}
    for row in sorted(rows, key=lambda r: (instant(r.scored_as_of), r.batch_id, r.order_id)):
        if instant(row.scored_as_of) <= limit:
            selected.setdefault(row.order_id, row)
    return list(selected.values())


def monitoring_windows(as_of: datetime) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    t = instant(as_of)
    return ((t - timedelta(days=14), t - timedelta(days=7)),
            (t - timedelta(days=7), t))
