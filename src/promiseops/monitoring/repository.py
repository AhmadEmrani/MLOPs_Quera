"""Read-only monitoring projection. Never selects future outcomes as labels."""
from __future__ import annotations
from datetime import timedelta
from typing import Any
from promiseops.monitoring.contracts import Prediction, instant

# Label availability is the maximum of source outcome publication and batch
# publication. Legacy NULL batch timestamps use the conservative receipt maximum;
# if no outcome receipts exist the label remains unavailable (NULL).
READ_MONITORING = """
SELECT DISTINCT ON (p.order_id)
    p.order_id,
    p.model_version,
    p.risk_probability,
    p.scored_as_of,
    p.feature_contract_sha256,
    p.features_jsonb,
    CASE WHEN out.order_id IS NOT NULL
       AND out.available_at <= %(as_of)s
       AND COALESCE(out.batch_available_at,
          (SELECT MAX(available_at) FROM batch_receipts WHERE kind = 'outcomes')) <= %(as_of)s
       AND out.delivered_at <= %(as_of)s
    THEN ((out.delivered_at AT TIME ZONE 'America/Sao_Paulo')::date > ord.promised_date)::integer
    ELSE NULL END AS mature_label,
    p.batch_id,
    CASE WHEN out.order_id IS NOT NULL
       AND out.available_at <= %(as_of)s
       AND COALESCE(out.batch_available_at,
          (SELECT MAX(available_at) FROM batch_receipts WHERE kind = 'outcomes')) <= %(as_of)s
       AND out.delivered_at <= %(as_of)s
    THEN GREATEST(out.available_at, COALESCE(out.batch_available_at,
          (SELECT MAX(available_at) FROM batch_receipts WHERE kind = 'outcomes')))
    ELSE NULL END AS label_released_at
FROM monitoring_predictions p
JOIN orders ord ON ord.order_id = p.order_id
LEFT JOIN outcomes out ON out.order_id = p.order_id
WHERE p.model_version = %(model_version)s
  AND p.scored_as_of >= %(earliest)s
  AND p.scored_as_of <= %(as_of)s
  AND ord.batch_available_at <= p.scored_as_of
ORDER BY p.order_id, p.scored_as_of ASC, p.batch_id ASC
LIMIT %(row_limit)s
"""


def fetch_predictions(conn: Any, *, model_version: str, as_of: object,
                      max_rows: int = 100000) -> list[Prediction]:
    """Earliest company-time prediction per order, max 60 company days.

    The SQL returns labels *only* when the source and receipt cutoffs have
    matured at as_of. No staging/production changes, DDL, or model writes.
    """
    if not isinstance(model_version, str) or not model_version or len(model_version) > 128:
        raise ValueError("model_version is required")
    if not 1 <= max_rows <= 1000000:
        raise ValueError("invalid monitoring row cap")
    t = instant(as_of)
    params = {"model_version": model_version, "as_of": t,
              "earliest": t - timedelta(days=60), "row_limit": max_rows + 1}
    with conn.cursor() as cursor:
        cursor.execute(READ_MONITORING, params)
        rows = cursor.fetchall()
    if len(rows) > max_rows:
        raise ValueError("monitoring row cap exceeded; split cohort, never silently truncate")
    return [Prediction(order_id=o, model_version=v, risk_probability=float(risk),
                       scored_as_of=instant(scored), feature_contract_sha256=contract,
                       features=features, label=(int(label) if label is not None else None),
                       batch_id=batch, label_available_at=(instant(released) if released is not None else None))
            for o, v, risk, scored, contract, features, label, batch, released in rows]


def connect_read_only(dsn: str):
    """Caller owns connection lifetime; no secrets in exceptions/logs."""
    if not dsn:
        raise ValueError("database URL must be provided via approved runtime secret")
    import psycopg
    if dsn.startswith("postgresql+psycopg://"):
        dsn = "postgresql://" + dsn[len("postgresql+psycopg://"):]
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    return psycopg.connect(dsn, connect_timeout=10,
                           options="-c default_transaction_read_only=on -c statement_timeout=30000")
