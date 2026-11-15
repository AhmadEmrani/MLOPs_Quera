"""Own-team transactional ingestion and a single leakage-safe feature builder."""

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
import psycopg

from promiseops.contracts import LOCAL_TZ, Batch, digest, instant

# دستورات ساخت جدول‌ها که در مرحله ۲ طراحی کردیم
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(128) PRIMARY KEY,
    customer_state CHAR(2) NOT NULL CHECK (customer_state ~ '^[A-Z]{2}$')
);

CREATE TABLE IF NOT EXISTS sellers (
    seller_id VARCHAR(128) PRIMARY KEY,
    seller_state CHAR(2) NOT NULL CHECK (seller_state ~ '^[A-Z]{2}$')
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(128) PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    weight_g DOUBLE PRECISION NOT NULL CHECK (weight_g >= 0)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(128) PRIMARY KEY,
    customer_id VARCHAR(128) NOT NULL REFERENCES customers(customer_id),
    approved_at TIMESTAMPTZ NOT NULL,
    promised_date DATE NOT NULL,
    batch_available_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    order_id VARCHAR(128) NOT NULL REFERENCES orders(order_id),
    item_id INTEGER NOT NULL CHECK (item_id >= 1),
    product_id VARCHAR(128) NOT NULL REFERENCES products(product_id),
    seller_id VARCHAR(128) NOT NULL REFERENCES sellers(seller_id),
    price DOUBLE PRECISION NOT NULL CHECK (price >= 0),
    freight DOUBLE PRECISION NOT NULL CHECK (freight >= 0),
    PRIMARY KEY (order_id, item_id)
);

CREATE TABLE IF NOT EXISTS payments (
    order_id VARCHAR(128) NOT NULL REFERENCES orders(order_id),
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    payment_type VARCHAR(40) NOT NULL,
    installments INTEGER NOT NULL CHECK (installments >= 0),
    value DOUBLE PRECISION NOT NULL CHECK (value >= 0),
    PRIMARY KEY (order_id, sequence)
);

CREATE TABLE IF NOT EXISTS outcomes (
    order_id VARCHAR(128) PRIMARY KEY REFERENCES orders(order_id),
    delivered_at TIMESTAMPTZ NOT NULL,
    available_at TIMESTAMPTZ NOT NULL CHECK (available_at >= delivered_at)
);

CREATE TABLE IF NOT EXISTS batch_receipts (
    batch_id VARCHAR(128) PRIMARY KEY,
    kind VARCHAR(16) NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    payload_sha256 CHAR(64) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS batch_quarantine (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(128),
    payload_sha256 CHAR(64),
    reason TEXT NOT NULL,
    raw_payload TEXT NOT NULL,
    quarantined_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    request_id VARCHAR(128) PRIMARY KEY,
    order_id VARCHAR(128) NOT NULL REFERENCES orders(order_id),
    risk_probability DOUBLE PRECISION NOT NULL CHECK (risk_probability >= 0 AND risk_probability <= 1),
    model_version VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _get_raw_conn(connection: Any):
    """کمک می‌کند چه ورودی SQLAlchemy باشد چه psycopg، به اتصال اصلی دسترسی داشته باشیم."""
    if hasattr(connection, "driver_connection"):
        return connection.driver_connection
    if hasattr(connection, "connection"):
        return connection.connection
    return connection


def init_db(connection: Any) -> None:
    """ساخت تمام جدول‌های مورد نیاز در صورت عدم وجود."""
    raw_conn = _get_raw_conn(connection)
    with raw_conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    raw_conn.commit()


@contextmanager
def training_snapshot(connection):
    """Use one repeatable-read, read-only snapshot for a training run's reads.

    Schema creation runs *before* the snapshot because init_db commits. No
    ingest or serving transaction behavior is changed by this context manager.
    """
    if hasattr(connection, "exec_driver_sql"):
        # Compatibility with SQLAlchemy Connection; training worker uses psycopg.
        connection.exec_driver_sql(SCHEMA_SQL)
        connection.commit()
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield
    elif hasattr(connection, "transaction"):
        init_db(connection)
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            yield
    else:
        raise TypeError("training_snapshot requires a psycopg or SQLAlchemy connection")


def _training_query(connection, statement: str, parameters: dict[str, Any]) -> list[tuple]:
    """Execute a read query on the caller's existing snapshot."""
    if hasattr(connection, "exec_driver_sql"):
        from sqlalchemy import text

        return list(connection.execute(text(statement), parameters).all())
    # Static internal SQL only; bind values using psycopg named placeholders.
    for parameter in parameters:
        statement = statement.replace(f":{parameter}", f"%({parameter})s")
    with connection.cursor() as cursor:
        cursor.execute(statement, parameters)
        return cursor.fetchall()


def training_source_lineage(
    connection,
    *,
    as_of: datetime,
    train_order_ids: tuple[str, ...],
    validation_order_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize source receipts and customer overlap within the same snapshot.

    Receipts are an observed upstream inventory, NOT a guaranteed mapping of
    specific orders to batches: the database doesn't store that relationship.
    """
    receipts = _training_query(
        connection,
        """SELECT batch_id, kind, available_at, payload_sha256
           FROM batch_receipts WHERE available_at <= :as_of
           ORDER BY available_at, kind, batch_id""",
        {"as_of": as_of},
    )
    train_ids = list(train_order_ids)
    validation_ids = list(validation_order_ids)
    all_ids = train_ids + validation_ids
    overlap = _training_query(
        connection,
        """SELECT COUNT(*) FROM (
               SELECT customer_id FROM orders
               WHERE order_id = ANY(CAST(:all_ids AS varchar[]))
               GROUP BY customer_id
               HAVING BOOL_OR(order_id = ANY(CAST(:train_ids AS varchar[])))
                  AND BOOL_OR(order_id = ANY(CAST(:validation_ids AS varchar[])))
           ) AS overlapping_customers""",
        {"all_ids": all_ids, "train_ids": train_ids, "validation_ids": validation_ids},
    )[0][0]
    availability_audit = _training_query(
        connection,
        """SELECT COUNT(*) FROM orders
           WHERE order_id = ANY(CAST(:all_ids AS varchar[]))
             AND batch_available_at > approved_at""",
        {"all_ids": all_ids},
    )[0][0]
    return {
        "receipts_scope": "all_receipts_available_by_as_of_not_proven_consumed",
        "observed_receipts": [
            {
                "batch_id": batch_id,
                "kind": kind,
                "available_at": available_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "payload_sha256": payload_sha256,
            }
            for batch_id, kind, available_at, payload_sha256 in receipts
        ],
        "customer_overlap_n": int(overlap),
        # This is an audit signal; feed publication time may differ from the
        # time approval-time features became known in the source system.
        "order_batch_after_approval_n": int(availability_audit),
    }


def _quarantine_and_raise(raw_conn, batch_id: str | None, sha256: str | None, reason: str, raw_text: str):
    """برگرداندن تغییرات ناقص (Rollback)، ثبت بسته خراب در جدول قرنطینه و اعلام خطا."""
    raw_conn.rollback()
    with raw_conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute(
            """
            INSERT INTO batch_quarantine (batch_id, payload_sha256, reason, raw_payload)
            VALUES (%s, %s, %s, %s)
            """,
            (batch_id, sha256, reason, raw_text),
        )
    raw_conn.commit()
    raise ValueError(f"Batch quarantined: {reason}")


def _check_no_duplicates(items_list: list, key_fn, label: str):
    """بررسی اینکه داخل خود بسته ورودی، کلید تکراری وجود نداشته باشد."""
    seen = set()
    for item in items_list:
        k = key_fn(item)
        if k in seen:
            raise ValueError(f"Duplicate key in batch for {label}: {k}")
        seen.add(k)


def ingest_batch(connection, payload, *, as_of, expected_sha256=None):
    """Return a receipt. Validate, quarantine invalid input, commit once, handle retry."""
    raw_conn = _get_raw_conn(connection)
    init_db(raw_conn)

    # 1. تبدیل ورودی به دیکشنری و متن خام (برای ذخیره در قرنطینه در صورت بروز خطا)
    try:
        if isinstance(payload, (str, bytes)):
            raw_text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            data = json.loads(raw_text)
        elif isinstance(payload, dict):
            data = payload
            raw_text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        else:
            raise TypeError("Unsupported payload format")
    except (ValueError, TypeError, KeyError, Exception) as exc:  # noqa: BLE001
        _quarantine_and_raise(raw_conn, None, None, f"Invalid JSON: {exc}", str(payload))

    batch_id_hint = data.get("batch_id") if isinstance(data, dict) else None

    # 2. محاسبه و بررسی Checksum (کد امنیتی SHA256)
    try:
        actual_sha256 = digest(data)
    except (ValueError, TypeError, KeyError, Exception) as exc:  # noqa: BLE001
        _quarantine_and_raise(raw_conn, batch_id_hint, None, f"Digest calculation failed: {exc}", raw_text)

    if expected_sha256 and actual_sha256 != expected_sha256:
        _quarantine_and_raise(
            raw_conn, batch_id_hint, actual_sha256, "Checksum mismatch (payload_sha256)", raw_text
        )

    # 3. اعتبارسنجی ساختار و قوانین پایه با Pydantic (کلاس Batch در contracts.py)
    try:
        batch = Batch.model_validate(data)
        as_of_utc = instant(as_of) if as_of is not None else None
        if as_of_utc and batch.available_at > as_of_utc:
            raise ValueError("Batch available_at is in the future relative to as_of")
    except (ValueError, TypeError, KeyError, Exception) as exc:  # noqa: BLE001
        _quarantine_and_raise(
            raw_conn, batch_id_hint, actual_sha256, f"Contract validation failed: {exc}", raw_text
        )

    # 4. شروع تراکنش دیتابیس (یا همه چیز با هم ثبت می‌شود یا هیچ‌چیز!)
    try:
        with raw_conn.cursor() as cur:
            # الف) بررسی تکراری بودن Batch (قانون Idempotency)
            cur.execute(
                "SELECT payload_sha256, available_at, kind FROM batch_receipts WHERE batch_id = %s",
                (batch.batch_id,),
            )
            existing_receipt = cur.fetchone()
            if existing_receipt:
                existing_sha, existing_avail, existing_kind = existing_receipt
                if existing_sha == actual_sha256:
                    # دقیقاً همان بسته قبلی است! بدون دست زدن به دیتابیس، رسید را برگردان
                    raw_conn.rollback()
                    return {
                        "batch_id": batch.batch_id,
                        "kind": existing_kind,
                        "available_at": existing_avail.isoformat(),
                        "payload_sha256": existing_sha,
                        "status": "noop",
                    }
                else:
                    raise ValueError(f"Batch ID {batch.batch_id} already committed with a different payload")

            # ب) اگر بسته از نوع سفارش‌ها (orders) است:
            if batch.kind == "orders":
                # بررسی نبود کلید تکراری داخل خود بسته
                _check_no_duplicates(batch.customers, lambda x: x.customer_id, "customers")
                _check_no_duplicates(batch.sellers, lambda x: x.seller_id, "sellers")
                _check_no_duplicates(batch.products, lambda x: x.product_id, "products")
                _check_no_duplicates(batch.orders, lambda x: x.order_id, "orders")
                _check_no_duplicates(batch.items, lambda x: (x.order_id, x.item_id), "items")
                _check_no_duplicates(batch.payments, lambda x: (x.order_id, x.sequence), "payments")

                # بررسی اینکه هر سفارش در این بسته حداقل 1 کالا و 1 پرداخت داشته باشد
                # و هیچ کالا یا پرداختی برای سفارشی خارج از این بسته نباشد
                batch_order_ids = {o.order_id for o in batch.orders}
                item_order_ids = {i.order_id for i in batch.items}
                payment_order_ids = {p.order_id for p in batch.payments}

                if item_order_ids != batch_order_ids:
                    raise ValueError(
                        "Every order in batch must have items, and items cannot belong to external orders"
                    )
                if payment_order_ids != batch_order_ids:
                    raise ValueError(
                        "Every order in batch must have payments, and payments cannot belong to external orders"
                    )

                # ثبت مشتریان (بررسی تغییرناپذیری: اگر قبلاً بوده، ایالتش نباید عوض شده باشد)
                for c in batch.customers:
                    cur.execute(
                        "SELECT customer_state FROM customers WHERE customer_id = %s", (c.customer_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        if row[0] != c.customer_state:
                            raise ValueError(f"Immutable customer {c.customer_id} changed state")
                    else:
                        cur.execute(
                            "INSERT INTO customers (customer_id, customer_state) VALUES (%s, %s)",
                            (c.customer_id, c.customer_state),
                        )

                # ثبت فروشندگان (بررسی تغییرناپذیری)
                for s in batch.sellers:
                    cur.execute("SELECT seller_state FROM sellers WHERE seller_id = %s", (s.seller_id,))
                    row = cur.fetchone()
                    if row:
                        if row[0] != s.seller_state:
                            raise ValueError(f"Immutable seller {s.seller_id} changed state")
                    else:
                        cur.execute(
                            "INSERT INTO sellers (seller_id, seller_state) VALUES (%s, %s)",
                            (s.seller_id, s.seller_state),
                        )

                # ثبت محصولات (بررسی تغییرناپذیری)
                for p in batch.products:
                    cur.execute(
                        "SELECT category, weight_g FROM products WHERE product_id = %s", (p.product_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        if row[0] != p.category or float(row[1]) != float(p.weight_g):
                            raise ValueError(f"Immutable product {p.product_id} changed attributes")
                    else:
                        cur.execute(
                            "INSERT INTO products (product_id, category, weight_g) VALUES (%s, %s, %s)",
                            (p.product_id, p.category, p.weight_g),
                        )

                # ثبت سفارش‌ها (سفارش تکراری که قبلاً در دیتابیس بوده نباید دوباره بیاید)
                for o in batch.orders:
                    cur.execute("SELECT 1 FROM orders WHERE order_id = %s", (o.order_id,))
                    if cur.fetchone():
                        raise ValueError(f"Order {o.order_id} already exists and is immutable")
                    cur.execute(
                        """
                        INSERT INTO orders (order_id, customer_id, approved_at, promised_date, batch_available_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (o.order_id, o.customer_id, o.approved_at, o.promised_date, batch.available_at),
                    )

                # ثبت اقلام سفارش (Items)
                for i in batch.items:
                    cur.execute(
                        """
                        INSERT INTO items (order_id, item_id, product_id, seller_id, price, freight)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (i.order_id, i.item_id, i.product_id, i.seller_id, i.price, i.freight),
                    )

                # ثبت پرداخت‌ها (Payments)
                for pay in batch.payments:
                    cur.execute(
                        """
                        INSERT INTO payments (order_id, sequence, payment_type, installments, value)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (pay.order_id, pay.sequence, pay.payment_type, pay.installments, pay.value),
                    )

            # ج) اگر بسته از نوع نتایج تحویل (outcomes) است:
            elif batch.kind == "outcomes":
                _check_no_duplicates(batch.outcomes, lambda x: x.order_id, "outcomes")

                for out in batch.outcomes:
                    # بررسی اینکه سفارش اصلی در دیتابیس وجود داشته باشد و زمان تحویل قبل از تایید نباشد
                    cur.execute("SELECT approved_at FROM orders WHERE order_id = %s", (out.order_id,))
                    order_row = cur.fetchone()
                    if not order_row:
                        raise ValueError(f"Outcome references unknown order_id {out.order_id}")
                    approved_at = order_row[0]
                    if out.delivered_at < approved_at:
                        raise ValueError(f"Delivery precedes approval for order {out.order_id}")

                    # بررسی تغییرناپذیری Outcome
                    cur.execute(
                        "SELECT delivered_at, available_at FROM outcomes WHERE order_id = %s",
                        (out.order_id,),
                    )
                    existing_out = cur.fetchone()
                    if existing_out:
                        if existing_out[0] != out.delivered_at or existing_out[1] != out.available_at:
                            raise ValueError(f"Immutable outcome for order {out.order_id} changed")
                    else:
                        cur.execute(
                            """
                            INSERT INTO outcomes (order_id, delivered_at, available_at)
                            VALUES (%s, %s, %s)
                            """,
                            (out.order_id, out.delivered_at, out.available_at),
                        )

            # د) ثبت رسید نهایی (Receipt) در همان تراکنش
            cur.execute(
                """
                INSERT INTO batch_receipts (batch_id, kind, available_at, payload_sha256)
                VALUES (%s, %s, %s, %s)
                """,
                (batch.batch_id, batch.kind, batch.available_at, actual_sha256),
            )

        # اگر همه مراحل بالا بدون خطا انجام شد، تغییرات را نهایی (Commit) کن
        raw_conn.commit()
        return {
            "batch_id": batch.batch_id,
            "kind": batch.kind,
            "available_at": batch.available_at.isoformat(),
            "payload_sha256": actual_sha256,
            "status": "committed",
        }

    except (ValueError, TypeError, KeyError, Exception) as exc:  # noqa: BLE001
        # اگر هر خطایی (مثل نبود کلید خارجی یا تغییر داده) رخ داد، همه چیز را برگردان و قرنطینه کن
        _quarantine_and_raise(raw_conn, batch.batch_id, actual_sha256, str(exc), raw_text)


FEATURE_COLUMNS = [
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
]


def build_features(
    connection, order_ids=None, *, as_of, labeled=False, ensure_schema: bool = True
) -> pd.DataFrame:
    """Aggregate relations without fan-out or outcome-derived features.

    Training calls this with ensure_schema=False while its read-only snapshot
    is active. Serving and other callers retain the original default.
    """
    raw_conn = _get_raw_conn(connection)
    if ensure_schema:
        init_db(raw_conn)
    as_of_utc = instant(as_of)

    # اگر لیست خاصی از order_idها داده شده، بررسی کنیم که خالی نباشد
    if order_ids is not None:
        order_ids_list = list(order_ids)
        if not order_ids_list:
            cols = ["order_id", *FEATURE_COLUMNS]
            if labeled:
                cols.extend(["approved_at", "label_available_at", "late"])
            return pd.DataFrame(columns=cols)
    else:
        order_ids_list = None

    # کوئری SQL ضد Fan-out با استفاده از CTE (جدول‌های موقت مستقل)
    query = """
    WITH target_orders AS (
        SELECT
            o.order_id,
            o.customer_id,
            o.approved_at,
            o.promised_date,
            c.customer_state
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.approved_at <= %(as_of)s
          AND o.batch_available_at <= %(as_of)s
          AND (%(order_ids)s::varchar[] IS NULL OR o.order_id = ANY(%(order_ids)s::varchar[]))
    ),
    -- 1. جمع‌بندی مستقل جدول اقلام (Items) به ازای هر سفارش
    item_agg AS (
        SELECT
            i.order_id,
            COUNT(*):: INTEGER AS item_count,
            COUNT(DISTINCT i.seller_id)::INTEGER AS seller_count,
            SUM(i.price)::DOUBLE PRECISION AS total_price,
            SUM(i.freight)::DOUBLE PRECISION AS total_freight,
            SUM(p.weight_g)::DOUBLE PRECISION AS total_weight_g,
            AVG(CASE WHEN s.seller_state <> t.customer_state THEN 1.0 ELSE 0.0 END)::DOUBLE PRECISION AS interstate_fraction
        FROM target_orders t
        JOIN items i ON t.order_id = i.order_id
        JOIN products p ON i.product_id = p.product_id
        JOIN sellers s ON i.seller_id = s.seller_id
        GROUP BY i.order_id
    ),
    -- انتخاب فروشنده و دسته‌بندیِ کالای اصلی هر سفارش (گران‌ترین قلم، در صورت تساوی اولین item_id)
    primary_item AS (
        SELECT DISTINCT ON (i.order_id)
            i.order_id,
            s.seller_state AS primary_seller_state,
            p.category AS primary_category
        FROM target_orders t
        JOIN items i ON t.order_id = i.order_id
        JOIN products p ON i.product_id = p.product_id
        JOIN sellers s ON i.seller_id = s.seller_id
        ORDER BY i.order_id, i.price DESC, i.item_id ASC
    ),
    -- 2. جمع‌بندی مستقل جدول پرداخت‌ها (Payments) به ازای هر سفارش
    payment_agg AS (
        SELECT
            pay.order_id,
            SUM(pay.value)::DOUBLE PRECISION AS payment_total,
            MAX(pay.installments)::INTEGER AS max_installments
        FROM target_orders t
        JOIN payments pay ON t.order_id = pay.order_id
        GROUP BY pay.order_id
    ),
    -- انتخاب روش پرداخت اصلی هر سفارش (بزرگترین مبلغ، در صورت تساوی اولین sequence)
    primary_payment AS (
        SELECT DISTINCT ON (pay.order_id)
            pay.order_id,
            pay.payment_type AS primary_payment_type
        FROM target_orders t
        JOIN payments pay ON t.order_id = pay.order_id
        ORDER BY pay.order_id, pay.value DESC, pay.sequence ASC
    )
    SELECT
        t.order_id,
        t.approved_at,
        t.promised_date,
        ia.item_count,
        ia.seller_count,
        ia.total_price,
        ia.total_freight,
        ia.total_weight_g,
        pa.payment_total,
        pa.max_installments,
        ia.interstate_fraction,
        t.customer_state,
        pi.primary_seller_state,
        pi.primary_category,
        pp.primary_payment_type,
        out.delivered_at,
        out.available_at AS outcome_available_at
    FROM target_orders t
    JOIN item_agg ia ON t.order_id = ia.order_id
    JOIN primary_item pi ON t.order_id = pi.order_id
    JOIN payment_agg pa ON t.order_id = pa.order_id
    JOIN primary_payment pp ON t.order_id = pp.order_id
    LEFT JOIN outcomes out
      ON t.order_id = out.order_id
     AND out.available_at <= %(as_of)s
    ORDER BY t.order_id ASC
    """

    with raw_conn.cursor() as cur:
        cur.execute(query, {"as_of": as_of_utc, "order_ids": order_ids_list})
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=colnames)
    if df.empty:
        cols = ["order_id", *FEATURE_COLUMNS]
        if labeled:
            cols.extend(["approved_at", "label_available_at", "late"])
        return pd.DataFrame(columns=cols)

    # اگر labeled=True باشد، فقط سفارش‌هایی که تا لحظه as_of نتیجه تحویلشان آمده را نگه می‌داریم
    if labeled:
        df = df[df["delivered_at"].notna() & df["outcome_available_at"].notna()].copy()
        if df.empty:
            return pd.DataFrame(
                columns=["order_id", *FEATURE_COLUMNS, "approved_at", "label_available_at", "late"]
            )

    # محاسبه ویژگی‌های زمانی بر اساس منطقه زمانی برزیل (America/Sao_Paulo)
    local_approved = df["approved_at"].apply(lambda dt: dt.astimezone(LOCAL_TZ))
    local_approved_date = local_approved.apply(lambda dt: dt.date())

    df["promise_days"] = [
        (prom - app_date).days for prom, app_date in zip(df["promised_date"], local_approved_date)
    ]
    df["approval_hour"] = local_approved.apply(lambda dt: dt.hour).astype(int)
    df["approval_weekday"] = local_approved.apply(lambda dt: dt.weekday()).astype(int)

    # اگر labeled=True باشد، ستون هدف (late) را طبق قرارداد حساب می‌کنیم:
    # late = local_date(delivered_at) > promised_date
    if labeled:
        local_delivered_date = df["delivered_at"].apply(lambda dt: dt.astimezone(LOCAL_TZ).date())
        df["late"] = [
            int(deliv_date > prom_date)
            for deliv_date, prom_date in zip(local_delivered_date, df["promised_date"])
        ]
        df["label_available_at"] = df["outcome_available_at"]
        return df[
            ["order_id", *FEATURE_COLUMNS, "approved_at", "label_available_at", "late"]
        ].reset_index(drop=True)

    # در حالت پیش‌بینی (Serving)، فقط شناسه سفارش و 15 ویژگی مجاز برگردانده می‌شوند
    return df[["order_id", *FEATURE_COLUMNS]].reset_index(drop=True)


class ConflictError(ValueError):
    """خطای ویژه زمانی که یک request_id تکراری برای یک order_id متفاوت ارسال شود (کد 409 در API)."""


def get_connection(db_url: str | None = None):
    """ایجاد اتصال به دیتابیس PostgreSQL از روی متغیر محیطی PROMISEOPS_DATABASE_URL."""

    url = db_url or os.environ.get("PROMISEOPS_DATABASE_URL")
    if not url:
        raise RuntimeError("PROMISEOPS_DATABASE_URL is not configured")
    # SQLAlchemy گاهی اول آدرس postgresql+psycopg:// می‌گذارد که برای psycopg خالص تمیزش می‌کنیم
    clean_url = url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(clean_url)


def get_or_save_prediction(
    connection,
    *,
    request_id: str,
    order_id: str,
    compute_fn,
) -> dict[str, Any]:
    """
    مدیریت جدول predictions برای سرویس FastAPI:
    - اگر request_id قبلاً با همین order_id ثبت شده: همان نتیجه قبلی را برمی‌گرداند.
    - اگر request_id قبلاً با یک order_id دیگر ثبت شده: خطای ConflictError (برای کد 409) می‌دهد.
    - اگر جدید است: تابع compute_fn() را صدا می‌زند، نتیجه را در دیتابیس ذخیره می‌کند و برمی‌گرداند.
    """
    raw_conn = _get_raw_conn(connection)
    init_db(raw_conn)

    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT order_id, risk_probability, model_version FROM predictions WHERE request_id = %s",
            (request_id,),
        )
        row = cur.fetchone()
        if row:
            saved_order_id, saved_prob, saved_version = row
            if saved_order_id != order_id:
                raise ConflictError(
                    f"request_id {request_id} already used for order {saved_order_id}, not {order_id}"
                )
            return {
                "order_id": saved_order_id,
                "request_id": request_id,
                "risk_probability": float(saved_prob),
                "model_version": saved_version,
            }

    # اگر تکراری نبود، پیش‌بینی را محاسبه کن (خروجی compute_fn باید شامل risk_probability و model_version باشد)
    result = compute_fn()
    risk_prob = float(result["risk_probability"])
    model_ver = str(result["model_version"])

    try:
        with raw_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (request_id, order_id, risk_probability, model_version)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (request_id) DO NOTHING
                """,
                (request_id, order_id, risk_prob, model_ver),
            )
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise

    # دوباره از دیتابیس می‌خوانیم تا اگر دو درخواست همزمان (Race Condition) آمده بودند، جواب یکسان برگردد
    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT order_id, risk_probability, model_version FROM predictions WHERE request_id = %s",
            (request_id,),
        )
        saved_order_id, saved_prob, saved_version = cur.fetchone()
        if saved_order_id != order_id:
            raise ConflictError(f"request_id {request_id} conflict on concurrent insert")
        return {
            "order_id": saved_order_id,
            "request_id": request_id,
            "risk_probability": float(saved_prob),
            "model_version": saved_version,
        }


def sync_feed(connection, *, base_url: str, token: str, as_of: str | datetime | None = None) -> list[dict]:
    """
    دریافت لیست دسته‌های داده (Batches) از سرور شرکت (GET /feed)،
    مرتب‌سازی بر اساس زمان انتشار (و اولویت orders قبل از outcomes در زمان برابر)،
    بررسی file_sha256 و payload_sha256، و وارد کردن آن‌ها به PostgreSQL.
    """

    headers = {"Authorization": f"Bearer {token}"}
    params = {}
    if as_of is not None:
        params["as_of"] = instant(as_of).isoformat()

    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        resp = client.get("/feed", params=params)
        resp.raise_for_status()
        feed_data = resp.json()

        entries = feed_data.get("entries", feed_data if isinstance(feed_data, list) else [])
        effective_as_of = feed_data.get("as_of", as_of) if isinstance(feed_data, dict) else as_of

        # مرتب‌سازی طبق قرارداد: اول زمان انتشار، و در زمان برابر اول orders (0) بعد outcomes (1)
        def _sort_key(entry: dict):
            avail = instant(entry["available_at"])
            kind_rank = 0 if entry.get("kind") == "orders" else 1
            return (avail, kind_rank, entry.get("batch_id", ""))

        sorted_entries = sorted(entries, key=_sort_key)
        receipts = []

        for entry in sorted_entries:
            file_name = entry.get("file") or entry.get("path") or entry.get("batch_id")
            file_resp = client.get(f"/feed/{str(file_name).lstrip('/')}")
            file_resp.raise_for_status()
            raw_bytes = file_resp.content

            # بررسی کد امنیتی بایت‌های فایل (file_sha256) در صورت وجود در فید
            expected_file_sha = entry.get("file_sha256")
            if expected_file_sha:
                actual_file_sha = hashlib.sha256(raw_bytes).hexdigest()
                if actual_file_sha != expected_file_sha:
                    raw_conn = _get_raw_conn(connection)
                    _quarantine_and_raise(
                        raw_conn,
                        entry.get("batch_id"),
                        actual_file_sha,
                        "file_sha256 mismatch on downloaded bytes",
                        raw_bytes.decode("utf-8", errors="replace"),
                    )

            receipt = ingest_batch(
                connection,
                raw_bytes,
                as_of=effective_as_of or entry["available_at"],
                expected_sha256=entry.get("payload_sha256"),
            )
            receipts.append(receipt)

        return receipts
