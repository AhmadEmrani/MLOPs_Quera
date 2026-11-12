import copy
import json
import os
from pathlib import Path

import psycopg
import pytest

from promiseops.contracts import Batch, PredictionRequest, digest
from promiseops.data import (
    ConflictError,
    build_features,
    get_or_save_prediction,
    ingest_batch,
    init_db,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def db_conn():
    """اتصال به دیتابیس تست و پاکسازی جدول‌ها قبل از هر تست برای ایزوله بودن کامل."""
    urls = [
        os.environ.get("PROMISEOPS_DATABASE_URL"),
        "postgresql://postgres:postgres@localhost:5433/promiseops",
        "postgresql://postgres:postgres@localhost:5432/promiseops",
    ]
    conn = None
    for url in urls:
        if not url:
            continue
        clean_url = url.replace("postgresql+psycopg://", "postgresql://")
        try:
            conn = psycopg.connect(clean_url, connect_timeout=3)
            break
        except psycopg.Error:
            continue

    if conn is None:
        pytest.skip("PostgreSQL database is not reachable for integration tests")

    init_db(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            TRUNCATE TABLE
                predictions,
                outcomes,
                payments,
                items,
                orders,
                products,
                sellers,
                customers,
                batch_receipts,
                batch_quarantine
            RESTART IDENTITY CASCADE
            """
        )
    conn.commit()
    yield conn
    conn.close()


def test_supplied_fixtures_follow_contract():
    for path in (ROOT / "fixtures").glob("synthetic-*.json"):
        Batch.model_validate(json.loads(path.read_text()))


def test_prediction_ids_cannot_be_paths():
    with pytest.raises(ValueError):
        PredictionRequest(order_id="../../secret")


def test_ingest_idempotency_and_tamper_quarantine(db_conn):
    orders_data = json.loads((ROOT / "fixtures" / "synthetic-orders.json").read_text())

    # 1. ورود بار اول باید committed شود
    r1 = ingest_batch(db_conn, orders_data, as_of="2018-01-01T00:00:00Z", expected_sha256=digest(orders_data))
    assert r1["status"] == "committed"

    # 2. ورود بار دوم با همان داده باید noop شود و ردیف اضافه نسازد
    r2 = ingest_batch(db_conn, orders_data, as_of="2018-01-01T00:00:00Z", expected_sha256=digest(orders_data))
    assert r2["status"] == "noop"

    # 3. ارسال همان batch_id با محتوای دستکاری‌شده باید خطا بدهد و وارد قرنطینه شود
    tampered = copy.deepcopy(orders_data)
    tampered["customers"][0]["customer_state"] = "MG"
    with pytest.raises(ValueError, match="quarantined"):
        ingest_batch(db_conn, tampered, as_of="2018-01-01T00:00:00Z", expected_sha256=digest(tampered))

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM batch_quarantine WHERE batch_id = %s", (orders_data["batch_id"],))
        assert cur.fetchone()[0] == 1


def test_ingest_atomic_rollback_on_missing_fk(db_conn):
    # ساخت یک بسته که سفارش آن به مشتری ناموجود اشاره می‌کند
    bad_batch = {
        "schema_version": "1.0",
        "batch_id": "bad-fk-batch-001",
        "kind": "orders",
        "available_at": "2017-04-15T00:00:00+00:00",
        "customers": [{"customer_id": "c_valid", "customer_state": "SP"}],
        "sellers": [{"seller_id": "s_valid", "seller_state": "RJ"}],
        "products": [{"product_id": "p_valid", "category": "books", "weight_g": 500.0}],
        "orders": [
            {
                "order_id": "o_broken",
                "customer_id": "c_missing_fk",  # این مشتری وجود ندارد!
                "approved_at": "2017-04-10T10:00:00+00:00",
                "promised_date": "2017-04-20",
            }
        ],
        "items": [
            {
                "order_id": "o_broken",
                "item_id": 1,
                "product_id": "p_valid",
                "seller_id": "s_valid",
                "price": 100.0,
                "freight": 10.0,
            }
        ],
        "payments": [
            {
                "order_id": "o_broken",
                "sequence": 1,
                "payment_type": "credit_card",
                "installments": 1,
                "value": 110.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="quarantined"):
        ingest_batch(db_conn, bad_batch, as_of="2018-01-01T00:00:00Z", expected_sha256=digest(bad_batch))

    # بررسی اینکه تراکنش کامل Rollback شده و حتی مشتری اول هم در جدول customers نمانده است
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customers")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM batch_quarantine WHERE batch_id = 'bad-fk-batch-001'")
        assert cur.fetchone()[0] == 1


def test_build_features_no_fanout_and_leakage_safe(db_conn):
    # ساخت یک سفارش با ۲ قلم کالا (جمع قیمت: 150) و ۳ پرداخت (جمع پرداخت: 180)
    multi_batch = {
        "schema_version": "1.0",
        "batch_id": "fanout-check-orders",
        "kind": "orders",
        "available_at": "2017-06-01T12:00:00+00:00",
        "customers": [{"customer_id": "c1", "customer_state": "SP"}],
        "sellers": [
            {"seller_id": "s1", "seller_state": "SP"},
            {"seller_id": "s2", "seller_state": "RJ"},
        ],
        "products": [
            {"product_id": "p1", "category": "electronics", "weight_g": 200.0},
            {"product_id": "p2", "category": "toys", "weight_g": 300.0},
        ],
        "orders": [
            {
                "order_id": "ord_fanout",
                "customer_id": "c1",
                "approved_at": "2017-06-01T10:00:00+00:00",
                "promised_date": "2017-06-10",
            }
        ],
        "items": [
            {
                "order_id": "ord_fanout",
                "item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "price": 50.0,
                "freight": 10.0,
            },
            {
                "order_id": "ord_fanout",
                "item_id": 2,
                "product_id": "p2",
                "seller_id": "s2",
                "price": 100.0,
                "freight": 20.0,
            },
        ],
        "payments": [
            {
                "order_id": "ord_fanout",
                "sequence": 1,
                "payment_type": "voucher",
                "installments": 1,
                "value": 30.0,
            },
            {
                "order_id": "ord_fanout",
                "sequence": 2,
                "payment_type": "voucher",
                "installments": 1,
                "value": 50.0,
            },
            {
                "order_id": "ord_fanout",
                "sequence": 3,
                "payment_type": "credit_card",
                "installments": 4,
                "value": 100.0,
            },
        ],
    }
    outcome_batch = {
        "schema_version": "1.0",
        "batch_id": "fanout-check-outcomes",
        "kind": "outcomes",
        "available_at": "2017-06-20T00:00:00+00:00",
        "outcomes": [
            {
                "order_id": "ord_fanout",
                "delivered_at": "2017-06-12T15:00:00+00:00",  # دیرتر از 2017-06-10 -> late = 1
                "available_at": "2017-06-19T15:00:00+00:00",
            }
        ],
    }

    ingest_batch(db_conn, multi_batch, as_of="2017-07-01T00:00:00Z", expected_sha256=digest(multi_batch))
    ingest_batch(db_conn, outcome_batch, as_of="2017-07-01T00:00:00Z", expected_sha256=digest(outcome_batch))

    # الف) بررسی عدم چندبرابر شدن اعداد (No Fan-out)
    df = build_features(db_conn, ["ord_fanout"], as_of="2017-06-05T00:00:00Z", labeled=False)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["item_count"] == 2
    assert row["seller_count"] == 2
    assert row["total_price"] == 150.0  # اگر Fan-out می‌شد، 3 برابر یعنی 450 می‌شد!
    assert row["total_freight"] == 30.0
    assert row["total_weight_g"] == 500.0
    assert row["payment_total"] == 180.0  # اگر Fan-out می‌شد، 2 برابر یعنی 360 می‌شد!
    assert row["max_installments"] == 4
    assert row["interstate_fraction"] == 0.5
    assert row["primary_category"] == "toys"  # چون قیمتش 100 است و از 50 بیشتر است
    assert row["primary_payment_type"] == "credit_card"  # چون مبلغش 100 است

    # ب) بررسی جلوگیری از نشت داده زمانی (در تاریخ 15 ژوئن هنوز برچسب که 19 ژوئن می‌آید در دسترس نیست!)
    df_early = build_features(db_conn, None, as_of="2017-06-15T00:00:00Z", labeled=True)
    assert df_early.empty

    # ج) در تاریخ 20 ژوئن برچسب در دسترس است و باید late == 1 باشد
    df_mature = build_features(db_conn, None, as_of="2017-06-20T00:00:00Z", labeled=True)
    assert len(df_mature) == 1
    assert df_mature.iloc[0]["late"] == 1


def test_prediction_idempotency_and_conflict(db_conn):
    orders_data = json.loads((ROOT / "fixtures" / "synthetic-orders.json").read_text())
    ingest_batch(db_conn, orders_data, as_of="2018-01-01T00:00:00Z", expected_sha256=digest(orders_data))

    r1 = get_or_save_prediction(
        db_conn,
        request_id="req-100",
        order_id="o000000",
        compute_fn=lambda: {"risk_probability": 0.25, "model_version": "v1"},
    )
    r2 = get_or_save_prediction(
        db_conn,
        request_id="req-100",
        order_id="o000000",
        compute_fn=lambda: {"risk_probability": 0.85, "model_version": "v2"},
    )
    assert r1 == r2
    assert r2["model_version"] == "v1"

    with pytest.raises(ConflictError):
        get_or_save_prediction(
            db_conn,
            request_id="req-100",
            order_id="o000001",
            compute_fn=lambda: {"risk_probability": 0.5, "model_version": "v1"},
        )
