#!/usr/bin/env python3
"""Strict, idempotent source edits for PromiseOps golden/temporal refactor.

Run from the repository root. Uses anchored replacements and refuses unexpected
source layouts; no GitLab CI/GitHub settings or production resources are changed.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

MARKER = "PIT_GOLDEN_V3"


def replace_one(source: str, old: str, new: str, *, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one source anchor, got {source.count(old)}")
    return source.replace(old, new, 1)


def patch_data(s: str) -> str:
    s = replace_one(s,
        "    available_at TIMESTAMPTZ NOT NULL CHECK (available_at >= delivered_at)\n);\n\nCREATE TABLE IF NOT EXISTS batch_receipts",
        "    available_at TIMESTAMPTZ NOT NULL CHECK (available_at >= delivered_at),\n"
        "    batch_available_at TIMESTAMPTZ\n);\n"
        "-- Add the source receipt watermark to existing databases. Legacy rows\n"
        "-- stay NULL because their exact per-order batch is not recoverable.\n"
        "ALTER TABLE outcomes ADD COLUMN IF NOT EXISTS batch_available_at TIMESTAMPTZ;\n\n"
        "CREATE TABLE IF NOT EXISTS batch_receipts",label="outcome schema")
    s = replace_one(s,
        'INSERT INTO outcomes (order_id, delivered_at, available_at)\n                            VALUES (%s, %s, %s)\n                            """,\n                            (out.order_id, out.delivered_at, out.available_at),',
        'INSERT INTO outcomes (order_id, delivered_at, available_at, batch_available_at)\n'
        '                            VALUES (%s, %s, %s, %s)\n                            """,\n'
        '                            (out.order_id, out.delivered_at, out.available_at, batch.available_at),',
        label="outcome receipt persistence")
    # Point-in-time training asks for a feature-availability cutoff and a separate
    # label-maturity cutoff. Never inspect outcomes during serving.
    s = replace_one(s,
        '    connection, order_ids=None, *, as_of, labeled=False, ensure_schema: bool = True\n) -> pd.DataFrame:',
        '    connection, order_ids=None, *, as_of, labeled=False, ensure_schema: bool = True,\n'
        '    label_as_of: str | datetime | None = None,\n) -> pd.DataFrame:', label="feature API")
    s = replace_one(s,
        '    as_of_utc = instant(as_of)\n\n    # اگر لیست خاصی از order_idها داده شده',
        '    as_of_utc = instant(as_of)\n'
        '    label_cutoff_utc = instant(label_as_of) if label_as_of is not None else as_of_utc\n'
        '    if label_cutoff_utc < as_of_utc:\n'
        '        raise ValueError("label_as_of must not precede feature as_of")\n\n'
        '    # اگر لیست خاصی از order_idها داده شده',label="label cutoff validation")
    s = replace_one(s,
        '    # کوئری SQL ضد Fan-out با استفاده از CTE (جدول‌های موقت مستقل)\n    query = """',
        '    # PIT_GOLDEN_V3: serving avoids even joining the outcomes table.\n'
        '    # For legacy outcomes without per-order receipt provenance, use the\n'
        '    # latest known outcome receipt as a conservative availability bound.\n'
        '    outcome_cols = (\n'
        '        "out.delivered_at, "\n'
        '        "GREATEST(out.available_at, COALESCE(out.batch_available_at, "\n'
        '        "(SELECT MAX(available_at) FROM batch_receipts WHERE kind = \'outcomes\'))) "\n'
        '        "AS outcome_available_at"\n'
        '        if labeled else\n'
        '        "NULL::timestamptz AS delivered_at, NULL::timestamptz AS outcome_available_at"\n'
        '    )\n'
        '    outcome_join = (\n'
        '        "LEFT JOIN outcomes out ON t.order_id = out.order_id "\n'
        '        "AND GREATEST(out.available_at, COALESCE(out.batch_available_at, "\n'
        '        "(SELECT MAX(available_at) FROM batch_receipts WHERE kind = \'outcomes\'))) "\n'
        '        "<= %(label_as_of)s"\n'
        '        if labeled else ""\n'
        '    )\n'
        '    # SQL fragments above are static and NOT constructed from input.\n'
        '    query = """',label="PIT join options")
    s = replace_one(s,
        '        out.delivered_at,\n        out.available_at AS outcome_available_at\n',
        '        __OUTCOME_COLS__\n',label="outcome projection")
    s = replace_one(s,
        '    LEFT JOIN outcomes out\n      ON t.order_id = out.order_id\n     AND out.available_at <= %(as_of)s\n',
        '    __OUTCOME_JOIN__\n',label="outcome join")
    s = replace_one(s,
        '    with raw_conn.cursor() as cur:\n        cur.execute(query, {"as_of": as_of_utc, "order_ids": order_ids_list})\n        rows = cur.fetchall()',
        '    query = query.replace("__OUTCOME_COLS__", outcome_cols).replace("__OUTCOME_JOIN__", outcome_join)\n'
        '    params = {"as_of": as_of_utc, "order_ids": order_ids_list}\n'
        '    if labeled:\n'
        '        params["label_as_of"] = label_cutoff_utc\n'
        '    with raw_conn.cursor() as cur:\n'
        '        cur.execute(query, params)\n'
        '        rows = cur.fetchall()',label="query params and fragments")
    # The second read was unnecessary for newly inserted rows. RETURNING also
    # preserves the correct race resolution for same request_id competitors.
    s = replace_one(s,
        '                ON CONFLICT (request_id) DO NOTHING\n                """,\n                (request_id, order_id, risk_prob, model_ver),\n            )\n            # rowcount is 0 for an idempotency race: never persist metadata from\n            # the losing computation, whose prediction was not committed.\n            inserted = cur.rowcount == 1',
        '                ON CONFLICT (request_id) DO NOTHING\n'
        '                RETURNING order_id, risk_probability, model_version\n'
        '                """,\n                (request_id, order_id, risk_prob, model_ver),\n            )\n'
        '            # RETURNING is empty only for a conflict (possibly concurrent).\n'
        '            inserted_row = cur.fetchone()\n'
        '            inserted = inserted_row is not None',
        label="prediction returning")
    s = replace_one(s,
        '        raw_conn.commit()\n    except Exception:\n        raw_conn.rollback()\n        raise\n\n    # دوباره از دیتابیس می‌خوانیم',
        '        raw_conn.commit()\n'
        '    except Exception:\n        raw_conn.rollback()\n        raise\n\n'
        '    if inserted_row is not None:\n'
        '        return {\n'
        '            "order_id": inserted_row[0], "request_id": request_id,\n'
        '            "risk_probability": float(inserted_row[1]),\n'
        '            "model_version": inserted_row[2],\n'
        '        }\n\n'
        '    # دوباره از دیتابیس می‌خوانیم',
        label="return without reread")
    return s


def patch_model(s: str) -> str:
    old = '''        feature_rows = build_features(
            connection,
            order_ids,
            as_of=evaluation_time,
            labeled=True,
            ensure_schema=False,
        )
        validated_rows = validate_feature_rows(feature_rows, labeled=True)
        labeled_frame = pd.DataFrame([row.model_dump() for row in validated_rows])
'''
    new = '''        # PIT_GOLDEN_V3: no training features after train_end and no
        # validation features after validation_end. Labels may mature later,
        # but are never included in the feature vector.
        train_features = build_features(
            connection, order_ids, as_of=train_cutoff, label_as_of=evaluation_time,
            labeled=True, ensure_schema=False,
        )
        validation_features = build_features(
            connection, order_ids, as_of=validation_cutoff,
            label_as_of=evaluation_time, labeled=True, ensure_schema=False,
        )
        validation_features = validation_features.loc[
            validation_features["approved_at"] > train_cutoff
        ]
        feature_rows = pd.concat((train_features, validation_features), ignore_index=True)
        validated_rows = validate_feature_rows(feature_rows, labeled=True)
        labeled_frame = pd.DataFrame([row.model_dump() for row in validated_rows])
'''
    s=replace_one(s,old,new,label="train and validation PIT")
    s=replace_one(s, '        feature_aggregation_as_of=evaluation_time,',
                  '        feature_aggregation_as_of=train_cutoff,',label="training reference bound")
    s=replace_one(s, '        "test_policy": "untouched_not_used_in_training_or_selection",',
        '        "temporal_feature_policy": "train_at_train_end_validation_at_validation_end_labels_at_as_of",\n'
        '        "legacy_outcome_availability_policy": "latest_outcome_receipt_conservative_bound",\n'
        '        "test_policy": "untouched_not_used_in_training_or_selection",',label="manifest temporal policy")
    return s


def patch_fake(s: str) -> str:
    # Only tests that mock INSERT need to return a tuple for RETURNING.
    s=replace_one(s, '                self.db.pending_predictions[req] = (order, risk, version)\n                self.rowcount = 1',
        '                self.db.pending_predictions[req] = (order, risk, version)\n'
        '                self.one = (order, risk, version)  # INSERT ... RETURNING\n'
        '                self.rowcount = 1',label="fake returning")
    return s


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate all source anchors but do not write")
    args=parser.parse_args()
    changes={
        "src/promiseops/data.py":patch_data,
        "src/promiseops/model.py":patch_model,
        "tests/test_feature_persistence.py":patch_fake,
    }
    planned={}
    for path, fn in changes.items():
        target=Path(path)
        if not target.exists():
            raise SystemExit(f"Missing {path}. Run from PromiseOps repository root.")
        original=target.read_text(encoding="utf-8")
        if path.endswith("data.py") and MARKER in original:
            print(f"Already installed: {path}")
            continue
        if path.endswith("model.py") and MARKER in original:
            print(f"Already installed: {path}")
            continue
        if path.endswith("test_feature_persistence.py") and "# INSERT ... RETURNING" in original:
            print(f"Already installed: {path}")
            continue
        updated=fn(original)
        ast.parse(updated,filename=path)
        planned[target]=updated
    if args.check:
        for path in planned:
            print(f"Would update: {path}")
        print("Source anchors and syntax OK. No changes made.")
    else:
        for path,body in planned.items():
            path.write_text(body,encoding="utf-8")
            print(f"Updated: {path}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
