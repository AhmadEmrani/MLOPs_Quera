#!/usr/bin/env python3
"""Apply the T1-T6 monitoring integration to the observed PromiseOps branch.

Safe by default: verify all anchors before writing; reruns are idempotent.
Does not connect to Kubernetes, databases, Airflow, GitLab or MLflow.
"""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(value: str, old: str, new: str, path: str) -> str:
    if new in value:
        return value
    if value.count(old) != 1:
        raise RuntimeError(f"Unrecognized source at {path}; refusing to overwrite (anchor count={value.count(old)})")
    return value.replace(old, new, 1)


def generate() -> dict[Path, str]:
    target: dict[Path, str] = {}
    file = ROOT / "src/promiseops/cli.py"
    data = file.read_text(encoding="utf-8")
    data = replace_once(data, "def score_batch(\n    conn: Any, model: Any, release: dict, *, as_of: datetime, since: datetime, batch_id: str\n) -> dict:",
        "def score_batch(\n    conn: Any, model: Any, release: dict, *, as_of: datetime, since: datetime, batch_id: str,\n    monitoring_rows: list[dict] | None = None,\n) -> dict:", str(file))
    data = replace_once(data, '        for order_id, row in zip(frame["order_id"], probs, strict=True):\n',
        '        for offset, (order_id, row) in enumerate(zip(frame["order_id"], probs, strict=True)):\n', str(file))
    data = replace_once(data, "            scored.append((str(order_id), prob))\n",
        '''            scored.append((str(order_id), prob))
            if monitoring_rows is not None:
                from promiseops.feature_lineage import observation
                snapshot = observation(frame.iloc[[offset]], feature_as_of=as_of, release=release)
                monitoring_rows.append({
                    "batch_id": batch_id,
                    "order_id": str(order_id),
                    "model_version": version,
                    "risk_probability": prob,
                    "scored_as_of": as_of,
                    "feature_contract_sha256": snapshot["feature_contract_sha256"],
                    "feature_hash": snapshot["feature_hash"],
                    "features": snapshot["features"],
                })
''', str(file))
    target[file] = data

    file = ROOT / "src/promiseops/scheduled_batch.py"
    data = file.read_text(encoding="utf-8")
    data = replace_once(data, '''                result = score_batch(
                    conn,
                    model,
                    release,
                    as_of=as_of,
                    since=since,
                    batch_id=batch_id,
                )''', '''                monitoring_rows: list[dict] = []
                result = score_batch(
                    conn,
                    model,
                    release,
                    as_of=as_of,
                    since=since,
                    batch_id=batch_id,
                    monitoring_rows=monitoring_rows,
                )''', str(file))
    data = replace_once(data, '''                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO promiseops_scoring_runs(batch_id, since_at, as_of, result)
''', '''                with conn.cursor() as cur:
                    # One transaction for *all* probabilities and cursor advance.
                    # Explicit migration 004 must exist before running this CronJob.
                    cur.executemany(
                        """
                        INSERT INTO monitoring_predictions (
                            batch_id, order_id, model_version, risk_probability,
                            scored_as_of, feature_contract_sha256, feature_hash, features_jsonb
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (batch_id, order_id, model_version) DO NOTHING
                        """,
                        [(
                            row["batch_id"], row["order_id"], row["model_version"],
                            row["risk_probability"], row["scored_as_of"],
                            row["feature_contract_sha256"], row["feature_hash"],
                            json.dumps(row["features"], sort_keys=True,
                                       ensure_ascii=False, allow_nan=False),
                        ) for row in monitoring_rows],
                    )
                    cur.execute(
                        """
                        INSERT INTO promiseops_scoring_runs(batch_id, since_at, as_of, result)
''', str(file))
    data = replace_once(data, '                encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))', '                if len(monitoring_rows) != int(result["scored_count"]):\n                    raise ValueError("Monitoring cohort count does not match scored_count")\n                encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))', str(file))
    target[file] = data

    file = ROOT / ".gitlab-ci.yml"
    data = file.read_text(encoding="utf-8")
    data = replace_once(data, '  - local: "/.gitlab/ci/load.yml"\n',
        '  - local: "/.gitlab/ci/load.yml"\n  - local: "/.gitlab/ci/monitoring.yml"\n', str(file))
    target[file] = data

    file = ROOT / "src/promiseops/monitor.py"
    data = file.read_text(encoding="utf-8")
    source_stub = '''def build_report(predictions, released_outcomes, reference, *, as_of):
    """Use unique orders, separate drift/maturity windows, minimum samples and reasons."""
    raise NotImplementedError("Implement maturity-aware monitoring")
'''
    facade = '''"""Compatibility facade: implementation lives in promiseops.monitoring."""
from promiseops.monitoring.service import build_report
__all__ = ["build_report"]
'''
    if data.strip() not in (source_stub.strip(), facade.strip()):
        raise RuntimeError("Unexpected monitor.py changes; refusing to replace it")
    target[file] = facade
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = generate()
    changed = [f for f, content in target.items() if f.read_text(encoding="utf-8") != content]
    if args.check:
        print("Monitoring patch anchors OK. Files requiring modification:", len(changed))
        return 0
    for file in changed:
        file.write_text(target[file], encoding="utf-8")
        print("Patched", file.relative_to(ROOT))
    print("Active Learning T1-T6 integration applied. No production actions performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
