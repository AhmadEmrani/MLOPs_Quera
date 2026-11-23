#!/usr/bin/env python3
"""Read-only, aggregate inspection of PromiseOps monitoring readiness."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import sys



def instant(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('MONITOR_AS_OF requires a timezone offset')
    return value.astimezone(UTC)

def iso(value: datetime) -> str:
    return instant(value).isoformat().replace('+00:00', 'Z')


def normalize_dsn(dsn: str) -> str:
    if dsn.startswith('postgresql+psycopg://'):
        return 'postgresql://' + dsn[len('postgresql+psycopg://'):]
    if dsn.startswith('postgres://'):
        return 'postgresql://' + dsn[len('postgres://'):]
    return dsn


def model_label(version: str) -> str:
    # Immutable model hash is metadata, not a credential, but avoid dumping arbitrary values.
    if re.fullmatch(r'sha256:[a-f0-9]{64}', version):
        return version
    return 'non_sha256_model_identity'


def inspect(conn, cutoff):
    start = cutoff - timedelta(days=14)
    middle = cutoff - timedelta(days=7)
    result = {
        'schema_version': 1, 'inspection': 'monitoring_database_read_only',
        'as_of': iso(cutoff),
        'windows': [{'from': iso(start), 'until_exclusive': iso(middle)},
                    {'from': iso(middle), 'until_inclusive': iso(cutoff)}],
        'minimum_orders_per_window': 100,
    }
    with conn.cursor() as cur:
        cur.execute('SELECT to_regclass(%s), to_regclass(%s), to_regclass(%s)',
                    ('monitoring_predictions', 'promiseops_scoring_runs',
                     'prediction_feature_observations'))
        monitoring, scoring, feature_obs = cur.fetchone()
        result['tables'] = {'monitoring_predictions': monitoring is not None,
                            'promiseops_scoring_runs': scoring is not None,
                            'prediction_feature_observations': feature_obs is not None}
        if scoring is not None:
            cur.execute('SELECT count(*)::bigint, min(as_of), max(as_of) FROM promiseops_scoring_runs')
            count, first, last = cur.fetchone()
            result['historical_batch_runs'] = {'count': count, 'first_as_of': iso(first) if first else None,
                                               'last_as_of': iso(last) if last else None}
        if feature_obs is not None:
            cur.execute('SELECT count(*)::bigint FROM prediction_feature_observations')
            result['api_feature_observations'] = cur.fetchone()[0]
        if monitoring is None:
            result['status'] = 'migration_004_missing'
            return result
        cur.execute('SELECT count(*)::bigint, min(scored_as_of), max(scored_as_of) '
                    'FROM monitoring_predictions')
        count, first, last = cur.fetchone()
        result['monitoring_predictions'] = {'count': count, 'first_scored_as_of': iso(first) if first else None,
                                            'last_scored_as_of': iso(last) if last else None}
        if not count:
            result['status'] = 'no_monitoring_predictions'
            return result
        cur.execute('''
            SELECT model_version,
                count(DISTINCT order_id) FILTER (
                    WHERE scored_as_of >= %s AND scored_as_of < %s) AS first_window,
                count(DISTINCT order_id) FILTER (
                    WHERE scored_as_of >= %s AND scored_as_of <= %s) AS second_window
            FROM monitoring_predictions
            WHERE scored_as_of >= %s AND scored_as_of <= %s
            GROUP BY model_version
            ORDER BY (count(DISTINCT order_id)) DESC
            LIMIT 10
        ''', (start, middle, middle, cutoff, start, cutoff))
        cohorts = [{'model_version': model_label(v), 'window1_unique_orders': a,
                    'window2_unique_orders': b, 'two_windows_sufficient': a >= 100 and b >= 100}
                   for v, a, b in cur.fetchall()]
        result['model_window_cohorts'] = cohorts
        result['status'] = ('drift_windows_sufficient' if any(c['two_windows_sufficient'] for c in cohorts)
                            else 'insufficient_scored_orders_in_14_day_windows')
        result['note'] = 'This checks only Drift cohort sizes; label maturity and model/reference identity must be verified separately.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of', required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/monitoring/db-readiness.json'))
    args = parser.parse_args(argv)
    try:
        cutoff = instant(args.as_of)
        dsn = os.environ.get('PROMISEOPS_DATABASE_URL', '')
        if not dsn:
            raise ValueError('DB credentials were not provided by the authorized CI secret')
        import psycopg
        with psycopg.connect(normalize_dsn(dsn), connect_timeout=10,
                             options='-c default_transaction_read_only=on -c statement_timeout=30000') as conn:
            report = inspect(conn, cutoff)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print('Monitoring DB check: status=%s migration_004=%s batch_runs=%s monitoring_rows=%s' % (
            report['status'], report['tables']['monitoring_predictions'],
            report.get('historical_batch_runs', {}).get('count', 'unknown'),
            report.get('monitoring_predictions', {}).get('count', 'unknown')))
        return 0
    except Exception as error:
        # DB transports can embed credentials in error strings. Only print exception class.
        print('Monitoring DB inspection failed: ' + type(error).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
