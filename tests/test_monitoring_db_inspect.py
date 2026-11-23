from datetime import datetime
from ci.inspect_monitoring_db import inspect, normalize_dsn, model_label


class Cursor:
    def __init__(self, status):
        self.status = status
        self.queries = []
    def execute(self, query, params=None):
        self.queries.append((query, params))
    def fetchone(self):
        if len(self.queries) == 1:
            return ((None if self.status == 'missing' else 'monitoring_predictions'),
                    'promiseops_scoring_runs', None)
        if len(self.queries) == 2:
            return (10, None, None)
        if len(self.queries) == 3:
            return (0, None, None)
        raise AssertionError('Unexpected query')
    def __enter__(self): return self
    def __exit__(self, *_): pass

class Connection:
    def __init__(self, status): self.cursor_obj = Cursor(status)
    def cursor(self): return self.cursor_obj


def test_missing_table_is_reported_without_attempting_mutation():
    c = Connection('missing')
    result = inspect(c, datetime.fromisoformat('2018-06-29T23:59:59-03:00'))
    assert result['status'] == 'migration_004_missing'
    assert all('SELECT' in query for query, _ in c.cursor_obj.queries)


def test_empty_table_is_reported():
    c = Connection('exists')
    result = inspect(c, datetime.fromisoformat('2018-06-29T23:59:59-03:00'))
    assert result['status'] == 'no_monitoring_predictions'
    assert result['monitoring_predictions']['count'] == 0


def test_safe_dsn_and_model_label():
    assert normalize_dsn('postgresql+psycopg://host/db') == 'postgresql://host/db'
    assert normalize_dsn('postgres://host/db') == 'postgresql://host/db'
    assert model_label('secret') == 'non_sha256_model_identity'
