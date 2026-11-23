"""Regression tests for fewer reads and atomic snapshot consistency.

Reuses the isolated, transaction-aware DB fake from the feature persistence suite.
"""
from test_feature_persistence import FakeDB, predict


def test_fresh_insert_needs_no_post_commit_select():
    db = FakeDB()
    reads = []
    original = db.cursor

    class CountingCursor:
        def __enter__(self):
            self.cur = original().__enter__()
            return self
        def __exit__(self, *args):
            return self.cur.__exit__(*args)
        def execute(self, sql, params=None):
            if "SELECT order_id, risk_probability, model_version FROM predictions" in sql:
                reads.append(params[0])
            return self.cur.execute(sql, params)
        def fetchone(self):
            return self.cur.fetchone()
        @property
        def rowcount(self):
            return self.cur.rowcount

    db.cursor=lambda: CountingCursor()
    r=predict(db)
    assert r["risk_probability"] == 0.35
    assert reads == ["r1"]  # before INSERT only; RETURNING replaces second SELECT
    assert db.commits == 1
    assert len(db.predictions) == len(db.observations) == 1


def test_atomic_rollback_and_recovery_after_failure():
    db = FakeDB()
    db.fail_observation = True
    try:
        predict(db)
        raise AssertionError("Expected snapshot write failure")
    except RuntimeError:
        pass
    assert not db.predictions and not db.observations
    db.fail_observation = False
    assert predict(db)["order_id"] == "o1"
    assert len(db.predictions) == len(db.observations) == 1
    assert db.rollbacks == 1 and db.commits == 1
