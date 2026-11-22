from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'ci/check_feature_lineage.py'
spec = spec_from_file_location('feature_lineage_inspect', SCRIPT)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_connection_urls_normalize_without_touching_credentials():
    assert module.normalize_dsn('postgresql+psycopg://user:secret@example.local:5432/data') == \
           'postgresql://user:secret@example.local:5432/data'
    assert module.normalize_dsn('postgres://local/db') == 'postgresql://local/db'
    assert module.normalize_dsn('postgresql://local/db') == 'postgresql://local/db'
