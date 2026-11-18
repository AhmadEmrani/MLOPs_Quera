
"""Build Airflow DAG files and the PromiseOps ZIP offline."""

import ast
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("src")
package = root / "promiseops"
output = Path("dist/airflow")
output.mkdir(parents=True, exist_ok=True)

assert (package / "__init__.py").is_file(), (
    "src/promiseops/__init__.py not found"
)

for name in ("ingestion.py", "training.py", "config_probe.py"):
    source = Path("dags") / name

    ast.parse(
        source.read_text(encoding="utf-8"),
        filename=str(source),
    )

    shutil.copy2(source, output / name)
    print(f"DAG syntax OK: {source}")

archive_path = output / "promiseops.zip"

with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue

        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue

        if path.suffix == ".py":
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )

        archive.write(path, path.relative_to(root))

with ZipFile(archive_path) as archive:
    files = set(archive.namelist())

    for expected in (
        "promiseops/__init__.py",
        "promiseops/contracts.py",
        "promiseops/data.py",
    ):
        assert expected in files, (
            f"Missing {expected} in Airflow ZIP"
        )

    assert archive.testzip() is None

size = archive_path.stat().st_size

assert size < 180_000, (
    f"ZIP is {size} bytes; exceeds conservative CI limit. "
    "Review package contents and ConfigMap delivery strategy."
)

sys.path.insert(0, str(archive_path.resolve()))

spec = find_spec("promiseops")

assert spec is not None and spec.origin is not None
assert str(archive_path.resolve()) in spec.origin, spec.origin

print(f"Package location OK: {spec.origin}")
print(f"ZIP size: {size} bytes")
