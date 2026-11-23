#!/bin/sh
# Offline, no unverified dependency downloads and no credential prints.
set -eu
set +x
. ci/require_mlflow_ci_env.sh
test -d /opt/promiseops/tracking-wheelhouse || {
  echo 'Missing authorized tracking-wheelhouse in CI toolbox.' >&2
  exit 1
}
cp -a /opt/promiseops/tracking-wheelhouse ./tracking-wheelhouse
uv venv --python /usr/local/bin/python .registry-venv
uv pip sync --python .registry-venv/bin/python --no-index \
  --find-links tracking-wheelhouse --require-hashes requirements-tracking.lock
