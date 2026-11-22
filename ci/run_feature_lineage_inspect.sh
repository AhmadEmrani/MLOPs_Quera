#!/bin/sh
set -eu
set +x

TEAM_NAME="${TEAM:-g02}"
case "$TEAM_NAME" in g0[1-7]) ;; *) echo "Invalid TEAM namespace" >&2; exit 1 ;; esac
NAMESPACE="${TEAM_NAME}-prod"

# Read existing staff-managed secret only; never print or archive it.
DB_URL_B64="$(kubectl get secret runtime-config -n "$NAMESPACE" \
    -o jsonpath='{.data.PROMISEOPS_DATABASE_URL}')"
if [ -z "$DB_URL_B64" ]; then
    echo "Production runtime-config lacks PROMISEOPS_DATABASE_URL" >&2
    exit 1
fi
PROMISEOPS_DATABASE_URL="$(printf '%s' "$DB_URL_B64" | base64 -d)"
unset DB_URL_B64
export PROMISEOPS_DATABASE_URL

mkdir -p reports
.load-venv/bin/python ci/check_feature_lineage.py \
    --hours "${PROMISEOPS_FEATURE_INSPECT_HOURS:-24}" \
    --output reports/feature-lineage.json
unset PROMISEOPS_DATABASE_URL
