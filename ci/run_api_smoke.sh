#!/bin/sh
set -eu
set +x

NAMESPACE="${1:?namespace argument required}"
API_SERVICE="${PROMISEOPS_API_SERVICE:-promiseops}"
ORDER_ID="${PROMISEOPS_SMOKE_ORDER_ID:-}"
API_KEY_B64="$(kubectl get secret runtime-config -n "$NAMESPACE" \
  -o jsonpath='{.data.PROMISEOPS_API_KEY}')"
if [ -z "$API_KEY_B64" ]; then
  echo "runtime-config does not contain PROMISEOPS_API_KEY in ${NAMESPACE}" >&2
  exit 1
fi

PROMISEOPS_API_KEY="$(printf '%s' "$API_KEY_B64" | base64 -d)"
unset API_KEY_B64
PROMISEOPS_API_URL="http://${API_SERVICE}.${NAMESPACE}.svc:80"
export PROMISEOPS_API_KEY PROMISEOPS_API_URL PROMISEOPS_SMOKE_ORDER_ID="$ORDER_ID"
python ci/smoke_api.py
unset PROMISEOPS_API_KEY
