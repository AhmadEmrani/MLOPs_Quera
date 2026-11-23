#!/bin/sh
# Manual, bounded production load test. Requires CI runtime kubectl read access.
set -eu
set +x
umask 077

PROFILE="${PROMISEOPS_LOAD_PROFILE:-pilot}"
case "$PROFILE" in
  pilot)
    RPS=1
    SECONDS=10
    ;;
  acceptance)
    if [ "${PROMISEOPS_LOAD_WINDOW_APPROVED:-}" != "yes" ]; then
      echo "Acceptance requires PROMISEOPS_LOAD_WINDOW_APPROVED=yes after the agreed test window is approved." >&2
      exit 1
    fi
    RPS=10
    SECONDS=180
    ;;
  *)
    echo "PROMISEOPS_LOAD_PROFILE must be 'pilot' or 'acceptance'." >&2
    exit 1
    ;;
esac

TEAM_NAME="${TEAM:-g02}"
if [ "$TEAM_NAME" != "g02" ]; then
  echo "Refusing load test outside the configured g02 team." >&2
  exit 1
fi
NAMESPACE="${TEAM_NAME}-prod"
PYTHON_BIN="${PROMISEOPS_LOAD_PYTHON:-.load-venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Offline CI Python environment not found; run the job's uv setup first." >&2
  exit 1
fi

# These values are fixed to the team-approved historical post-validation window.
VALIDATION_END="${PROMISEOPS_LOAD_VALIDATION_END:-2018-04-30T23:59:59-03:00}"
AS_OF="${PROMISEOPS_LOAD_AS_OF:-2018-06-29T23:59:59-03:00}"
ORDER_LIMIT="${PROMISEOPS_LOAD_ORDER_LIMIT:-200}"
OUTPUT_DIR="reports/load"
mkdir -p "$OUTPUT_DIR"
PRIVATE_DIR="$(mktemp -d)"
trap 'rm -rf "$PRIVATE_DIR"' EXIT HUP INT TERM
ORDERS_FILE="$PRIVATE_DIR/order-ids.json"
SUMMARY_FILE="$OUTPUT_DIR/selection-${PROFILE}.json"
REPORT_FILE="$OUTPUT_DIR/load-${PROFILE}.json"
CONTEXT_FILE="$OUTPUT_DIR/context-${PROFILE}.txt"

# kubectl is used only to read the existing, staff-provided runtime Secret.
# Never echo, print or save credentials in an artifact.
API_KEY_ENCODED="$(kubectl get secret runtime-config -n "$NAMESPACE" -o jsonpath='{.data.PROMISEOPS_API_KEY}')"
DB_URL_ENCODED="$(kubectl get secret runtime-config -n "$NAMESPACE" -o jsonpath='{.data.PROMISEOPS_DATABASE_URL}')"
if [ -z "$API_KEY_ENCODED" ] || [ -z "$DB_URL_ENCODED" ]; then
  echo "runtime-config is missing the API key or database URL in $NAMESPACE." >&2
  exit 1
fi
PROMISEOPS_API_KEY="$(printf '%s' "$API_KEY_ENCODED" | base64 -d)"
PROMISEOPS_DATABASE_URL="$(printf '%s' "$DB_URL_ENCODED" | base64 -d)"
unset API_KEY_ENCODED DB_URL_ENCODED
export PROMISEOPS_API_KEY PROMISEOPS_DATABASE_URL

# Only aggregates and a cohort checksum go to artifacts; the private ID file
# stays in a temporary directory removed on exit (including a failed test).
"$PYTHON_BIN" ci/select_load_orders.py \
  --validation-end "$VALIDATION_END" --as-of "$AS_OF" \
  --limit "$ORDER_LIMIT" --minimum 20 \
  --order-ids "$ORDERS_FILE" --summary "$SUMMARY_FILE"
unset PROMISEOPS_DATABASE_URL

# Evidence of the actual running deployment, not merely the Git checkout.
DEPLOY_IMAGE="$(kubectl get deployment promiseops -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')"
if [ -z "$DEPLOY_IMAGE" ]; then
  echo "Cannot establish production deployment image identity." >&2
  exit 1
fi
{
  printf 'namespace=%s\n' "$NAMESPACE"
  printf 'profile=%s\n' "$PROFILE"
  printf 'target_rps=%s\n' "$RPS"
  printf 'test_seconds=%s\n' "$SECONDS"
  printf 'image=%s\n' "$DEPLOY_IMAGE"
  printf 'ci_commit=%s\n' "${CI_COMMIT_SHA:-unknown}"
  printf 'ci_pipeline_id=%s\n' "${CI_PIPELINE_ID:-local}"
  printf 'ci_job_id=%s\n' "${CI_JOB_ID:-local}"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$CONTEXT_FILE"

echo "Starting manually triggered $PROFILE load test: $RPS RPS for $SECONDS seconds (plus five warm-up requests)."
echo "Target: promiseops.$NAMESPACE.svc; selected IDs and credentials are not logged."
"$PYTHON_BIN" acceptance/load.py \
  --url "http://promiseops.${NAMESPACE}.svc:80" \
  --order-ids "$ORDERS_FILE" --rps "$RPS" --seconds "$SECONDS" \
  --output "$REPORT_FILE"
unset PROMISEOPS_API_KEY
