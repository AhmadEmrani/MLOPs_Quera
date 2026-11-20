#!/bin/sh
set -eu

NAMESPACE="${1:?namespace argument required}"
MAX_WAIT_SECONDS="${MAX_SCORING_WAIT_SECONDS:-3300}"
WAITED=0

while :; do
  JOBS_JSON="$(kubectl get jobs -n "$NAMESPACE" -l promiseops.io/workload=batch-score -o json)"
  ACTIVE_JOBS="$(printf '%s' "$JOBS_JSON" | python -c '
import json, sys
items = json.load(sys.stdin).get("items", [])
for job in items:
    status = job.get("status") or {}
    terminal = any(
        c.get("status") == "True" and c.get("type") in {"Complete", "Failed"}
        for c in status.get("conditions", [])
    )
    if not terminal:
        print(job["metadata"]["name"])
')"
  if [ -z "$ACTIVE_JOBS" ]; then
    echo "No active scoring job in ${NAMESPACE}."
    exit 0
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_SECONDS" ]; then
    echo "Scoring is still active after ${WAITED}s; refusing production rollout." >&2
    exit 1
  fi
  echo "Waiting for active scoring job(s) in ${NAMESPACE}: $(printf '%s' "$ACTIVE_JOBS" | tr '\n' ' ')"
  sleep 10
  WAITED=$((WAITED + 10))
done
