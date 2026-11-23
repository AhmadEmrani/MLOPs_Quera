#!/bin/sh
set -eu
set +x

NAMESPACE="${1:?namespace argument required}"
DIGEST="${2:?image digest argument required}"
RUNTIME_SA="${3:-${PROMISEOPS_RUNTIME_SERVICE_ACCOUNT:-promiseops-api}}"
IMAGE_REPOSITORY="${CI_REGISTRY_IMAGE:?CI_REGISTRY_IMAGE is required}/app"
RELEASE_NAME="promiseops"
CHART="deploy/chart"
VALUES="deploy/chart/values.prod.yaml"
CRONJOB="${RELEASE_NAME}-batch"

python - "$DIGEST" <<'PYDIGEST'
import re
import sys

if re.fullmatch(r"sha256:[0-9a-f]{64}", sys.argv[1]) is None:
    raise SystemExit("production image digest must be sha256:<64 lowercase hex>")
if sys.argv[1] == "sha256:" + "0" * 64:
    raise SystemExit("the zero digest is not a deployable image")
PYDIGEST
case "$RUNTIME_SA" in *[!A-Za-z0-9._-]*|'') echo "Invalid runtime service account name" >&2; exit 1 ;; esac
# Runtime ServiceAccount is staff-managed; CI has no get permission.
# Keep the configured name; Kubernetes validates it when creating Pods.
kubectl get secret registry-pull runtime-config -n "$NAMESPACE" -o name >/dev/null

HELD_CRONJOB=0
cleanup() {
  result=$?
  trap - EXIT HUP INT TERM
  if [ "$HELD_CRONJOB" = 1 ]; then
    kubectl patch cronjob "$CRONJOB" -n "$NAMESPACE" \
      --type=merge -p '{"spec":{"suspend":false}}' >/dev/null 2>&1 || true
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Suspend scheduling first, then wait for both running and pending scoring Jobs.
# This closes the race between an idle check and Helm creating the surge Pod.
if [ "$(kubectl auth can-i get cronjobs -n "$NAMESPACE")" != "yes" ] \
  || [ "$(kubectl auth can-i patch cronjobs -n "$NAMESPACE")" != "yes" ]; then
  echo "CI identity cannot inspect and suspend CronJobs in ${NAMESPACE}; refusing production rollout." >&2
  exit 1
fi
HELD_CRONJOB=1
if kubectl get cronjob "$CRONJOB" -n "$NAMESPACE" >/dev/null 2>&1; then
  kubectl patch cronjob "$CRONJOB" -n "$NAMESPACE" \
    --type=merge -p '{"spec":{"suspend":true}}'
fi
sh ci/wait_for_scoring_idle.sh "$NAMESPACE"

python ci/helm_secret_budget.py --namespace "$NAMESPACE" --quota-fallback "${PROMISEOPS_SECRET_QUOTA_FALLBACK:-20}"
helm upgrade --install "$RELEASE_NAME" "$CHART" \
  --namespace "$NAMESPACE" \
  --values "$VALUES" \
  --set-string "image.repository=${IMAGE_REPOSITORY}" \
  --set-string "image.digest=${DIGEST}" \
  --set-string "serviceAccount.name=${RUNTIME_SA}" \
  --set batchJob.suspend=true \
  --atomic --wait --timeout 8m --history-max 5

kubectl rollout status "deployment/${RELEASE_NAME}" -n "$NAMESPACE" --timeout=8m
ATTEMPT=0
AVAILABLE=0
while [ "$ATTEMPT" -lt 60 ]; do
  AVAILABLE="$(kubectl get deployment "$RELEASE_NAME" -n "$NAMESPACE" \
    -o jsonpath='{.status.availableReplicas}')"
  AVAILABLE="${AVAILABLE:-0}"
  [ "$AVAILABLE" -ge 2 ] && break
  ATTEMPT=$((ATTEMPT + 1))
  sleep 5
done
[ "$AVAILABLE" -ge 2 ] || { echo "Production did not reach two available replicas" >&2; exit 1; }

# Persist the normal unsuspended schedule as the current Helm revision. This
# changes only the CronJob and does not roll API Pods or consume surge capacity.
python ci/helm_secret_budget.py --namespace "$NAMESPACE" --quota-fallback "${PROMISEOPS_SECRET_QUOTA_FALLBACK:-20}"
helm upgrade "$RELEASE_NAME" "$CHART" \
  --namespace "$NAMESPACE" \
  --values "$VALUES" \
  --set-string "image.repository=${IMAGE_REPOSITORY}" \
  --set-string "image.digest=${DIGEST}" \
  --set-string "serviceAccount.name=${RUNTIME_SA}" \
  --set batchJob.suspend=false \
  --atomic --wait --timeout 8m --history-max 5
HELD_CRONJOB=0
trap - EXIT
trap - HUP INT TERM

echo "Production release ready: ${IMAGE_REPOSITORY}@${DIGEST}; available replicas=${AVAILABLE}."
