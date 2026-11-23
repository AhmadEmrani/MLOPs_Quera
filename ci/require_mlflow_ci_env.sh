#!/bin/sh
# Source from GitLab CI before using MLflow. Never print URI values or credentials.
# Gateway-provisioned training Worker env vars are NOT inherited by CI Runner.
set +x

_promiseops_mlflow_ci_error() {
  printf '%s\n' 'MLflow CI access is not configured for this job.' >&2
  printf '%s\n' 'The training gateway injects MLflow settings into worker Pods, NOT GitLab CI jobs.' >&2
  printf '%s\n' 'Check GitLab variables or existing authorized g02-prod/runtime-config MLflow Secret.' >&2
  printf '%s\n' "Use Environment scope '*'. Protected variables require the main branch to be protected." >&2
  printf 'CI_COMMIT_REF_PROTECTED=%s\n' "${CI_COMMIT_REF_PROTECTED:-unknown}" >&2
  printf '%s\n' 'Check protected status, environment scope and permissions. Do not print secret values.' >&2
  return 1
}

_promiseops_mlflow_ci_uri="${MLFLOW_TRACKING_URI:-}"
_promiseops_mlflow_ci_source=standard
if [ -z "$_promiseops_mlflow_ci_uri" ] && [ -n "${PROMISEOPS_MLFLOW_TRACKING_URI:-}" ]; then
  _promiseops_mlflow_ci_uri="$PROMISEOPS_MLFLOW_TRACKING_URI"
  _promiseops_mlflow_ci_source=alternative
fi
if [ -z "$_promiseops_mlflow_ci_uri" ] && [ -n "${MLFLOW_TRACKING_URI_FILE:-}" ]; then
  _promiseops_mlflow_ci_uri="$MLFLOW_TRACKING_URI_FILE"
  _promiseops_mlflow_ci_source=file_path
fi

# A GitLab File-type CI variable contains a path to a temporary file.
if [ -n "$_promiseops_mlflow_ci_uri" ] && [ -f "$_promiseops_mlflow_ci_uri" ]; then
  if [ ! -r "$_promiseops_mlflow_ci_uri" ]; then
    _promiseops_mlflow_ci_error
    return 1
  fi
  _promiseops_mlflow_ci_uri="$(cat "$_promiseops_mlflow_ci_uri")" || return 1
  _promiseops_mlflow_ci_source=file
fi

# The team platform already supplies this URI in the existing runtime-config
# Kubernetes Secret. GitLab CI does not automatically inherit that env value.
# Use the SAME g02-prod registry endpoint for Golden, release, and promotion.
# Only the currently authorized CI Kubernetes identity can read this Secret.
if [ -z "$_promiseops_mlflow_ci_uri" ]; then
  _promiseops_mlflow_secret_namespace="${PROMISEOPS_MLFLOW_SECRET_NAMESPACE:-g02-prod}"
  case "$_promiseops_mlflow_secret_namespace" in
    g02-prod|g02-staging) ;;
    *)
      printf '%s\n' 'MLflow Secret namespace must be g02-prod or g02-staging.' >&2
      return 1 ;;
  esac
  if ! command -v kubectl >/dev/null 2>&1; then
    printf '%s\n' 'kubectl is unavailable in this GitLab CI job.' >&2
    return 1
  fi
  if ! kubectl auth can-i get secret/runtime-config \
      -n "$_promiseops_mlflow_secret_namespace" --quiet >/dev/null 2>&1; then
    printf '%s\n' 'CI identity cannot read the authorized MLflow runtime-config Secret.' >&2
    return 1
  fi
  if ! _promiseops_mlflow_ci_encoded="$(kubectl get secret runtime-config \
      -n "$_promiseops_mlflow_secret_namespace" \
      -o jsonpath='{.data.MLFLOW_TRACKING_URI}' 2>/dev/null)"; then
    printf '%s\n' 'MLflow Secret read failed (value hidden).' >&2
    return 1
  fi
  if [ -z "$_promiseops_mlflow_ci_encoded" ]; then
    printf '%s\n' 'MLFLOW_TRACKING_URI is missing or empty in runtime-config.' >&2
    return 1
  fi
  if ! _promiseops_mlflow_ci_uri="$(printf '%s' "$_promiseops_mlflow_ci_encoded" | base64 -d 2>/dev/null)"; then
    unset _promiseops_mlflow_ci_encoded
    printf '%s\n' 'MLflow Secret URI decoding failed (value hidden).' >&2
    return 1
  fi
  unset _promiseops_mlflow_ci_encoded
  _promiseops_mlflow_ci_source="kubernetes-runtime-config"
  unset _promiseops_mlflow_secret_namespace
fi
case "$_promiseops_mlflow_ci_uri" in
  http://*|https://*) ;;
  *)
    printf '%s\n' 'MLflow CI URI is not an HTTP(S) endpoint; value hidden.' >&2
    return 1 ;;
esac
case "$_promiseops_mlflow_ci_uri" in
  *[[:space:]]*)
    printf '%s\n' 'MLflow CI URI contains whitespace; value hidden.' >&2
    return 1 ;;
esac
# Do not accept an interactive UI URL, embedded credentials, or a fragment.
# The value is supplied via stdin and never echoed or passed on a command line.
if ! printf '%s' "$_promiseops_mlflow_ci_uri" | python -c '
import sys
from urllib.parse import urlsplit
value = sys.stdin.read()
try:
    parsed = urlsplit(value)
    valid = (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
        and parsed.username is None and parsed.password is None
        and parsed.query == "" and parsed.fragment == ""
        and not any(ch.isspace() for ch in value)
        and (parsed.port is None or parsed.port > 0)
    )
except ValueError:
    valid = False
sys.exit(0 if valid else 1)
' >/dev/null 2>&1; then
  printf '%s\n' 'MLflow tracking URI is invalid (value hidden).' >&2
  return 1
fi
export MLFLOW_TRACKING_URI="$_promiseops_mlflow_ci_uri"
export PROMISEOPS_MLFLOW_URI_SOURCE="$_promiseops_mlflow_ci_source"
printf 'MLflow CI URI present (source=%s; value hidden).\n' "$_promiseops_mlflow_ci_source"
unset _promiseops_mlflow_ci_uri _promiseops_mlflow_ci_source
