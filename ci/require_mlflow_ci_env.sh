#!/bin/sh
# Source from GitLab CI before using MLflow. Never print URI values or credentials.
# Gateway-provisioned training Worker env vars are NOT inherited by CI Runner.
set +x

_promiseops_mlflow_ci_error() {
  printf '%s\n' 'MLflow CI access is not configured for this job.' >&2
  printf '%s\n' 'The training gateway injects MLflow settings into worker Pods, NOT GitLab CI jobs.' >&2
  printf '%s\n' 'Configure MLFLOW_TRACKING_URI in GitLab > Settings > CI/CD > Variables.' >&2
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

if [ -z "$_promiseops_mlflow_ci_uri" ]; then
  _promiseops_mlflow_ci_error
  return 1
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
export MLFLOW_TRACKING_URI="$_promiseops_mlflow_ci_uri"
printf 'MLflow CI URI present (source=%s; value hidden).\n' "$_promiseops_mlflow_ci_source"
unset _promiseops_mlflow_ci_uri _promiseops_mlflow_ci_source
