#!/usr/bin/env python3
"""Safely install the shared Kubernetes MLflow URI fallback in PromiseOps CI.

Run from the repository root on refactor/codesAndLifeCycle. Changes only
existing CI wiring and a shared env resolver; never deploys or touches secrets.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path("ci/require_mlflow_ci_env.sh")
INSPECT = Path(".gitlab/ci/inspect.yml")
CONFIG_PROBE = Path("ci/inspect_ci_config.py")

FALLBACK_OLD = '''if [ -z "$_promiseops_mlflow_ci_uri" ]; then
  _promiseops_mlflow_ci_error
  return 1
fi'''
FALLBACK_NEW = '''# The team platform already supplies this URI in the existing runtime-config
# Kubernetes Secret. GitLab CI does not automatically inherit that env value.
# Use the SAME g02-prod registry endpoint for Golden, release, and promotion.
# Only the currently authorized CI Kubernetes identity can read this Secret.
if [ -z "$_promiseops_mlflow_ci_uri" ]; then
  _promiseops_mlflow_secret_namespace="${PROMISEOPS_MLFLOW_SECRET_NAMESPACE:-g02-prod}"
  case "$_promiseops_mlflow_secret_namespace" in
    g02-prod|g02-staging) ;;
    *)
      printf '%s\\n' 'MLflow Secret namespace must be g02-prod or g02-staging.' >&2
      return 1 ;;
  esac
  if ! command -v kubectl >/dev/null 2>&1; then
    printf '%s\\n' 'kubectl is unavailable in this GitLab CI job.' >&2
    return 1
  fi
  if ! kubectl auth can-i get secret/runtime-config \\
      -n "$_promiseops_mlflow_secret_namespace" --quiet >/dev/null 2>&1; then
    printf '%s\\n' 'CI identity cannot read the authorized MLflow runtime-config Secret.' >&2
    return 1
  fi
  if ! _promiseops_mlflow_ci_encoded="$(kubectl get secret runtime-config \\
      -n "$_promiseops_mlflow_secret_namespace" \\
      -o jsonpath='{.data.MLFLOW_TRACKING_URI}' 2>/dev/null)"; then
    printf '%s\\n' 'MLflow Secret read failed (value hidden).' >&2
    return 1
  fi
  if [ -z "$_promiseops_mlflow_ci_encoded" ]; then
    printf '%s\\n' 'MLFLOW_TRACKING_URI is missing or empty in runtime-config.' >&2
    return 1
  fi
  if ! _promiseops_mlflow_ci_uri="$(printf '%s' "$_promiseops_mlflow_ci_encoded" | base64 -d 2>/dev/null)"; then
    unset _promiseops_mlflow_ci_encoded
    printf '%s\\n' 'MLflow Secret URI decoding failed (value hidden).' >&2
    return 1
  fi
  unset _promiseops_mlflow_ci_encoded
  _promiseops_mlflow_ci_source="kubernetes-runtime-config"
  unset _promiseops_mlflow_secret_namespace
fi'''

VALIDATE_OLD = '''export MLFLOW_TRACKING_URI="$_promiseops_mlflow_ci_uri"
printf 'MLflow CI URI present (source=%s; value hidden).\\n' "$_promiseops_mlflow_ci_source"'''
VALIDATE_NEW = '''# Do not accept an interactive UI URL, embedded credentials, or a fragment.
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
  printf '%s\\n' 'MLflow tracking URI is invalid (value hidden).' >&2
  return 1
fi
export MLFLOW_TRACKING_URI="$_promiseops_mlflow_ci_uri"
export PROMISEOPS_MLFLOW_URI_SOURCE="$_promiseops_mlflow_ci_source"
printf 'MLflow CI URI present (source=%s; value hidden).\\n' "$_promiseops_mlflow_ci_source"'''


def replace_once(body: str, old: str, new: str, desc: str) -> str:
    if new in body:
        return body
    count = body.count(old)
    if count != 1:
        raise ValueError(f"{desc}: expected exactly one source anchor; found {count}. Refusing to overwrite.")
    return body.replace(old, new, 1)


def rewrite_helper(body: str) -> str:
    body = replace_once(body, FALLBACK_OLD, FALLBACK_NEW, "MLflow URI fallback")
    body = replace_once(body, VALIDATE_OLD, VALIDATE_NEW, "MLflow URI safe validation")
    body = replace_once(
        body,
        "printf '%s\\n' 'Configure MLFLOW_TRACKING_URI in GitLab > Settings > CI/CD > Variables.' >&2",
        "printf '%s\\n' 'Check GitLab variables or existing authorized g02-prod/runtime-config MLflow Secret.' >&2",
        "missing URI diagnostic",
    )
    return body


INSPECT_OLD = '''      mkdir -p reports/mlflow
      python ci/inspect_mlflow_ci_access.py --phase preflight --report reports/mlflow/ci-access.json'''
INSPECT_NEW = '''      mkdir -p reports/mlflow
      . ci/require_mlflow_ci_env.sh
      python ci/inspect_mlflow_ci_access.py --phase preflight --report reports/mlflow/ci-access.json'''

NEW_DIRECT = '''inspect-mlflow-direct:
  stage: inspect
  needs: []
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
  allow_failure: true
  script:
    - set -eu
    - . ci/require_mlflow_ci_env.sh
    - python ci/check_mlflow_http.py

'''


def rewrite_inspect(body: str) -> str:
    body = replace_once(body, INSPECT_OLD, INSPECT_NEW, "MLflow inspect preflight")
    m = re.search(r"^inspect-mlflow-direct:\n.*?(?=^[a-z][a-z0-9-]*:\n|\Z)", body, re.M | re.S)
    if m is None:
        raise ValueError("Missing inspect-mlflow-direct anchor; do not overwrite unknown layout")
    if m.group(0) != NEW_DIRECT:
        body = body[:m.start()] + NEW_DIRECT + body[m.end():]
    return body


PROBE_NOTE = '''print(
    "\\nMLflow note: this displays raw GitLab CI env only. The approved "
    "Kubernetes runtime-config fallback is loaded by "
    "ci/require_mlflow_ci_env.sh in MLflow jobs."
)
'''


def rewrite_ci_config(body: str) -> str:
    if PROBE_NOTE in body:
        return body
    if "NOTE: This checks GitLab CI" not in body:
        raise ValueError("CI configuration inspection layout changed")
    return body.rstrip() + "\n\n" + PROBE_NOTE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify anchors; make no changes")
    args = parser.parse_args()
    changes = (
        (HELPER, rewrite_helper),
        (INSPECT, rewrite_inspect),
        (CONFIG_PROBE, rewrite_ci_config),
    )
    results = []
    for path, fn in changes:
        target = ROOT / path
        if not target.is_file():
            raise SystemExit(f"Missing {path}: run from the matching repository branch")
        original = target.read_text(encoding="utf-8")
        revised = fn(original)
        results.append((target, original, revised))
    for target, original, revised in results:
        state = "already applied" if original == revised else "would update" if args.check else "updated"
        if not args.check and original != revised:
            target.write_text(revised, encoding="utf-8")
        print(f"{target.relative_to(ROOT)}: {state}")
    print("No Kubernetes or MLflow operations performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
