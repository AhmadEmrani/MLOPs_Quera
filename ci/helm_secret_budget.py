#!/usr/bin/env python3
"""Guard Helm upgrades against namespace Secret quotas without reading Secret data.

Only release=promiseops, type=helm.sh/release.v1, owner=helm, name=promiseops,
and status=superseded can be deleted. Keep deployed, all non-superseded,
plus the two newest superseded rollback versions. No credential values/logs.
Run inside the existing GitLab resource_group for the target namespace.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

RELEASE = "promiseops"
NAME_RE = re.compile(r"sh\.helm\.release\.v1\.promiseops\.v([1-9][0-9]*)\Z")
NAMESPACE_RE = re.compile(r"g0[1-7]-(?:staging|prod)\Z")
# Only metadata; NEVER ask kubectl for .data or full Secret JSON.
ALL_SECRETS = (
    '{range .items[*]}{.metadata.name}{"\\t"}{.type}{"\\t"}'
    '{.metadata.labels.owner}{"\\t"}{.metadata.labels.name}{"\\t"}'
    '{.metadata.labels.version}{"\\t"}{.metadata.labels.status}{"\\n"}{end}'
)
ONE_SECRET = (
    '{.metadata.name}{"\\t"}{.type}{"\\t"}'
    '{.metadata.labels.owner}{"\\t"}{.metadata.labels.name}{"\\t"}'
    '{.metadata.labels.version}{"\\t"}{.metadata.labels.status}'
)


class BudgetBlocked(RuntimeError):
    """Safe, actionable reason; no Kubernetes or credential error strings."""


@dataclass(frozen=True)
class Secret:
    name: str
    kind: str
    owner: str
    release: str
    version_label: str
    status: str

    @property
    def version(self) -> int:
        match = NAME_RE.fullmatch(self.name)
        if match is None or self.version_label != match.group(1):
            raise BudgetBlocked("helm_release_metadata_mismatch")
        return int(match.group(1))

    @property
    def is_ours(self) -> bool:
        return (
            self.kind == "helm.sh/release.v1"
            and self.owner == "helm"
            and self.release == RELEASE
        )


def _run(args: list[str]) -> str:
    try:
        p = subprocess.run(args, text=True, capture_output=True, check=False, timeout=35)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BudgetBlocked(f"kubectl_unavailable_or_timeout:{type(error).__name__}") from None
    if p.returncode != 0:
        # CLI stderr may include cluster URLs or auth internals: never forward it.
        raise BudgetBlocked("kubectl_metadata_read_or_delete_denied")
    return p.stdout


def read_all(namespace: str) -> list[Secret]:
    output = _run(["kubectl", "-n", namespace, "get", "secrets", "-o", "jsonpath=" + ALL_SECRETS])
    items: list[Secret] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            raise BudgetBlocked("unexpected_secret_metadata_format")
        items.append(Secret(*fields))
    names = [x.name for x in items]
    if len(names) != len(set(names)):
        raise BudgetBlocked("duplicate_secret_name_in_listing")
    return items


def quota(namespace: str, *, fallback: int) -> tuple[int, int | None, str]:
    """ResourceQuota is authoritative when readable; fallback is a conservative cap."""
    try:
        import json
        output = _run(["kubectl", "-n", namespace, "get", "resourcequotas", "-o", "json"])
        data = json.loads(output)
        limits = []
        for item in data.get("items", []):
            hard = item.get("status", {}).get("hard", {}).get("count/secrets")
            used = item.get("status", {}).get("used", {}).get("count/secrets")
            if hard is not None:
                limits.append((int(hard), int(used) if used is not None else None))
        if limits:
            bound, used = min(limits, key=lambda pair: pair[0])
            return min(fallback, bound), used, "resourcequota"
    except (BudgetBlocked, ValueError, KeyError, TypeError):
        pass
    return fallback, None, "configured_fallback"


def plan(items: list[Secret], *, limit: int, minimum_free: int = 3,
         preferred_keep: int = 5, minimum_keep: int = 3,
         quota_used: int | None = None) -> tuple[list[Secret], int, int, list[int]]:
    """Compute a deletion plan without side effects.

    Keep active release, any pending/failed release, and minimum two previous
    superseded rollback versions. Prefer preserving five latest revisions.
    """
    if minimum_free < 1 or preferred_keep < minimum_keep or minimum_keep < 3:
        raise BudgetBlocked("invalid_retention_policy")
    if limit < minimum_free:
        raise BudgetBlocked("secret_quota_too_small")

    ours = [s for s in items if s.is_ours]
    revisions = [s.version for s in ours]
    if len(revisions) != len(set(revisions)):
        raise BudgetBlocked("duplicate_release_revision")
    if any(s.status not in {"deployed", "superseded", "failed", "pending-upgrade", "pending-install", "pending-rollback", "uninstalling", "uninstalled"} for s in ours):
        raise BudgetBlocked("unknown_helm_release_status")
    if len([s for s in ours if s.status == "deployed"]) > 1:
        raise BudgetBlocked("multiple_deployed_release_revisions")
    if ours and not any(s.status == "deployed" for s in ours):
        raise BudgetBlocked("no_deployed_release_revision_recovery_required")
    if any(s.status.startswith("pending") for s in ours):
        raise BudgetBlocked("helm_release_has_pending_revision")
    if any((s.kind == "helm.sh/release.v1" or NAME_RE.fullmatch(s.name)) and not s.is_ours for s in items if s.name.startswith("sh.helm.release.v1.promiseops.")):
        raise BudgetBlocked("helm_release_metadata_mismatch")

    # Never delete more recent than the latest three revisions or any non-superseded.
    protected = {s.version for s in sorted(ours, key=lambda s: s.version, reverse=True)[:minimum_keep]}
    # Additionally keep deployed and two most recent earlier superseded versions.
    current = next((s for s in ours if s.status == "deployed"), None)
    if current:
        protected.add(current.version)
        protected.update(s.version for s in sorted(
            (s for s in ours if s.status == "superseded" and s.version < current.version),
            key=lambda s: s.version, reverse=True,
        )[:2])
    # First shrink history to preferred_keep, then reclaim more only if required.
    first_pass_protected = {s.version for s in sorted(ours, key=lambda s: s.version, reverse=True)[:preferred_keep]}
    candidates = sorted((s for s in ours if s.status == "superseded" and s.version not in protected), key=lambda s: s.version)
    preferred = [s for s in candidates if s.version not in first_pass_protected]
    remaining = [s for s in candidates if s.version in first_pass_protected]
    base_used = max(len(items), quota_used or 0)
    deletions = list(preferred)
    free = limit - (base_used - len(deletions))
    for s in remaining:
        if free >= minimum_free:
            break
        deletions.append(s)
        free += 1
    if free < minimum_free:
        raise BudgetBlocked(
            "insufficient_secret_headroom_after_safe_prune; "
            "ask_platform_admin_to_raise_quota_or_remove_other_approved_secrets"
        )
    return deletions, base_used, free, sorted(protected)


def _confirm_unchanged(namespace: str, candidate: Secret) -> None:
    output = _run(["kubectl", "-n", namespace, "get", "secret", candidate.name,
                   "-o", "jsonpath=" + ONE_SECRET])
    if Secret(*output.split("\t")) != candidate:
        raise BudgetBlocked("release_revision_changed_during_cleanup")


def execute(namespace: str, *, dry_run: bool, limit: int, minimum_free: int = 3) -> dict:
    if not NAMESPACE_RE.fullmatch(namespace):
        raise BudgetBlocked("invalid_namespace")
    if not 3 <= limit <= 1000:
        raise BudgetBlocked("invalid_secret_quota_fallback")
    items = read_all(namespace)
    cap, observed_used, source = quota(namespace, fallback=limit)
    deletes, used, free, protected = plan(items, limit=cap,
                                         minimum_free=minimum_free,
                                         quota_used=observed_used)
    if deletes and not dry_run:
        # A Release can be in-progress outside CI; require the team's
        # single-deployer resource_group and re-verify before each deletion.
        allowed = _run(["kubectl", "auth", "can-i", "delete", "secrets", "-n", namespace]).strip()
        if allowed != "yes":
            raise BudgetBlocked("ci_identity_cannot_delete_superseded_helm_history")
        for candidate in deletes:
            _confirm_unchanged(namespace, candidate)
            _run(["kubectl", "-n", namespace, "delete", "secret", candidate.name, "--wait=true"])
        now = read_all(namespace)
        left = max(len(now), (observed_used or 0) - len(deletes))
        if cap - left < minimum_free:
            raise BudgetBlocked("secret_quota_changed_during_cleanup")
    result = {
        "namespace": namespace, "quota": cap, "quota_source": source,
        "current_secret_count": used, "planned_superseded_deletions": len(deletes),
        "headroom_after_plan": free, "protected_revisions": protected,
        "mode": "dry_run" if dry_run else "apply",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quota-fallback", type=int, default=20)
    parser.add_argument("--minimum-free", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = execute(args.namespace, dry_run=args.dry_run,
                         limit=args.quota_fallback, minimum_free=args.minimum_free)
    except BudgetBlocked as exc:
        print(f"HELM_SECRET_BUDGET: BLOCKED ({exc})", file=sys.stderr)
        return 1
    print("HELM_SECRET_BUDGET: PASS " + " ".join(f"{k}={v}" for k, v in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
