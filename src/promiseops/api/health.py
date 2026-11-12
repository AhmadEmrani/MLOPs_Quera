"""Operational endpoints. Unversioned: probes and Prometheus depend on fixed paths."""

from __future__ import annotations

from contextlib import closing
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST

from promiseops.api.deps import get_state, require_key
from promiseops.api.errors import error_response
from promiseops.api.state import AppState

StateDep = Annotated[AppState, Depends(get_state)]

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(state: StateDep):
    if state.model is None or not state.ensure_schema():
        return error_response(503, "not ready")
    try:
        with closing(state.open_connection()) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:  # noqa: BLE001 - never leak DB details
        return error_response(503, "not ready")
    return {"status": "ready"}


@router.get("/metrics", dependencies=[Depends(require_key)])
def metrics(state: StateDep):
    return Response(state.metrics.render(), media_type=CONTENT_TYPE_LATEST)
