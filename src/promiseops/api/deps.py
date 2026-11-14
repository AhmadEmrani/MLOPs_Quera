"""FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import Request

from promiseops.api.errors import UnauthorizedError
from promiseops.api.state import AppState

API_KEY_HEADER = "X-API-Key"


def get_state(request: Request) -> AppState:
    return request.app.state.ctx


def require_key(request: Request) -> None:
    supplied = request.headers.get(API_KEY_HEADER, "")
    expected = get_state(request).settings.api_key
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise UnauthorizedError
