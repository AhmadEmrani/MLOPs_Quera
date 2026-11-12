"""Domain errors and their generic, leak-free HTTP mapping."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("promiseops.api")


class OrderNotFoundError(LookupError):
    """The order does not exist or is not released yet."""


class UnauthorizedError(Exception):
    """Missing or wrong API key."""


class ModelUnavailableError(Exception):
    """No validated model bundle is loaded."""


def error_response(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(UnauthorizedError)
    async def _unauthorized(_request: Request, _exc: Exception):
        return error_response(401, "unauthorized")

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, _exc: Exception):
        return error_response(422, "invalid request")

    @app.exception_handler(OrderNotFoundError)
    async def _not_found(_request: Request, _exc: Exception):
        return error_response(404, "order not found")

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(_request: Request, _exc: Exception):
        return error_response(503, "model unavailable")
