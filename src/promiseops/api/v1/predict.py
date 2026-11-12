"""POST /predict (v1)."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from promiseops.api.deps import get_state, require_key
from promiseops.api.errors import ModelUnavailableError, error_response
from promiseops.api.services import predict_order
from promiseops.api.state import AppState
from promiseops.contracts import PredictionRequest, PredictionResponse
from promiseops.data import ConflictError

logger = logging.getLogger("promiseops.api")

StateDep = Annotated[AppState, Depends(get_state)]

router = APIRouter(dependencies=[Depends(require_key)])


@router.post("/predict")
async def predict(request: Request, state: StateDep):
    try:
        body = PredictionRequest.model_validate(await request.json())
    except (ValidationError, ValueError):
        return error_response(422, "invalid request")
    if state.model is None:
        raise ModelUnavailableError
    request_id = body.request_id or uuid.uuid4().hex
    try:
        result = await run_in_threadpool(predict_order, state, order_id=body.order_id, request_id=request_id)
    except ConflictError:
        return error_response(409, "request_id already used for another order")
    except Exception as error:
        if isinstance(error, LookupError):  # OrderNotFoundError -> registered handler
            raise
        logger.error("prediction failed: %s", type(error).__name__)
        return error_response(500, "internal error")
    response = PredictionResponse(**result)
    logger.info("prediction request_id=%s model_version=%s", request_id, response.model_version)
    return response.model_dump()
