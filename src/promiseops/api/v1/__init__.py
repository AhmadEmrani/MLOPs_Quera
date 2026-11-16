"""Version 1 of the public API."""

from fastapi import APIRouter

from promiseops.api.v1.predict import router as predict_router

router = APIRouter()
router.include_router(predict_router)
