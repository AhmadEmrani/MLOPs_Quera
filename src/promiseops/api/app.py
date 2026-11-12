"""Application factory."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from promiseops.api import health, v1
from promiseops.api.errors import register_error_handlers
from promiseops.api.metrics import install_metrics_middleware
from promiseops.api.state import AppState
from promiseops.model_pipeline.bundle import load_bundle
from promiseops.settings import Settings


def create_app(
    settings: Settings | None = None,
    *,
    bundle_loader: Callable = load_bundle,
    connection_factory: Callable | None = None,
) -> FastAPI:
    state = AppState(settings, bundle_loader=bundle_loader, connection_factory=connection_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state.startup()
        yield

    app = FastAPI(title="PromiseOps", lifespan=lifespan)
    app.state.ctx = state
    register_error_handlers(app)
    install_metrics_middleware(app, state.metrics)

    app.include_router(health.router)
    # /predict is the contract path; /v1/predict is the versioned alias.
    app.include_router(v1.router, include_in_schema=False)
    app.include_router(v1.router, prefix="/v1")
    return app
