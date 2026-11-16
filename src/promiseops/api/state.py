"""Runtime state shared by requests: settings, model bundle, DB access."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from promiseops.api.metrics import Metrics
from promiseops.data import get_connection, init_db
from promiseops.model_pipeline.bundle import load_bundle
from promiseops.settings import Settings

logger = logging.getLogger("promiseops.api")


class AppState:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        bundle_loader: Callable = load_bundle,
        connection_factory: Callable | None = None,
    ) -> None:
        self._configured = settings
        self.bundle_loader = bundle_loader
        self.connection_factory = connection_factory
        self.settings: Settings | None = None
        self.model = None
        self.release = None
        self.schema_ready = False
        self.metrics = Metrics()

    def startup(self) -> None:
        self.settings = self._configured if self._configured is not None else Settings.from_env()
        release_path = Path(self.settings.release_path)
        release_dir = release_path if release_path.is_dir() else release_path.parent
        try:
            self.model, self.release = self.bundle_loader(release_dir)
        except Exception as error:  # noqa: BLE001 - service stays up, /predict returns 503
            self.model = self.release = None
            logger.error("model bundle not loaded: %s", type(error).__name__)
        self.ensure_schema()

    def open_connection(self):
        if self.connection_factory is not None:
            return self.connection_factory()
        return get_connection(self.settings.database_url)

    def ensure_schema(self) -> bool:
        if self.schema_ready:
            return True
        try:
            with closing(self.open_connection()) as conn:
                init_db(conn)
            self.schema_ready = True
        except Exception as error:  # noqa: BLE001 - never leak DB details
            logger.error("schema init failed: %s", type(error).__name__)
        return self.schema_ready
