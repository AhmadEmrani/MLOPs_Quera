"""Runtime settings for the API, read from environment variables.

Secrets (API key, database URL) are never printed, logged or returned by the API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RELEASE_PATH = Path("/opt/promiseops/release/release.json")


class SettingsError(ValueError):
    """Raised when a required setting is missing. Never contains a value."""


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    api_key: str = field(repr=False)
    release_path: Path = DEFAULT_RELEASE_PATH

    def __repr__(self) -> str:
        return f"Settings(database_url='***', api_key='***', release_path={str(self.release_path)!r})"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        raw_release = source.get("PROMISEOPS_RELEASE", "").strip()
        return cls(
            database_url=_required(source, "PROMISEOPS_DATABASE_URL"),
            api_key=_required(source, "PROMISEOPS_API_KEY"),
            release_path=Path(raw_release) if raw_release else DEFAULT_RELEASE_PATH,
        )


def _required(source: Mapping[str, str], name: str) -> str:
    value = source.get(name, "").strip()
    if not value:
        raise SettingsError(f"missing required setting: {name}")
    return value
