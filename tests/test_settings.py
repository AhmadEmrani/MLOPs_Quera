from pathlib import Path

import pytest

from promiseops.settings import DEFAULT_RELEASE_PATH, Settings, SettingsError

DB_URL = "postgresql+psycopg://user:pw@db/app"
API_KEY = "s3cret-key"
BASE_ENV = {"PROMISEOPS_DATABASE_URL": DB_URL, "PROMISEOPS_API_KEY": API_KEY}


def test_reads_required_values_and_default_release_path():
    settings = Settings.from_env(BASE_ENV)

    assert settings.database_url == DB_URL
    assert settings.api_key == API_KEY
    assert settings.release_path == Path("/opt/promiseops/release/release.json")
    assert settings.release_path == DEFAULT_RELEASE_PATH


def test_release_path_comes_from_promiseops_release():
    settings = Settings.from_env({**BASE_ENV, "PROMISEOPS_RELEASE": "/tmp/artifacts/release.json"})

    assert settings.release_path == Path("/tmp/artifacts/release.json")


def test_blank_release_uses_default():
    settings = Settings.from_env({**BASE_ENV, "PROMISEOPS_RELEASE": "  "})

    assert settings.release_path == DEFAULT_RELEASE_PATH


def test_from_env_without_argument_reads_os_environ(monkeypatch):
    monkeypatch.setenv("PROMISEOPS_DATABASE_URL", DB_URL)
    monkeypatch.setenv("PROMISEOPS_API_KEY", API_KEY)
    monkeypatch.delenv("PROMISEOPS_RELEASE", raising=False)

    settings = Settings.from_env()

    assert settings.api_key == API_KEY
    assert settings.release_path == DEFAULT_RELEASE_PATH


def test_values_are_stripped():
    settings = Settings.from_env({**BASE_ENV, "PROMISEOPS_API_KEY": f"  {API_KEY} "})

    assert settings.api_key == API_KEY


@pytest.mark.parametrize("missing", ["PROMISEOPS_DATABASE_URL", "PROMISEOPS_API_KEY"])
def test_missing_setting_names_only_the_variable(missing):
    env = {k: v for k, v in BASE_ENV.items() if k != missing}

    with pytest.raises(SettingsError) as info:
        Settings.from_env(env)

    message = str(info.value)
    assert missing in message
    assert DB_URL not in message
    assert API_KEY not in message


def test_blank_required_value_is_rejected_without_leaking_other_secret():
    with pytest.raises(SettingsError) as info:
        Settings.from_env({**BASE_ENV, "PROMISEOPS_API_KEY": "   "})

    assert DB_URL not in str(info.value)


def test_repr_does_not_leak_secrets():
    text = repr(Settings.from_env(BASE_ENV))

    assert API_KEY not in text
    assert DB_URL not in text
    assert "user:pw" not in text
