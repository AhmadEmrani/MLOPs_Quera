"""Regression tests for company feed manifest shapes and cutoff guards."""

import httpx
import pytest

from promiseops import data


CUTOFF = "2018-11-05T00:00:00Z"
ENTRY = {
    "batch_id": "orders-sample",
    "available_at": "2018-03-01T00:00:00Z",
    "kind": "orders",
    "payload_sha256": "a" * 64,
    "file": "sample.json",
}


def _install_mock_client(monkeypatch, manifest):
    original_client = httpx.Client

    def handler(request):
        if request.url.path == "/feed":
            return httpx.Response(200, json=manifest)
        if request.url.path == "/feed/sample.json":
            return httpx.Response(200, content=b"{}")
        return httpx.Response(404)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(data.httpx, "Client", client_factory)


@pytest.mark.parametrize("manifest", [
    {"as_of": CUTOFF, "entries": [ENTRY]},
    [ENTRY],
])
def test_sync_feed_accepts_list_and_object_manifests(monkeypatch, manifest):
    _install_mock_client(monkeypatch, manifest)
    seen = []

    def fake_ingest(connection, raw, *, as_of, expected_sha256):
        seen.append(expected_sha256)
        return {"batch_id": "orders-sample", "status": "committed"}

    monkeypatch.setattr(data, "ingest_batch", fake_ingest)
    result = data.sync_feed(object(), base_url="http://example.test", token="hidden", as_of=CUTOFF)
    assert result == [{"batch_id": "orders-sample", "status": "committed"}]
    assert seen == ["a" * 64]


def test_sync_feed_rejects_bad_checksum(monkeypatch):
    _install_mock_client(monkeypatch, {"entries": [{**ENTRY, "payload_sha256": "invalid"}]})
    with pytest.raises(ValueError, match="payload_sha256"):
        data.sync_feed(object(), base_url="http://example.test", token="hidden", as_of=CUTOFF)


def test_sync_feed_rejects_wrong_cutoff_echo(monkeypatch):
    _install_mock_client(monkeypatch, {"as_of": "2018-11-06T00:00:00Z", "entries": [ENTRY]})
    with pytest.raises(ValueError, match="different as_of cutoff"):
        data.sync_feed(object(), base_url="http://example.test", token="hidden", as_of=CUTOFF)
