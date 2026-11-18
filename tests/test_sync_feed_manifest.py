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


def test_real_gateway_releases_shape_without_manifest_batch_id(monkeypatch):
    """Live G02 /feed uses releases, and batch_id exists only in file JSON."""
    from hashlib import sha256

    payload = b'{"batch_id":"orders-example","kind":"orders"}'
    release = {
        "available_at": "2018-03-01T00:00:00Z",
        "file": "example.json",
        "file_sha256": sha256(payload).hexdigest(),
        "kind": "orders",
        "orders": 1,
        "outcomes": 0,
        "payload_sha256": "a" * 64,
    }
    manifest = {"schema_version": "1.0", "as_of": CUTOFF, "releases": [release]}
    original_client = httpx.Client

    def handler(request):
        if request.url.path == "/feed":
            return httpx.Response(200, json=manifest)
        if request.url.path == "/feed/example.json":
            return httpx.Response(200, content=payload)
        return httpx.Response(404)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(data.httpx, "Client", client_factory)
    seen = []

    def fake_ingest(conn, raw, *, as_of, expected_sha256):
        seen.append((raw, expected_sha256))
        return {"batch_id": "orders-example", "status": "committed"}

    monkeypatch.setattr(data, "ingest_batch", fake_ingest)
    receipts = data.sync_feed(object(), base_url="http://example.test", token="hidden", as_of=CUTOFF)
    assert receipts[0]["batch_id"] == "orders-example"
    assert seen == [(payload, "a" * 64)]


def test_real_gateway_releases_reject_cutoff_mismatch():
    release = {
        "available_at": "2018-03-01T00:00:00Z",
        "file": "example.json",
        "kind": "orders",
        "payload_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="different as_of cutoff"):
        data.parse_feed_manifest(
            {"as_of": "2018-11-06T00:00:00Z", "releases": [release]},
            as_of=CUTOFF,
        )


def test_real_gateway_releases_reject_invalid_sha():
    release = {
        "available_at": "2018-03-01T00:00:00Z",
        "file": "example.json",
        "kind": "orders",
        "payload_sha256": "not-hex",
    }
    with pytest.raises(ValueError, match="payload_sha256"):
        data.parse_feed_manifest({"releases": [release]}, as_of=CUTOFF)


def test_training_gateway_request_uses_image_key():
    """Do not regress the explicit POST /training public contract."""
    import ast
    from pathlib import Path

    source = Path("dags/training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "validate_request":
            returns = [x for x in ast.walk(node) if isinstance(x, ast.Return)]
            assert len(returns) == 1
            keys = [key.value for key in returns[0].value.keys]
            assert "image" in keys
            assert "worker_image" not in keys
            return
    pytest.fail("validate_request not found")
