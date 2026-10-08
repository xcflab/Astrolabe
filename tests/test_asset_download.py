from __future__ import annotations

import hashlib
import io
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import urllib.error

import pytest

from astrolabe import asset_download


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_attempt_budget_scales_for_large_resumable_assets() -> None:
    assert asset_download._attempt_budget(1) == 3
    assert asset_download._attempt_budget(5_678_116_398) == 10


def test_download_resumes_retained_partial_bytes(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "asset.bin"
    Path(str(destination) + ".part").write_bytes(b"abc")
    requests = []

    def fake_urlopen(request, **_kwargs):
        requests.append(request)
        return _Response(
            b"def",
            status=206,
            headers={"Content-Range": "bytes 3-5/6"},
        )

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)
    expected = hashlib.sha256(b"abcdef").hexdigest()

    result = asset_download.download_url(
        "https://example.invalid/asset.bin",
        destination,
        minimum_size=6,
        expected_sha256=expected,
    )

    assert result.read_bytes() == b"abcdef"
    assert requests[0].get_header("Range") == "bytes=3-"
    assert not Path(str(destination) + ".part").exists()


def test_download_reuses_truncated_final_as_partial(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"abc")

    def fake_urlopen(request, **_kwargs):
        assert request.get_header("Range") == "bytes=3-"
        return _Response(
            b"def",
            status=206,
            headers={"Content-Range": "bytes 3-5/6"},
        )

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)
    result = asset_download.download_url(
        "https://example.invalid/asset.bin",
        destination,
        minimum_size=6,
        expected_sha256=hashlib.sha256(b"abcdef").hexdigest(),
    )

    assert result.read_bytes() == b"abcdef"


def test_dry_run_never_moves_or_rewrites_existing_invalid_asset(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"short")

    result = asset_download.download_url(
        "https://example.invalid/asset.bin",
        destination,
        minimum_size=100,
        expected_sha256="0" * 64,
        dry_run=True,
    )

    assert result == destination
    assert destination.read_bytes() == b"short"
    assert not Path(str(destination) + ".part").exists()
    assert not Path(str(destination) + ".invalid").exists()


def test_permanent_http_error_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Permanent HTTP error"):
        asset_download.download_url(
            "https://example.invalid/restricted.bin",
            tmp_path / "restricted.bin",
            attempts=3,
        )
    assert calls == 1


def test_checksum_failure_keeps_partial_and_does_not_retry(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def fake_urlopen(_request, **_kwargs):
        nonlocal calls
        calls += 1
        return _Response(b"wrong")

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "asset.bin"

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        asset_download.download_url(
            "https://example.invalid/asset.bin",
            destination,
            expected_sha256=hashlib.sha256(b"right").hexdigest(),
            attempts=3,
        )
    assert calls == 1
    assert Path(str(destination) + ".part").read_bytes() == b"wrong"


def test_validated_checksum_stamp_avoids_rehash_on_warm_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"content")
    expected = hashlib.sha256(b"content").hexdigest()
    asset_download.validate_file(destination, expected_sha256=expected)
    monkeypatch.setattr(asset_download, "_metadata_reuse_is_safe", lambda _stat: True)
    monkeypatch.setattr(
        asset_download,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("warm path rehashed")),
    )
    asset_download.validate_file(destination, expected_sha256=expected)


def test_checksum_cache_rejects_same_size_rewrite_with_preserved_mtime(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"content")
    expected = hashlib.sha256(b"content").hexdigest()
    asset_download.validate_file(destination, expected_sha256=expected)
    original = destination.stat()

    destination.write_bytes(b"corrupt")
    os.utime(destination, ns=(original.st_atime_ns, original.st_mtime_ns))

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        asset_download.validate_file(destination, expected_sha256=expected)


def test_read_only_asset_uses_external_validation_cache(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "shared" / "asset.bin"
    destination.parent.mkdir()
    destination.write_bytes(b"content")
    expected = hashlib.sha256(b"content").hexdigest()
    validation_cache = tmp_path / "validation-cache"
    monkeypatch.setenv("GH_ASSET_VALIDATION_CACHE", str(validation_cache))
    original_write_stamp = asset_download._write_stamp

    def reject_sidecar(stamp: Path, payload: str) -> None:
        if stamp == asset_download._checksum_stamp(destination):
            raise OSError("read-only asset directory")
        original_write_stamp(stamp, payload)

    monkeypatch.setattr(asset_download, "_write_stamp", reject_sidecar)
    asset_download.validate_file(destination, expected_sha256=expected)
    asset_download._PROCESS_VALIDATIONS.clear()
    monkeypatch.setattr(asset_download, "_metadata_reuse_is_safe", lambda _stat: True)
    monkeypatch.setattr(
        asset_download,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("fallback cache was ignored")),
    )

    asset_download.validate_file(destination, expected_sha256=expected)
    assert any(validation_cache.glob("*.sha256"))


def test_unreachable_network_fails_without_repeating_long_timeouts(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0

    def fake_urlopen(_request, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError(OSError(101, "Network is unreachable"))

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Permanent network error"):
        asset_download.download_url(
            "https://example.invalid/asset.bin",
            tmp_path / "asset.bin",
            attempts=3,
        )
    assert calls == 1


def test_transient_failure_uses_backoff_then_resumes(tmp_path: Path, monkeypatch) -> None:
    calls = 0
    sleeps = []

    def fake_urlopen(_request, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary")
        return _Response(b"complete")

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(asset_download.time, "sleep", sleeps.append)
    destination = tmp_path / "asset.bin"

    asset_download.download_url(
        "https://example.invalid/asset.bin",
        destination,
        minimum_size=8,
        attempts=2,
    )

    assert destination.read_bytes() == b"complete"
    assert calls == 2
    assert sleeps == [1]


def test_real_http_connection_drop_resumes_with_range(tmp_path: Path) -> None:
    payload = b"a" * (8 * 1024 * 1024) + b"b" * 1024
    observed_ranges: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers.get("Range")
            observed_ranges.append(range_header)
            if range_header is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[: 8 * 1024 * 1024])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return

            assert range_header == "bytes=8388608-"
            start = 8 * 1024 * 1024
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload) - start))
            self.send_header(
                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            )
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = tmp_path / "asset.bin"
        result = asset_download.download_url(
            f"http://127.0.0.1:{server.server_port}/asset.bin",
            destination,
            minimum_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            attempts=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.read_bytes() == payload
    assert observed_ranges == [None, "bytes=8388608-"]


def test_real_http_ignored_range_restarts_without_appending(tmp_path: Path) -> None:
    payload = b"complete-model" * 8192
    destination = tmp_path / "asset.bin"
    part = Path(str(destination) + ".part")
    part.write_bytes(payload[:12345])
    observed_ranges: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            observed_ranges.append(self.headers.get("Range"))
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = asset_download.download_url(
            f"http://127.0.0.1:{server.server_port}/asset.bin",
            destination,
            minimum_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            attempts=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert observed_ranges == ["bytes=12345-"]
    assert result.read_bytes() == payload
    assert result.stat().st_size == len(payload)


def test_real_http_truncated_206_without_sha_is_not_promoted(tmp_path: Path) -> None:
    payload = b"complete-model" * 8192
    initial = 12345
    sent = 16384

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            assert self.headers.get("Range") == f"bytes={initial}-"
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload) - initial))
            self.send_header(
                "Content-Range", f"bytes {initial}-{len(payload) - 1}/{len(payload)}"
            )
            self.end_headers()
            self.wfile.write(payload[initial : initial + sent])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "asset.bin"
    part = Path(str(destination) + ".part")
    part.write_bytes(payload[:initial])
    try:
        with pytest.raises(RuntimeError, match="response body is incomplete"):
            asset_download.download_url(
                f"http://127.0.0.1:{server.server_port}/asset.bin",
                destination,
                minimum_size=1,
                attempts=1,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert not destination.exists()
    assert part.stat().st_size == initial + sent


def test_real_http_416_short_partial_without_sha_is_quarantined(
    tmp_path: Path,
) -> None:
    total = 524288

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            assert self.headers.get("Range") == "bytes=34567-"
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "asset.bin"
    part = Path(str(destination) + ".part")
    part.write_bytes(b"x" * 34567)
    try:
        with pytest.raises(RuntimeError, match="did not prove"):
            asset_download.download_url(
                f"http://127.0.0.1:{server.server_port}/asset.bin",
                destination,
                minimum_size=1,
                attempts=1,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert not destination.exists()
    assert not part.exists()
    assert Path(str(part) + ".invalid").stat().st_size == 34567


def test_real_http_416_short_partial_with_matching_sha_is_quarantined(
    tmp_path: Path,
) -> None:
    total = 106496
    partial_payload = b"matching-short-partial" * 2048

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            assert self.headers.get("Range") == f"bytes={len(partial_payload)}-"
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "asset.bin"
    part = Path(str(destination) + ".part")
    part.write_bytes(partial_payload)
    try:
        with pytest.raises(RuntimeError, match="did not prove"):
            asset_download.download_url(
                f"http://127.0.0.1:{server.server_port}/asset.bin",
                destination,
                minimum_size=1,
                expected_sha256=hashlib.sha256(partial_payload).hexdigest(),
                attempts=1,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    invalid = Path(str(part) + ".invalid")
    assert not destination.exists()
    assert not part.exists()
    assert invalid.read_bytes() == partial_payload


def test_http_416_promotes_exact_length_partial_without_sha(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "asset.bin"
    part = Path(str(destination) + ".part")
    part.write_bytes(b"complete")

    def fake_urlopen(request, **_kwargs):
        raise urllib.error.HTTPError(
            request.full_url,
            416,
            "Range Not Satisfiable",
            {"Content-Range": "bytes */8"},
            None,
        )

    monkeypatch.setattr(asset_download.urllib.request, "urlopen", fake_urlopen)
    result = asset_download.download_url(
        "https://example.invalid/asset.bin",
        destination,
        minimum_size=1,
        attempts=1,
    )

    assert result.read_bytes() == b"complete"
    assert not part.exists()
