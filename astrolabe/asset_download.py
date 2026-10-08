"""Small resumable downloader shared by bootstrap and container asset prefetch."""

from __future__ import annotations

import hashlib
import fcntl
import errno
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


_PROCESS_VALIDATIONS: set[tuple[str, int, int, int, int, int, str]] = set()
_RESUME_PROGRESS_BYTES = 700 * 1024 * 1024
_METADATA_SETTLE_NS = 2_000_000_000


def _attempt_budget(minimum_size: int) -> int:
    progress_windows = (minimum_size + _RESUME_PROGRESS_BYTES - 1) // _RESUME_PROGRESS_BYTES
    return max(3, min(12, progress_windows + 2))


def _checksum_stamp(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def _fallback_checksum_stamp(path: Path) -> Path:
    cache_root = os.environ.get("GH_ASSET_VALIDATION_CACHE")
    if cache_root:
        root = Path(cache_root).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        root = root / "guided-astrolabe" / "asset-validation"
    key = hashlib.sha256(os.fsencode(str(path.resolve()))).hexdigest()
    return root / f"{key}.sha256"


def _write_stamp(stamp: Path, payload: str) -> None:
    stamp.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(stamp) + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="ascii")
        os.replace(tmp, stamp)
    finally:
        tmp.unlink(missing_ok=True)


def _write_checksum_stamp(
    path: Path, checksum: str, *, use_adjacent_stamp: bool = True
) -> None:
    stat = path.stat()
    payload = (
        f"v2 {checksum} {stat.st_size} {stat.st_mtime_ns} "
        f"{stat.st_ctime_ns} {stat.st_dev} {stat.st_ino}\n"
    )
    if use_adjacent_stamp:
        primary = _checksum_stamp(path)
        try:
            _write_stamp(primary, payload)
            return
        except OSError:
            pass
    fallback = _fallback_checksum_stamp(path)
    try:
        _write_stamp(fallback, payload)
    except OSError as exc:
        print(f"Checksum valid; could not write reuse stamp {fallback}: {exc}")


def _stamp_matches(
    path: Path, checksum: str, *, use_adjacent_stamp: bool = True
) -> bool:
    stat = path.stat()
    stamps = [_fallback_checksum_stamp(path)]
    if use_adjacent_stamp:
        stamps.insert(0, _checksum_stamp(path))
    for stamp in stamps:
        try:
            (
                version,
                recorded_checksum,
                recorded_size,
                recorded_mtime,
                recorded_ctime,
                recorded_device,
                recorded_inode,
            ) = stamp.read_text(encoding="ascii").strip().split()
        except (FileNotFoundError, OSError, ValueError):
            continue
        if (
            version == "v2"
            and recorded_checksum == checksum
            and int(recorded_size) == stat.st_size
            and int(recorded_mtime) == stat.st_mtime_ns
            and int(recorded_ctime) == stat.st_ctime_ns
            and int(recorded_device) == stat.st_dev
            and int(recorded_inode) == stat.st_ino
        ):
            return True
    return False


def _metadata_reuse_is_safe(stat: os.stat_result) -> bool:
    newest_change = max(stat.st_mtime_ns, stat.st_ctime_ns)
    return time.time_ns() - newest_change >= _METADATA_SETTLE_NS


def _move_checksum_stamp(source: Path, destination: Path) -> None:
    source_stamp = _checksum_stamp(source)
    if source_stamp.exists():
        os.replace(source_stamp, _checksum_stamp(destination))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(
    path: Path,
    *,
    minimum_size: int = 1,
    expected_sha256: str | None = None,
    use_adjacent_stamp: bool = True,
) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required asset file is missing: {path}")
    size = path.stat().st_size
    if size < minimum_size:
        raise RuntimeError(
            f"Asset file is truncated: {path} has {size} bytes; expected at least {minimum_size}"
        )
    if expected_sha256 is not None:
        stat = path.stat()
        process_key = (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
            expected_sha256,
        )
        reuse_is_safe = _metadata_reuse_is_safe(stat)
        if reuse_is_safe and process_key in _PROCESS_VALIDATIONS:
            return
        if reuse_is_safe and _stamp_matches(
            path,
            expected_sha256,
            use_adjacent_stamp=use_adjacent_stamp,
        ):
            _PROCESS_VALIDATIONS.add(process_key)
            return
        actual = sha256_file(path)
        after_hash = path.stat()
        if (
            after_hash.st_size,
            after_hash.st_mtime_ns,
            after_hash.st_ctime_ns,
            after_hash.st_dev,
            after_hash.st_ino,
        ) != (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        ):
            raise RuntimeError(f"Asset changed while its checksum was being validated: {path}")
        if actual != expected_sha256:
            raise RuntimeError(
                f"Asset SHA256 mismatch for {path}: {actual}; expected {expected_sha256}"
            )
        _write_checksum_stamp(
            path,
            actual,
            use_adjacent_stamp=use_adjacent_stamp,
        )
        _PROCESS_VALIDATIONS.add(process_key)


def _response_status(response) -> int | None:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    try:
        value = response.getcode()
    except AttributeError:
        return None
    return int(value) if value is not None else None


def _response_header(response, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _parse_content_range(
    value: str | None,
) -> tuple[int | None, int | None, int | None] | None:
    if not value:
        return None
    unit, separator, remainder = value.partition(" ")
    if not separator or unit.lower() != "bytes":
        return None
    byte_range, separator, total_text = remainder.partition("/")
    if not separator:
        return None
    try:
        total = None if total_text == "*" else int(total_text)
    except ValueError:
        return None
    if byte_range == "*":
        return None, None, total
    start, separator, _end = byte_range.partition("-")
    if not separator:
        return None
    try:
        return int(start), int(_end), total
    except ValueError:
        return None


def _response_content_length(response) -> int | None:
    value = _response_header(response, "Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid HTTP Content-Length: {value!r}") from exc
    if length < 0:
        raise RuntimeError(f"Invalid HTTP Content-Length: {value!r}")
    return length


def _download_url_unlocked(
    url: str,
    destination: str | os.PathLike[str],
    *,
    minimum_size: int = 1,
    expected_sha256: str | None = None,
    dry_run: bool = False,
    attempts: int = 3,
    timeout_seconds: float = 30.0,
) -> Path:
    """Download atomically while retaining ``.part`` bytes for a later retry."""
    dst = Path(destination)
    if dst.exists():
        try:
            validate_file(
                dst,
                minimum_size=minimum_size,
                expected_sha256=expected_sha256,
            )
        except RuntimeError:
            if dry_run:
                print(f"+ download {url} -> {dst}")
                return dst
            part = Path(str(dst) + ".part")
            if dst.stat().st_size < minimum_size:
                if not part.exists() or dst.stat().st_size > part.stat().st_size:
                    os.replace(dst, part)
                    _move_checksum_stamp(dst, part)
                else:
                    dst.unlink()
                    _checksum_stamp(dst).unlink(missing_ok=True)
                print(f"Resuming truncated asset from {part}")
            else:
                invalid = Path(str(dst) + ".invalid")
                invalid.unlink(missing_ok=True)
                os.replace(dst, invalid)
                _checksum_stamp(dst).unlink(missing_ok=True)
                print(f"Asset checksum failed; retained invalid file at {invalid}")
        else:
            print(f"Reusing asset: {dst}")
            return dst
    if dry_run:
        print(f"+ download {url} -> {dst}")
        return dst
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dst) + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        retry_after: float | None = None
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Astrolabe-bootstrap/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(
            f"+ download {url} -> {dst} "
            f"(attempt {attempt}/{attempts}, resume_bytes={offset})"
        )
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                status = _response_status(response)
                append = bool(offset and status == 206)
                content_range = _response_header(response, "Content-Range")
                content_length = _response_content_length(response)
                range_values = _parse_content_range(content_range)
                expected_received = content_length
                expected_total = None
                if status == 206:
                    if range_values is None:
                        raise RuntimeError(
                            f"HTTP 206 response has invalid Content-Range: {content_range!r}"
                        )
                    range_start, range_end, expected_total = range_values
                    if (
                        range_start != offset
                        or range_end is None
                        or range_end < range_start
                        or expected_total is None
                        or range_end != expected_total - 1
                    ):
                        raise RuntimeError(
                            "Range response does not contain the requested complete remainder: "
                            f"requested offset {offset}, Content-Range={content_range!r}"
                        )
                    expected_received = range_end - range_start + 1
                    if (
                        content_length is not None
                        and content_length != expected_received
                    ):
                        raise RuntimeError(
                            "HTTP Content-Length disagrees with Content-Range: "
                            f"length={content_length}, Content-Range={content_range!r}"
                        )
                mode = "ab" if append else "wb"
                transfer = "resume" if append else ("restart" if offset else "fresh")
                print(
                    f"+ response status={status} transfer={transfer} "
                    f"content_range={content_range or '-'}"
                )
                starting_size = offset if append else 0
                with part.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            received = part.stat().st_size - starting_size
            if expected_received is not None and received != expected_received:
                last_error = RuntimeError(
                    "HTTP response body is incomplete; retained partial file: "
                    f"expected {expected_received} bytes, received {received}"
                )
            else:
                if expected_total is not None and part.stat().st_size != expected_total:
                    last_error = RuntimeError(
                        "HTTP range transfer ended at the wrong total size; retained partial "
                        f"file: expected {expected_total} bytes, got {part.stat().st_size}"
                    )
                else:
                    try:
                        validate_file(
                            part,
                            minimum_size=minimum_size,
                            expected_sha256=expected_sha256,
                        )
                    except RuntimeError as exc:
                        if not part.exists() or part.stat().st_size >= minimum_size:
                            raise
                        last_error = exc
                    else:
                        os.replace(part, dst)
                        _move_checksum_stamp(part, dst)
                        return dst
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():
                content_range = _response_header(exc, "Content-Range")
                range_values = _parse_content_range(content_range)
                total = range_values[2] if range_values is not None else None
                part_size = part.stat().st_size
                complete_by_length = total is not None and part_size == total
                try:
                    if not complete_by_length:
                        raise RuntimeError(
                            "HTTP 416 did not prove that the retained partial file is complete: "
                            f"Content-Range={content_range!r}, part_size={part_size}"
                        )
                    validate_file(
                        part,
                        minimum_size=minimum_size,
                        expected_sha256=expected_sha256,
                    )
                except RuntimeError as validation_error:
                    invalid = Path(str(part) + ".invalid")
                    invalid.unlink(missing_ok=True)
                    os.replace(part, invalid)
                    last_error = validation_error
                    print(
                        "+ response status=416 action=quarantine-restart "
                        f"content_range={content_range or '-'} part_size={part_size} "
                        f"invalid_path={invalid}"
                    )
                    if attempt < attempts:
                        print(
                            f"Range resume was invalid; restarting without partial data: {last_error}"
                        )
                        continue
                    raise last_error
                else:
                    print(
                        "+ response status=416 action=promote-complete-part "
                        f"content_range={content_range or '-'} part_size={part_size}"
                    )
                    os.replace(part, dst)
                    _move_checksum_stamp(part, dst)
                    return dst
            else:
                last_error = exc
            header_value = exc.headers.get("Retry-After") if exc.headers else None
            if header_value is not None:
                try:
                    retry_after = max(0.0, float(header_value))
                except ValueError:
                    retry_after = None
            if exc.code not in {408, 429} and not 500 <= exc.code < 600:
                raise RuntimeError(
                    f"Permanent HTTP error downloading {url}; partial data remains at {part}: {exc}"
                ) from exc
        except RuntimeError:
            raise
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, OSError) and reason.errno in {
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.EACCES,
            }:
                raise RuntimeError(f"Permanent network error downloading {url}: {exc}") from exc
            last_error = exc
        except OSError as exc:
            last_error = exc
        if attempt < attempts:
            delay = retry_after if retry_after is not None else min(2 ** (attempt - 1), 30)
            print(
                f"Download failed; retained {part} for retry in {delay:g}s: {last_error}"
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(
        f"Failed to download {url} after {attempts} attempts; partial data remains at {part}: "
        f"{last_error}"
    ) from last_error


def download_url(
    url: str,
    destination: str | os.PathLike[str],
    *,
    minimum_size: int = 1,
    expected_sha256: str | None = None,
    dry_run: bool = False,
    attempts: int | None = None,
    timeout_seconds: float = 30.0,
) -> Path:
    """Download with a per-target process lock and resumable atomic writes."""
    dst = Path(destination)
    if attempts is None:
        attempts = _attempt_budget(minimum_size)
    if dry_run:
        return _download_url_unlocked(
            url,
            dst,
            minimum_size=minimum_size,
            expected_sha256=expected_sha256,
            dry_run=True,
            attempts=attempts,
            timeout_seconds=timeout_seconds,
        )
    if dst.exists():
        try:
            validate_file(
                dst,
                minimum_size=minimum_size,
                expected_sha256=expected_sha256,
            )
        except RuntimeError:
            pass
        else:
            print(f"Reusing asset: {dst}")
            return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(dst) + ".lock")
    with lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _download_url_unlocked(
            url,
            dst,
            minimum_size=minimum_size,
            expected_sha256=expected_sha256,
            dry_run=False,
            attempts=attempts,
            timeout_seconds=timeout_seconds,
        )
