from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RANGE_STATE_SCHEMA = "crossres-http-range-download-v1"
PART_META_SCHEMA = "crossres-http-range-part-v1"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


@dataclass(frozen=True, order=True)
class ByteRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid byte range {self.start}-{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def stem(self) -> str:
        return f"{self.start:020d}-{self.end:020d}"

    def as_json(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "bytes": self.length}


def plan_ranges(start: int, total_bytes: int, part_bytes: int) -> list[ByteRange]:
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if start < 0 or start > total_bytes:
        raise ValueError("start must be in [0, total_bytes]")
    if part_bytes <= 0:
        raise ValueError("part_bytes must be positive")
    return [
        ByteRange(offset, min(total_bytes - 1, offset + part_bytes - 1))
        for offset in range(start, total_bytes, part_bytes)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _part_paths(state_dir: Path, byte_range: ByteRange) -> tuple[Path, Path, Path]:
    part = state_dir / f"{byte_range.stem}.part"
    return part, part.with_suffix(".part.json"), part.with_suffix(".part.tmp")


def _part_metadata(byte_range: ByteRange, path: Path) -> dict[str, Any]:
    return {
        "schema": PART_META_SCHEMA,
        **byte_range.as_json(),
        "sha256": _sha256(path),
    }


def _part_is_valid(state_dir: Path, byte_range: ByteRange) -> bool:
    part, metadata_path, _ = _part_paths(state_dir, byte_range)
    if not part.is_file() or part.stat().st_size != byte_range.length:
        return False
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "schema": PART_META_SCHEMA,
        **byte_range.as_json(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    digest = metadata.get("sha256")
    return isinstance(digest, str) and digest == _sha256(part)


def _finalize_part(
    state_dir: Path, byte_range: ByteRange, source: Path
) -> dict[str, Any]:
    part, metadata_path, _ = _part_paths(state_dir, byte_range)
    if source.stat().st_size != byte_range.length:
        raise ValueError(
            f"{source}: {source.stat().st_size} bytes != {byte_range.length}"
        )
    metadata = _part_metadata(byte_range, source)
    if source != part:
        os.replace(source, part)
    _atomic_json(metadata_path, metadata)
    return metadata


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    match = _CONTENT_RANGE.fullmatch(value or "")
    if match is None:
        raise ValueError(f"invalid Content-Range response: {value!r}")
    return tuple(int(group) for group in match.groups())


def _append_http_range(
    *,
    url: str,
    temporary: Path,
    byte_range: ByteRange,
    total_bytes: int,
    timeout_seconds: float,
) -> int:
    current = temporary.stat().st_size if temporary.exists() else 0
    if current < 0 or current >= byte_range.length:
        raise ValueError(f"{temporary}: invalid partial length {current}")
    request_start = byte_range.start + current
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={request_start}-{byte_range.end}",
        "User-Agent": "vesuvius-crossres-range-mirror/1",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise ValueError(f"server ignored byte range: HTTP {status}")
        response_start, response_end, response_total = _parse_content_range(
            response.headers.get("Content-Range")
        )
        if (
            response_start != request_start
            or response_end != byte_range.end
            or response_total != total_bytes
        ):
            raise ValueError(
                "server returned the wrong byte range: "
                f"{response_start}-{response_end}/{response_total}, expected "
                f"{request_start}-{byte_range.end}/{total_bytes}"
            )
        expected_response_bytes = byte_range.end - request_start + 1
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected_response_bytes:
            raise ValueError(
                f"invalid Content-Length {content_length}; "
                f"expected {expected_response_bytes}"
            )
        mode = "ab" if current else "wb"
        with temporary.open(mode) as stream:
            bytes_since_sync = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                bytes_since_sync += len(block)
                if stream.tell() > byte_range.length:
                    raise ValueError("range response exceeded its declared boundary")
                if bytes_since_sync >= 32 * 1024 * 1024:
                    stream.flush()
                    os.fsync(stream.fileno())
                    bytes_since_sync = 0
            stream.flush()
            os.fsync(stream.fileno())
    return temporary.stat().st_size


def _download_part(
    *,
    url: str,
    state_dir: Path,
    byte_range: ByteRange,
    total_bytes: int,
    retries: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    part, metadata_path, temporary = _part_paths(state_dir, byte_range)
    if _part_is_valid(state_dir, byte_range):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    if not metadata_path.exists():
        if part.is_file() and part.stat().st_size == byte_range.length:
            return _finalize_part(state_dir, byte_range, part)
        if temporary.is_file() and temporary.stat().st_size == byte_range.length:
            return _finalize_part(state_dir, byte_range, temporary)

    if temporary.exists() and temporary.stat().st_size > byte_range.length:
        temporary.unlink()
    if metadata_path.exists() and part.exists():
        temporary.unlink(missing_ok=True)

    failures = 0
    previous_size = temporary.stat().st_size if temporary.exists() else 0
    while previous_size < byte_range.length:
        try:
            current_size = _append_http_range(
                url=url,
                temporary=temporary,
                byte_range=byte_range,
                total_bytes=total_bytes,
                timeout_seconds=timeout_seconds,
            )
            if current_size <= previous_size:
                raise OSError("range request made no forward progress")
            previous_size = current_size
            failures = 0
        except (
            OSError,
            TimeoutError,
            ValueError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            failures += 1
            if failures >= retries:
                raise RuntimeError(
                    f"failed byte range {byte_range.start}-{byte_range.end} "
                    f"after {failures} consecutive attempts"
                ) from error
            time.sleep(min(30.0, 1.5 * failures))
            previous_size = temporary.stat().st_size if temporary.exists() else 0
    return _finalize_part(state_dir, byte_range, temporary)


def _zip_member_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return len(archive.infolist())


def _assemble(
    *,
    output: Path,
    state_dir: Path,
    prefix_bytes: int,
    ranges: list[ByteRange],
    expected_bytes: int,
    verify_zip: bool,
) -> tuple[str, int | None]:
    temporary = state_dir / "assembled.tmp"
    digest = hashlib.sha256()
    written = 0
    with temporary.open("wb") as destination:
        sources: list[Path] = []
        if prefix_bytes:
            sources.append(output)
        sources.extend(_part_paths(state_dir, item)[0] for item in ranges)
        for source in sources:
            with source.open("rb") as stream:
                while True:
                    block = stream.read(8 * 1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
                    written += len(block)
        destination.flush()
        os.fsync(destination.fileno())
    if written != expected_bytes:
        raise ValueError(f"assembled {written} bytes, expected {expected_bytes}")
    members = _zip_member_count(temporary) if verify_zip else None
    os.replace(temporary, output)
    return digest.hexdigest(), members


def download_ranged_file(
    *,
    url: str,
    output: str | Path,
    expected_bytes: int,
    state_dir: str | Path | None = None,
    part_bytes: int = 256 * 1024 * 1024,
    workers: int = 8,
    retries: int = 20,
    timeout_seconds: float = 180.0,
    verify_zip: bool = False,
) -> dict[str, Any]:
    """Resume a large immutable HTTP object with checked parallel ranges."""

    if not url.startswith(("http://", "https://")):
        raise ValueError("url must use HTTP or HTTPS")
    if expected_bytes <= 0:
        raise ValueError("expected_bytes must be positive")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    if retries <= 0 or timeout_seconds <= 0:
        raise ValueError("retries and timeout_seconds must be positive")

    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    actual_bytes = target.stat().st_size if target.exists() else 0
    if actual_bytes > expected_bytes:
        raise ValueError(
            f"{target}: {actual_bytes} bytes exceeds expected {expected_bytes}"
        )

    root = (
        Path(state_dir).expanduser().resolve()
        if state_dir is not None
        else target.with_name(target.name + ".ranges")
    )
    state_path = root / "state.json"
    if actual_bytes == expected_bytes:
        members = _zip_member_count(target) if verify_zip else None
        result = {
            "schema": RANGE_STATE_SCHEMA,
            "state": "complete",
            "url": url,
            "output": str(target),
            "expected_bytes": expected_bytes,
            "zip_members": members,
            "already_complete": True,
        }
        if state_path.is_file():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            expected_identity = {
                "schema": RANGE_STATE_SCHEMA,
                "url": url,
                "output": str(target),
                "expected_bytes": expected_bytes,
            }
            for key, value in expected_identity.items():
                if previous.get(key) != value:
                    raise ValueError(
                        f"{state_path}: identity mismatch for {key}: "
                        f"{previous.get(key)!r} != {value!r}"
                    )
            previous.update(result)
            _atomic_json(state_path, previous)
        return result

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected_identity = {
            "schema": RANGE_STATE_SCHEMA,
            "url": url,
            "output": str(target),
            "expected_bytes": expected_bytes,
            "part_bytes": part_bytes,
        }
        for key, value in expected_identity.items():
            if state.get(key) != value:
                raise ValueError(
                    f"{state_path}: identity mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}"
                )
        prefix_bytes = int(state["prefix_bytes"])
        if actual_bytes != prefix_bytes:
            raise ValueError(
                f"{target}: prefix changed from {prefix_bytes} to {actual_bytes} bytes"
            )
        prefix_sha256 = (
            _sha256(target) if prefix_bytes else hashlib.sha256(b"").hexdigest()
        )
        if prefix_sha256 != state["prefix_sha256"]:
            raise ValueError(f"{target}: immutable prefix digest changed")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"{root}: non-empty range directory has no state.json")
        root.mkdir(parents=True, exist_ok=True)
        prefix_bytes = actual_bytes
        prefix_sha256 = (
            _sha256(target) if prefix_bytes else hashlib.sha256(b"").hexdigest()
        )
        state = {
            "schema": RANGE_STATE_SCHEMA,
            "state": "planned",
            "url": url,
            "output": str(target),
            "expected_bytes": expected_bytes,
            "part_bytes": part_bytes,
            "workers": workers,
            "prefix_bytes": prefix_bytes,
            "prefix_sha256": prefix_sha256,
            "ranges": [
                item.as_json()
                for item in plan_ranges(prefix_bytes, expected_bytes, part_bytes)
            ],
            "completed_ranges": 0,
            "completed_bytes": prefix_bytes,
        }
        _atomic_json(state_path, state)

    ranges = plan_ranges(prefix_bytes, expected_bytes, part_bytes)
    completed = [item for item in ranges if _part_is_valid(root, item)]
    completed_set = set(completed)
    pending = [item for item in ranges if item not in completed_set]
    completed_bytes = prefix_bytes + sum(item.length for item in completed)
    state.update(
        {
            "state": "downloading",
            "workers": workers,
            "completed_ranges": len(completed),
            "completed_bytes": completed_bytes,
        }
    )
    _atomic_json(state_path, state)
    print(
        f"range mirror: {len(completed)}/{len(ranges)} parts, "
        f"{completed_bytes:,}/{expected_bytes:,} bytes",
        flush=True,
    )

    started = time.monotonic()
    session_start_bytes = completed_bytes
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_part,
                url=url,
                state_dir=root,
                byte_range=item,
                total_bytes=expected_bytes,
                retries=retries,
                timeout_seconds=timeout_seconds,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            item = futures[future]
            future.result()
            completed_bytes += item.length
            completed.append(item)
            state.update(
                {
                    "completed_ranges": len(completed),
                    "completed_bytes": completed_bytes,
                }
            )
            _atomic_json(state_path, state)
            elapsed = max(time.monotonic() - started, 1.0e-6)
            rate = (completed_bytes - session_start_bytes) / elapsed / 2**20
            print(
                f"range mirror: {len(completed)}/{len(ranges)} parts, "
                f"{completed_bytes:,}/{expected_bytes:,} bytes, "
                f"{rate:.2f} MiB/s",
                flush=True,
            )

    for item in ranges:
        if not _part_is_valid(root, item):
            raise ValueError(f"range failed final integrity check: {item}")
    state["state"] = "assembling"
    _atomic_json(state_path, state)
    archive_sha256, zip_members = _assemble(
        output=target,
        state_dir=root,
        prefix_bytes=prefix_bytes,
        ranges=ranges,
        expected_bytes=expected_bytes,
        verify_zip=verify_zip,
    )
    state.update(
        {
            "state": "complete",
            "completed_ranges": len(ranges),
            "completed_bytes": expected_bytes,
            "archive_sha256": archive_sha256,
            "zip_members": zip_members,
        }
    )
    _atomic_json(state_path, state)
    provenance_path = target.with_name(target.name + ".crossres_download.json")
    _atomic_json(provenance_path, state)
    print(f"range mirror complete: {target}", flush=True)
    return state
