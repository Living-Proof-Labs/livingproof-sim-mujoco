# SPDX-License-Identifier: Apache-2.0
"""Review probes for #45 / #44, turned into tests. Safe to drop under tests/.

Covers gaps relative to tests/test_pins.py at 0a0650e: a truncated body with no
Content-Length, cache preservation on SizeMismatch, an unusable Content-Length
header, a matching cache that must not be re-downloaded, and two documented
limits (a correct body with a wrong Content-Length is retried; a pin that omits
bytes still treats a short body as ChecksumError when Content-Length is absent).
"""

import hashlib
import json
import logging
import os
from email.message import EmailMessage
from pathlib import Path

import pytest

from livingproof_sim_mujoco import pins

URL = "https://raw.githubusercontent.com/o/r/c/a.txt"
PAYLOAD = b"pinned"


def _files(root: Path) -> list[str]:
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            found.append(str(Path(dirpath, name).relative_to(root)))
    return sorted(found)


def _write_pins(root: Path, digest: str, fields: dict | None = None) -> None:
    entry = {"sha256": digest}
    if fields:
        entry.update(fields)
    document = {
        "groups": {
            "g": {
                "repository": "o/r",
                "commit": "c",
                "dest": ".cache/g",
                "files": {"a.txt": entry},
            }
        }
    }
    (root / "pins.json").write_text(json.dumps(document))


class _Response:
    def __init__(self, data: bytes, content_length: int | str | None = None):
        self.data = data
        self.status = 200
        self.headers = EmailMessage()
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, _size=-1):
        data, self.data = self.data, b""
        return data


@pytest.fixture
def clock(monkeypatch):
    sleeps = []
    spans = []

    def uniform(low, high):
        spans.append((low, high))
        return 0.0

    monkeypatch.setattr(pins.time, "sleep", sleeps.append)
    monkeypatch.setattr(pins.random, "uniform", uniform)
    return sleeps, spans


def test_fetch_retries_truncated_body_with_no_content_length_using_pin_bytes(
    tmp_path, clock, caplog
):
    """#44 can arrive with no Content-Length. The pin's bytes must still retry it."""
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest, {"bytes": len(PAYLOAD)})
    sleeps, spans = clock
    calls = []
    dest = tmp_path / ".cache/g/a.txt"
    short = PAYLOAD[:3]

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            return _Response(short, content_length=None)
        return _Response(PAYLOAD, content_length=None)

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    assert pins.fetch(tmp_path, opener) == [dest]
    assert dest.read_bytes() == PAYLOAD
    assert calls == [URL, URL]
    assert sleeps == [10.0]
    assert spans == [(-2.0, 2.0)]
    warning = caplog.records[0].getMessage()
    assert f"{len(PAYLOAD)} bytes (pin)" in warning
    assert f"downloaded {len(short)} bytes" in warning
    assert "Content-Length" not in warning


def test_fetch_leaves_cached_file_in_place_on_size_mismatch(tmp_path, clock):
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest, {"bytes": len(PAYLOAD)})
    stale = b"stale-cache"
    dest = tmp_path / ".cache/g/a.txt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(stale)
    sleeps, _spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        return _Response(b"no", content_length=len(PAYLOAD))

    with pytest.raises(pins.DownloadError, match="failed after 5 attempts") as raised:
        pins.fetch(tmp_path, opener)
    assert isinstance(raised.value.__cause__, pins.SizeMismatch)
    assert dest.read_bytes() == stale
    assert len(calls) == pins.MAX_ATTEMPTS
    assert sleeps == [10.0, 20.0, 40.0, 80.0]
    assert ".part" not in " ".join(_files(tmp_path))


def test_fetch_does_not_redownload_a_matching_cache(tmp_path, clock):
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest, {"bytes": len(PAYLOAD)})
    dest = tmp_path / ".cache/g/a.txt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD)
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        raise AssertionError("opener should not run for a cache hit")

    assert pins.fetch(tmp_path, opener) == [dest]
    assert dest.read_bytes() == PAYLOAD
    assert calls == []
    assert sleeps == []
    assert spans == []


def test_fetch_ignores_unusable_content_length_and_uses_pin_bytes(tmp_path, clock, caplog):
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest, {"bytes": len(PAYLOAD)})
    sleeps, _spans = clock
    calls = []
    dest = tmp_path / ".cache/g/a.txt"

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            return _Response(PAYLOAD[:3], content_length="not-a-number")
        return _Response(PAYLOAD, content_length=len(PAYLOAD))

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    assert pins.fetch(tmp_path, opener) == [dest]
    assert dest.read_bytes() == PAYLOAD
    assert calls == [URL, URL]
    assert sleeps == [10.0]
    messages = [record.getMessage() for record in caplog.records]
    assert any("ignoring unusable Content-Length" in message for message in messages)
    assert any(f"{len(PAYLOAD)} bytes (pin)" in message for message in messages)


def test_fetch_retries_when_content_length_does_not_match_a_full_pin_body(tmp_path, clock):
    """Current behaviour: a body of the pin size is still retried if Content-Length differs.

    The sha256 is not consulted. After five attempts this is DownloadError, not
    ChecksumError. A tighter policy would checksum when the pin's bytes match.
    """
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest, {"bytes": len(PAYLOAD)})
    sleeps, _spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        return _Response(PAYLOAD, content_length=len(PAYLOAD) + 1)

    with pytest.raises(pins.DownloadError, match="failed after 5 attempts") as raised:
        pins.fetch(tmp_path, opener)
    message = str(raised.value)
    assert isinstance(raised.value.__cause__, pins.SizeMismatch)
    assert f"{len(PAYLOAD) + 1} bytes (Content-Length)" in message
    assert f"downloaded {len(PAYLOAD)} bytes" in message
    assert "pin" not in message.split("downloaded")[0]
    assert len(calls) == pins.MAX_ATTEMPTS
    assert sleeps == [10.0, 20.0, 40.0, 80.0]
    assert _files(tmp_path) == ["pins.json"]


def test_fetch_without_pin_bytes_or_content_length_does_not_retry_a_short_body(tmp_path, clock):
    """Residual #44 path when a pin omits bytes and the response has no Content-Length."""
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest)
    sleeps, spans = clock
    calls = []
    short = PAYLOAD[:3]
    assert hashlib.sha256(short).hexdigest() != digest

    def opener(url):
        calls.append(url)
        return _Response(short, content_length=None)

    with pytest.raises(pins.ChecksumError, match="expected " + digest):
        pins.fetch(tmp_path, opener)
    assert calls == [URL]
    assert sleeps == []
    assert spans == []
    assert _files(tmp_path) == ["pins.json"]
