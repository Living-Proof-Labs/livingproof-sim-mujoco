# SPDX-License-Identifier: Apache-2.0
"""Retry transient pin downloads. Hash mismatches still fail immediately."""

import hashlib
import json
import logging
import os
import urllib.error
from datetime import datetime, timezone
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


def _write_pins(root: Path, digest: str) -> None:
    document = {
        "groups": {
            "g": {
                "repository": "o/r",
                "commit": "c",
                "dest": ".cache/g",
                "files": {"a.txt": {"sha256": digest}},
            }
        }
    }
    (root / "pins.json").write_text(json.dumps(document))


class _Body:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, _size=-1):
        data, self.data = self.data, b""
        return data


class _PartialReset:
    """Return a few bytes, then fail the way a reset mid-download does."""

    def __init__(self):
        self.calls = 0

    def read(self, _size=-1):
        self.calls += 1
        if self.calls == 1:
            return b"partial"
        raise ConnectionResetError(104, "Connection reset by peer")


def _http_error(url: str, code: int, retry_after: str | None = None):
    headers = EmailMessage()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, f"status {code}", headers, None)


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


def test_fetch_retries_resets_and_503_then_verifies(tmp_path, clock, caplog):
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest)
    sleeps, spans = clock
    calls = []
    dest = tmp_path / ".cache/g/a.txt"

    def opener(url):
        calls.append(url)
        attempt = len(calls)
        if attempt == 3:
            parts = [path for path in dest.parent.iterdir() if path.name.endswith(".part")]
            assert len(parts) == 1 and parts[0].stat().st_size == 0
            assert not dest.exists()
        if attempt == 1:
            raise urllib.error.URLError("Connection reset by peer")
        if attempt == 2:
            return _PartialReset()
        if attempt == 3:
            raise _http_error(url, 503)
        return _Body(PAYLOAD)

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    assert pins.fetch(tmp_path, opener) == [dest]
    assert dest.read_bytes() == PAYLOAD
    assert calls == [URL, URL, URL, URL]
    assert sleeps == [10.0, 20.0, 40.0]
    assert spans == [(-2.0, 2.0), (-4.0, 4.0), (-8.0, 8.0)]
    assert _files(tmp_path) == [".cache/g/a.txt", "pins.json"]
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 3
    assert all(URL in message and "retrying in" in message for message in warnings)
    assert "Connection reset by peer" in warnings[0]
    assert "503" in warnings[2]


def test_fetch_honours_retry_after_on_429(tmp_path, clock):
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            raise _http_error(url, 429, retry_after="7")
        return _Body(PAYLOAD)

    assert pins.fetch(tmp_path, opener) == [tmp_path / ".cache/g/a.txt"]
    assert calls == [URL, URL]
    assert sleeps == [7.0]
    assert spans == []


def test_fetch_honours_an_http_date_retry_after(tmp_path, clock, monkeypatch):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(pins, "datetime", Frozen)
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            raise _http_error(url, 503, retry_after="Wed, 11 Nov 2026 12:00:09 GMT")
        return _Body(PAYLOAD)

    assert pins.fetch(tmp_path, opener) == [tmp_path / ".cache/g/a.txt"]
    assert sleeps == [9.0]
    assert spans == []


def test_fetch_caps_a_huge_retry_after(tmp_path, clock, caplog):
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            raise _http_error(url, 429, retry_after="999999")
        return _Body(PAYLOAD)

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    assert pins.fetch(tmp_path, opener) == [tmp_path / ".cache/g/a.txt"]
    assert calls == [URL, URL]
    assert sleeps == [float(pins.MAX_RETRY_AFTER_SECONDS)]
    assert pins.MAX_RETRY_AFTER_SECONDS == 120
    assert spans == []
    assert any(
        "999999" in record.getMessage() and "cap" in record.getMessage()
        for record in caplog.records
    )


def test_fetch_falls_back_when_retry_after_is_unparseable(tmp_path, clock, caplog):
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            raise _http_error(url, 429, retry_after="soon")
        return _Body(PAYLOAD)

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    assert pins.fetch(tmp_path, opener) == [tmp_path / ".cache/g/a.txt"]
    assert (tmp_path / ".cache/g/a.txt").read_bytes() == PAYLOAD
    assert calls == [URL, URL]
    assert sleeps == [10.0]
    assert spans == [(-2.0, 2.0)]
    assert any(
        "soon" in record.getMessage() and "unparseable" in record.getMessage()
        for record in caplog.records
    )


def test_fetch_raises_after_the_last_transient_failure(tmp_path, clock, caplog):
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    with pytest.raises(pins.DownloadError, match="failed after 5 attempts") as raised:
        pins.fetch(tmp_path, opener)
    message = str(raised.value)
    assert URL in message
    assert "Connection reset by peer" in message
    assert len(calls) == 5
    assert sleeps == [10.0, 20.0, 40.0, 80.0]
    assert spans == [(-2.0, 2.0), (-4.0, 4.0), (-8.0, 8.0), (-16.0, 16.0)]
    assert len(caplog.records) == 5
    assert all(URL in record.getMessage() for record in caplog.records)
    assert _files(tmp_path) == ["pins.json"]


def test_fetch_does_not_retry_404(tmp_path, clock, caplog):
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, _spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        raise _http_error(url, 404)

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    with pytest.raises(urllib.error.HTTPError) as raised:
        pins.fetch(tmp_path, opener)
    assert raised.value.code == 404
    assert calls == [URL]
    assert sleeps == []
    assert len(caplog.records) == 1
    assert URL in caplog.records[0].getMessage()
    assert "retrying" not in caplog.records[0].getMessage()
    assert _files(tmp_path) == ["pins.json"]


def test_fetch_does_not_retry_a_hash_mismatch_or_leave_a_file(tmp_path, clock):
    _write_pins(tmp_path, "0" * 64)
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        return _Body(PAYLOAD)

    with pytest.raises(pins.ChecksumError, match="expected " + "0" * 64):
        pins.fetch(tmp_path, opener)
    assert calls == [URL]
    assert sleeps == []
    assert spans == []
    assert _files(tmp_path) == ["pins.json"]


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
        TimeoutError("timed out"),
    ],
)
def test_fetch_retries_refused_connections_and_timeouts(tmp_path, clock, failure):
    _write_pins(tmp_path, hashlib.sha256(PAYLOAD).hexdigest())
    sleeps, _spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        if len(calls) == 1:
            raise failure
        return _Body(PAYLOAD)

    assert pins.fetch(tmp_path, opener) == [tmp_path / ".cache/g/a.txt"]
    assert len(calls) == 2
    assert sleeps == [10.0]
