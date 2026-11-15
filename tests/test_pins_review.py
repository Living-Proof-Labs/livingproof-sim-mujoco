# SPDX-License-Identifier: Apache-2.0
"""Review probes for #45 / #44, turned into tests. Safe to drop under tests/.

Covers gaps relative to tests/test_pins.py at 0a0650e: a truncated body with no
Content-Length, cache preservation on SizeMismatch, an unusable Content-Length
header, a matching cache that must not be re-downloaded, and two documented
limits that #43 changes: a pin-sized body with a wrong Content-Length is
checksummed, and a pin that omits bytes is refused by name before download.
"""

import hashlib
import json
import logging
import os
import urllib.request
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


def test_fetch_checksums_a_pin_sized_body_when_content_length_is_wrong(tmp_path, clock, caplog):
    """A body whose length matches the pin is hashed even when Content-Length disagrees.

    Replaces the #45 characterization ``test_fetch_retries_when_content_length_does_not_match_a_full_pin_body``,
    which expected five retries and then DownloadError without hashing.
    """
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest, {"bytes": len(PAYLOAD)})
    sleeps, spans = clock
    calls = []
    dest = tmp_path / ".cache/g/a.txt"

    def opener(url):
        calls.append(url)
        return _Response(PAYLOAD, content_length=len(PAYLOAD) + 1)

    caplog.set_level(logging.WARNING, logger="livingproof_sim_mujoco.pins")
    assert pins.fetch(tmp_path, opener) == [dest]
    assert dest.read_bytes() == PAYLOAD
    assert calls == [URL]
    assert sleeps == []
    assert spans == []
    warning = " ".join(record.getMessage() for record in caplog.records)
    assert "Content-Length" in warning
    assert str(len(PAYLOAD) + 1) in warning
    assert f"downloaded {len(PAYLOAD)} bytes" in warning or str(len(PAYLOAD)) in warning
    assert _files(tmp_path) == [".cache/g/a.txt", "pins.json"]


def test_fetch_still_requires_sha256_when_content_length_mismatches_pin_bytes(tmp_path, clock):
    """A wrong sha256 is still ChecksumError, with no retry, when only Content-Length disagrees."""
    _write_pins(tmp_path, "0" * 64, {"bytes": len(PAYLOAD)})
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        return _Response(PAYLOAD, content_length=len(PAYLOAD) + 1)

    with pytest.raises(pins.ChecksumError, match="expected " + "0" * 64):
        pins.fetch(tmp_path, opener)
    assert calls == [URL]
    assert sleeps == []
    assert spans == []
    assert _files(tmp_path) == ["pins.json"]


def test_fetch_requires_pin_bytes_and_names_the_pin(tmp_path, clock):
    """A pin without bytes names that pin and does not download.

    Replaces the #45 characterization
    ``test_fetch_without_pin_bytes_or_content_length_does_not_retry_a_short_body``,
    which expected ChecksumError with no retry for a short body.
    """
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    _write_pins(tmp_path, digest)
    sleeps, spans = clock
    calls = []

    def opener(url):
        calls.append(url)
        return _Response(PAYLOAD[:3], content_length=None)

    with pytest.raises(ValueError, match=r"g:a\.txt") as raised:
        pins.fetch(tmp_path, opener)
    message = str(raised.value)
    assert "bytes" in message
    assert calls == []
    assert sleeps == []
    assert spans == []
    assert _files(tmp_path) == ["pins.json"]


# Published robot pins named in #50. Every file entry with a sha256 must have
# bytes. Skips unless LIVINGPROOF_NETWORK=1, so CI stays offline.
PUBLISHED_ROBOT_PINS = (
    ("livingproof-unitree-go1", "v1.0.0"),
    ("livingproof-unitree-go1", "v1.0.3"),
    ("livingproof-unitree-go1", "v2.0.0"),
    ("livingproof-unitree-g1", "v1.0.0"),
    ("livingproof-unitree-g1", "v1.0.4"),
    ("livingproof-unitree-g1", "v2.0.0"),
    ("livingproof-unitree-g1", "v2.0.1"),
    ("livingproof-booster-t1", "v1.0.0"),
    ("livingproof-booster-t1", "v1.0.3"),
    ("livingproof-booster-t1", "v2.0.0"),
)


def _published_pin_file_entries(obj, prefix=""):
    if isinstance(obj, dict):
        if "sha256" in obj:
            yield prefix, obj
            return
        for key, value in obj.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            yield from _published_pin_file_entries(value, path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _published_pin_file_entries(value, f"{prefix}[{index}]")


@pytest.mark.skipif(
    os.environ.get("LIVINGPROOF_NETWORK") != "1",
    reason="Set LIVINGPROOF_NETWORK=1 to fetch published pins.json files.",
)
@pytest.mark.parametrize("repo,tag", PUBLISHED_ROBOT_PINS)
def test_published_robot_pins_include_bytes_on_every_file(repo, tag):
    url = f"https://raw.githubusercontent.com/Living-Proof-Labs/{repo}/{tag}/pins.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    entries = list(_published_pin_file_entries(data))
    assert entries, f"{repo} {tag}: no sha256 file entries"
    missing = [path for path, entry in entries if entry.get("bytes") is None]
    assert missing == [], f"{repo} {tag} missing bytes: {missing}"
    for path, entry in entries:
        assert isinstance(entry["bytes"], int) and entry["bytes"] >= 0, (repo, tag, path, entry)
