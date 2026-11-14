"""Pinned upstream files: download by commit into a robot package's cache, verify sha256.

A robot package keeps a ``pins.json`` (groups of files at a repository commit, each with a
sha256, an optional ``bytes`` length, and a ``dest`` directory). ``fetch(root)`` downloads
missing files and verifies every file, cached or not. Transient download errors are retried,
including a body that is short of or a different size from ``Content-Length`` or the pin's
``bytes``. A sha256 mismatch on a body of the expected size is not retried.
"""

from __future__ import annotations

import email.utils
import hashlib
import http.client
import json
import logging
import os
import random
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Five attempts. The four waits are 10s, 20s, 40s and 80s, each jittered by 20%
# (about 2-3 minutes altogether). A usable Retry-After replaces that wait, and is
# never longer than MAX_RETRY_AFTER_SECONDS.
MAX_ATTEMPTS = 5
MAX_RETRY_AFTER_SECONDS = 120
_BACKOFF_BASE_SECONDS = 10.0
_BACKOFF_JITTER = 0.2
_RETRY_AFTER_INTEGER = re.compile(r"-?\d+")


class ChecksumError(RuntimeError):
    pass


class DownloadError(RuntimeError):
    """A pinned file could not be downloaded after every transient attempt."""


class SizeMismatch(Exception):
    """The body length did not match Content-Length or the pin's ``bytes``.

    A short or substituted response is transient and is retried. ``ChecksumError``
    is only for a body of the expected size whose sha256 does not match.
    """

    def __init__(self, label: str, actual: int, expected: list[tuple[str, int]]):
        self.label = label
        self.actual = actual
        self.expected = expected
        counts = " and ".join(f"{count} bytes ({source})" for source, count in expected)
        super().__init__(f"{label}: expected {counts}, downloaded {actual} bytes")


def load_pins(root: Path) -> dict:
    return json.loads((Path(root) / "pins.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def url_for(group: dict, path: str, entry: dict) -> str:
    return entry.get("url") or (
        f"https://raw.githubusercontent.com/{group['repository']}/{group['commit']}/{path}"
    )


def local(group_name: str, path: str, root: Path) -> Path:
    return Path(root) / load_pins(root)["groups"][group_name]["dest"] / path


def _ignore_retry_after(url: str, text: str, reason: str) -> None:
    logger.warning(
        "ignoring Retry-After %r for %s (%s); using normal backoff",
        text,
        url,
        reason,
    )


def _retry_after_seconds(url: str, exc: BaseException) -> float | None:
    """Seconds to wait, or None when the header is absent or unusable.

    Integer seconds and HTTP-date values are accepted. A negative, blank, or
    unparseable value is ignored so the caller uses jittered backoff. Parsing
    never raises.
    """
    headers = getattr(exc, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        _ignore_retry_after(url, text, "missing")
        return None
    if _RETRY_AFTER_INTEGER.fullmatch(text):
        seconds = int(text)
        if seconds < 0:
            _ignore_retry_after(url, text, "negative")
            return None
        return float(seconds)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        when = None
    if when is None:
        _ignore_retry_after(url, text, "unparseable")
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delay = (when - datetime.now(timezone.utc)).total_seconds()
    if delay < 0:
        _ignore_retry_after(url, text, "negative")
        return None
    return delay


def _backoff_seconds(attempt: int) -> float:
    """Wait after a failed attempt (1-based) before the next try."""
    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    spread = delay * _BACKOFF_JITTER
    return delay + random.uniform(-spread, spread)


def _is_transient(exc: BaseException) -> bool:
    """Connection resets, refusals, timeouts, status-less URLErrors, HTTP 5xx and 429.

    A body whose length does not match Content-Length or the pin is the same kind of
    failure: the response ended early or was substituted, and another attempt may work.
    """
    if isinstance(exc, SizeMismatch):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        return code == 429 or (isinstance(code, int) and 500 <= code <= 599)
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, http.client.IncompleteRead))


def _close_quietly(obj) -> None:
    close = getattr(obj, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        logger.debug("closing download response failed", exc_info=True)


def _http_status(response) -> int | None:
    for name in ("status", "code"):
        value = getattr(response, name, None)
        if isinstance(value, int):
            return value
    return None


def _raise_for_status(url: str, response) -> None:
    status = _http_status(response)
    if status is None or status < 400:
        return
    raise urllib.error.HTTPError(
        url,
        status,
        str(getattr(response, "reason", "") or "HTTP error"),
        getattr(response, "headers", None),
        None,
    )


def _copy_body(response, handle) -> None:
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            return
        handle.write(chunk)


def _content_length(url: str, response) -> int | None:
    """Declared body length, or None when the header is absent or unusable."""
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Content-Length")
    if raw is None:
        return None
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    logger.warning("ignoring unusable Content-Length %r for %s", raw, url)
    return None


def _pin_bytes(entry: dict, label: str) -> int | None:
    """The pin's ``bytes`` count, when the field is present."""
    if "bytes" not in entry:
        return None
    value = entry["bytes"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}: pin bytes must be a non-negative integer, got {value!r}")
    return value


def _size_expectations(
    actual: int, content_length: int | None, pinned_bytes: int | None
) -> list[tuple[str, int]]:
    """Sources whose declared length is not ``actual``. Empty when the body matches."""
    mismatched: list[tuple[str, int]] = []
    if content_length is not None and actual != content_length:
        mismatched.append(("Content-Length", content_length))
    if pinned_bytes is not None and actual != pinned_bytes:
        mismatched.append(("pin", pinned_bytes))
    return mismatched


def _place_verified(
    url: str,
    dest: Path,
    expected: str,
    label: str,
    opener,
    expected_bytes: int | None,
) -> None:
    """Stream ``url`` to a temp file, check size and ``expected``, then replace ``dest``.

    Size is checked before the sha256. A short or wrong-sized body raises
    ``SizeMismatch`` so the caller can retry it. A hash mismatch is raised only
    when the body has the expected length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            response = opener(url)
            try:
                _raise_for_status(url, response)
                content_length = _content_length(url, response)
                _copy_body(response, handle)
            finally:
                _close_quietly(response)
        actual = tmp.stat().st_size
        mismatched = _size_expectations(actual, content_length, expected_bytes)
        if mismatched:
            raise SizeMismatch(label, actual, mismatched)
        got = sha256(tmp)
        if got != expected:
            raise ChecksumError(f"{label}: expected {expected}, downloaded {got}")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _download(
    url: str,
    dest: Path,
    expected: str,
    label: str,
    opener,
    expected_bytes: int | None,
) -> None:
    """Download one pinned file, retrying only transient failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _place_verified(url, dest, expected, label, opener, expected_bytes)
            return
        except ChecksumError:
            raise
        except Exception as exc:
            _close_quietly(exc)
            give_up = not _is_transient(exc) or attempt == MAX_ATTEMPTS
            if give_up:
                logger.warning(
                    "download attempt %s/%s failed for %s: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    url,
                    exc,
                )
                if _is_transient(exc):
                    raise DownloadError(
                        f"download of {url} failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                raise
            delay = _retry_after_seconds(url, exc)
            if delay is None:
                delay = _backoff_seconds(attempt)
            elif delay > MAX_RETRY_AFTER_SECONDS:
                logger.warning(
                    "Retry-After for %s is %.0fs, above the %ss cap; waiting %ss",
                    url,
                    delay,
                    MAX_RETRY_AFTER_SECONDS,
                    MAX_RETRY_AFTER_SECONDS,
                )
                delay = float(MAX_RETRY_AFTER_SECONDS)
            logger.warning(
                "download attempt %s/%s failed for %s: %s; retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                url,
                exc,
                delay,
            )
            time.sleep(delay)


def fetch(root: Path, opener=urllib.request.urlopen) -> list[Path]:
    """Download missing files and verify every file, cached or not.

    Transient network failures are retried, as is a body that does not match
    Content-Length or the pin's ``bytes``. A hash mismatch on a body of the
    expected size raises ChecksumError immediately and leaves the previous
    cached file in place.
    """
    root = Path(root)
    done = []
    for name, group in load_pins(root)["groups"].items():
        for path, entry in group["files"].items():
            dest = root / group["dest"] / path
            label = f"{name}:{path}"
            expected_bytes = _pin_bytes(entry, label)
            if not dest.exists() or sha256(dest) != entry["sha256"]:
                _download(
                    url_for(group, path, entry),
                    dest,
                    entry["sha256"],
                    label,
                    opener,
                    expected_bytes,
                )
            done.append(dest)
    return done
