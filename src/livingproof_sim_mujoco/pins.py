"""Pinned upstream files: download by commit into a robot package's cache, verify sha256.

A robot package keeps a ``pins.json`` (groups of files at a repository commit, each with a
sha256 and a ``dest`` directory). ``fetch(root)`` downloads missing files and verifies every
file, cached or not. Transient download errors are retried; a hash mismatch is not.
"""

from __future__ import annotations

import email.utils
import hashlib
import http.client
import json
import logging
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Five attempts. The four waits are 10s, 20s, 40s and 80s, each jittered by 20%
# (about 2-3 minutes altogether). Retry-After replaces that wait when it is present.
MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 10.0
_BACKOFF_JITTER = 0.2


class ChecksumError(RuntimeError):
    pass


class DownloadError(RuntimeError):
    """A pinned file could not be downloaded after every transient attempt."""


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


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        when = email.utils.parsedate_to_datetime(text)
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _backoff_seconds(attempt: int) -> float:
    """Wait after a failed attempt (1-based) before the next try."""
    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    spread = delay * _BACKOFF_JITTER
    return delay + random.uniform(-spread, spread)


def _is_transient(exc: BaseException) -> bool:
    """Connection resets, refusals, timeouts, status-less URLErrors, HTTP 5xx and 429."""
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


def _place_verified(url: str, dest: Path, expected: str, label: str, opener) -> None:
    """Stream ``url`` to a temp file, check ``expected``, then atomically replace ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            response = opener(url)
            try:
                _raise_for_status(url, response)
                _copy_body(response, handle)
            finally:
                _close_quietly(response)
        got = sha256(tmp)
        if got != expected:
            raise ChecksumError(f"{label}: expected {expected}, downloaded {got}")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _download(url: str, dest: Path, expected: str, label: str, opener) -> None:
    """Download one pinned file, retrying only transient failures."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _place_verified(url, dest, expected, label, opener)
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
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = _backoff_seconds(attempt)
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

    Transient network failures are retried. A hash mismatch raises ChecksumError
    immediately and leaves the previous cached file in place.
    """
    root = Path(root)
    done = []
    for name, group in load_pins(root)["groups"].items():
        for path, entry in group["files"].items():
            dest = root / group["dest"] / path
            if not dest.exists() or sha256(dest) != entry["sha256"]:
                _download(
                    url_for(group, path, entry),
                    dest,
                    entry["sha256"],
                    f"{name}:{path}",
                    opener,
                )
            done.append(dest)
    return done
