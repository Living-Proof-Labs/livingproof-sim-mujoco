# SPDX-License-Identifier: Apache-2.0
"""The robot tag and commit this repository's lock actually pins."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
_COMMIT = re.compile(r"[0-9a-f]{40}")


def lock_pinning(package: str, tag: str, commit: str) -> str:
    """A lock fragment whose git source for ``package`` is ``tag`` at ``commit``."""
    return (
        "[[package]]\n"
        f'name = "{package}"\n'
        'version = "0.1.0"\n'
        "source = { git = "
        f'"https://github.com/Living-Proof-Labs/{package}?tag={tag}#{commit}" }}\n'
    )


def locked_robot_source(package: str, text: str | None = None) -> tuple[str, str]:
    """``(tag, commit)`` recorded for ``package`` in the lock.

    ``text`` defaults to this repository's ``uv.lock``. The revision is that
    package's git source: the ``tag`` query and the commit fragment. Nothing
    here assumes a particular tag.
    """
    raw = (ROOT / "uv.lock").read_text() if text is None else text
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise AssertionError(f"uv.lock is not valid TOML: {exc}") from exc
    urls = [
        item["source"]["git"]
        for item in data.get("package", [])
        if isinstance(item, dict)
        and item.get("name") == package
        and isinstance(item.get("source"), dict)
        and isinstance(item["source"].get("git"), str)
    ]
    if len(urls) != 1:
        raise AssertionError(f"uv.lock has {len(urls)} git sources for {package}, expected one")
    return _tag_and_commit(package, urls[0])


def _tag_and_commit(package: str, git: str) -> tuple[str, str]:
    parts = urlsplit(git)
    name = parts.path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    if name != package:
        raise AssertionError(f"uv.lock git source for {package} points at {git}")
    tags = parse_qs(parts.query).get("tag", [])
    commit = parts.fragment
    if len(tags) != 1 or not tags[0] or _COMMIT.fullmatch(commit) is None:
        raise AssertionError(f"uv.lock git source for {package} has no tag and commit: {git}")
    return tags[0], commit
