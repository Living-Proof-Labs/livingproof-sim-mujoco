"""Pinned upstream files: download by commit into a robot package's cache, verify sha256.

A robot package keeps a ``pins.json`` (groups of files at a repository commit, each with a
sha256 and a ``dest`` directory). ``fetch(root)`` downloads missing files and verifies every
file, cached or not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


class ChecksumError(RuntimeError):
    pass


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


def fetch(root: Path, opener=urllib.request.urlopen) -> list[Path]:
    """Download missing files; verify every file, cached or not. Raises on a mismatch."""
    root = Path(root)
    done = []
    for name, group in load_pins(root)["groups"].items():
        for path, entry in group["files"].items():
            dest = root / group["dest"] / path
            if not dest.exists() or sha256(dest) != entry["sha256"]:
                data = opener(url_for(group, path, entry)).read()
                got = hashlib.sha256(data).hexdigest()
                if got != entry["sha256"]:
                    raise ChecksumError(
                        f"{name}:{path}: expected {entry['sha256']}, downloaded {got}"
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            elif sha256(dest) != entry["sha256"]:
                raise ChecksumError(f"{name}:{path}: cached file does not match its pin")
            done.append(dest)
    return done
