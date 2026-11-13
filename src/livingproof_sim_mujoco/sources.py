# SPDX-License-Identifier: Apache-2.0
"""World and scenario parameter files for one evidence run.

A verification-lock ``sources`` entry wins. Its sha256 is checked and a
mismatch is an ERROR; pins are not consulted after that. When the lock has no
entry, the robot's ``pins.json`` is used.

# TODO(#40 step 7): remove the pins.json fallback, and the vendored parameter
# file used when the pins do not list one, once the verification lock always
# carries the world and the scenario files.
# https://github.com/Living-Proof-Labs/robot-capability-specs/issues/40
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path

from livingproof_sim_mujoco.compat.hallway import FROZEN_WORLD_PATH, parameters_document
from livingproof_sim_mujoco.pins import load_pins

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _entry(lock: dict | None, kind: str) -> dict | None:
    """``{"path", "sha256"}`` when the lock records this file, else None."""
    if not isinstance(lock, dict):
        return None
    sources = lock.get("sources")
    if not isinstance(sources, dict) or kind not in sources:
        return None
    entry = sources.get(kind)
    if not isinstance(entry, dict):
        raise SystemExit(
            f"ERROR: lock sources.{kind} must be an object with path and sha256, "
            f"got {entry!r}. "
            'Example: "sources": {"world": {"path": "world.json", "sha256": "<64 hex>"}, '
            '"parameters": {"path": "parameters.json", "sha256": "<64 hex>"}}'
        )
    path, digest = entry.get("path"), entry.get("sha256")
    if not isinstance(path, str) or not path:
        raise SystemExit(
            f'ERROR: lock sources.{kind} needs a relative path. Example: "path": "world.json"'
        )
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise SystemExit(
            f"ERROR: lock sources.{kind} sha256 must be 64 hex characters, got {digest!r}. "
            "Example: sha256 of the file named by path."
        )
    return {"path": path, "sha256": digest}


def _read_locked(root: Path, entry: dict, kind: str) -> bytes:
    raw = Path(entry["path"])
    if raw.is_absolute() or ".." in raw.parts:
        raise SystemExit(
            f"ERROR: lock sources.{kind} path must stay inside the robot repository, "
            f'got {entry["path"]!r}. Example: "path": "world.json"'
        )
    path = Path(root) / raw
    if not path.is_file():
        raise SystemExit(
            f"ERROR: lock sources.{kind} file {entry['path']} is missing "
            f"(expected sha256 {entry['sha256']}). "
            "The lock entry wins over pins.json, so the pins file was not used. "
            "Put the file at that path or remove the lock entry. "
            f'Example: "sources": {{"{kind}": {{"path": "{entry["path"]}", '
            f'"sha256": "{entry["sha256"]}"}}}}'
        )
    # One read: the bytes that are checked are the bytes that are loaded.
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(
            f"ERROR: lock sources.{kind} sha256 mismatch for {entry['path']}: "
            f"lock says {entry['sha256']}, file is {digest}. "
            "The lock entry wins over pins.json, so the pins file was not used. "
            "Replace the file or update the lock sha256."
        )
    return data


def _pinned(root: Path, relative: str) -> Path | None:
    """Path of ``relative`` when some pins group lists it and the file is present."""
    pins_path = Path(root) / "pins.json"
    if not pins_path.is_file():
        return None
    try:
        groups = load_pins(root).get("groups", {})
    except (OSError, ValueError):
        return None
    if not isinstance(groups, dict):
        return None
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        files = group.get("files")
        if not isinstance(files, dict) or relative not in files:
            continue
        dest = Path(root) / group.get("dest", "") / relative
        if dest.is_file():
            return dest
    return None


def _world_relative(lock: dict | None) -> str:
    world = lock.get("world") if isinstance(lock, dict) else None
    if isinstance(world, dict) and world.get("id") and world.get("version"):
        return f"worlds/{world['id']}/{world['version']}/world.json"
    return FROZEN_WORLD_PATH


def _parameters_relative(lock: dict | None) -> str | None:
    scenario = lock.get("scenario") if isinstance(lock, dict) else None
    if isinstance(scenario, dict):
        scenario = scenario.get("id")
    if not isinstance(scenario, str) or "@" not in scenario:
        return None
    return f"specs/scenarios/{scenario.split('@', 1)[0]}/parameters.json"


@dataclass(frozen=True)
class WorldSource:
    """The world file one run loads, and the sha256 of the bytes that were read.

    ``origin`` is ``lock`` (a ``sources.world`` entry), ``pins`` or ``adapter``.
    For ``adapter`` the robot package supplies the document, so no bytes were
    read here and ``sha256`` is None.
    """

    document: dict | None
    origin: str
    sha256: str | None
    path: str | None


def world_source(root: Path, lock: dict | None) -> WorldSource:
    """The lock file, else the pins file, else an ``adapter`` source with no document."""
    entry = _entry(lock, "world")
    if entry is not None:
        raw = _read_locked(root, entry, "world")
        digest = hashlib.sha256(raw).hexdigest()
        return WorldSource(json.loads(raw), "lock", digest, entry["path"])
    # TODO(#40 step 7): remove this pins.json fallback. See the module docstring.
    pinned = _pinned(root, _world_relative(lock))
    if pinned is None:
        return WorldSource(None, "adapter", None, None)
    raw = pinned.read_bytes()
    relative = pinned.relative_to(Path(root)).as_posix()
    return WorldSource(json.loads(raw), "pins", hashlib.sha256(raw).hexdigest(), relative)


def world_document(root: Path, lock: dict | None) -> dict | None:
    """The lock file, else the pins file, else None so the adapter can supply it."""
    return world_source(root, lock).document


def parameters_document_for(root: Path, lock: dict | None) -> tuple[dict, str]:
    """``(document, source)`` with source ``lock``, ``pins``, or ``vendored``.

    The vendored file is only used when the lock has no entry and the pins do
    not list the scenario parameter file. That keeps robots whose pins predate
    the parameter file able to record the frozen scenario.
    """
    entry = _entry(lock, "parameters")
    if entry is not None:
        return json.loads(_read_locked(root, entry, "parameters")), "lock"
    # TODO(#40 step 7): remove this pins.json fallback. See the module docstring.
    relative = _parameters_relative(lock)
    pinned = _pinned(root, relative) if relative else None
    if pinned is not None:
        return json.loads(pinned.read_text()), "pins"
    # TODO(#40 step 7): remove this vendored copy once the lock always has the file.
    return parameters_document(), "vendored"
