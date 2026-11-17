# SPDX-License-Identifier: Apache-2.0
"""Shared livingproof-world schema and shape checks for #89 and #88.

Validates a world against the official livingproof-worlds JSON Schema (vendored
from the pinned release in worlds/schema/pins.json) before the hand checks.
The hand checks stay as a backstop for rules a Python validator may not
enforce, such as NaN or inf, and for range and count: a box size must be
positive, a segment has exactly two points, and a polyline has at least two.

This module is identical on the format-0.3 and declared-world branches.
"""

from __future__ import annotations

from functools import cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).parent / "worlds" / "schema"
SCHEMA_BY_FORMAT = {
    "livingproof-world/0.1": "world.schema.json",
    "livingproof-world/0.2": "world.schema.json",
    "livingproof-world/0.3": "world-0.3.schema.json",
}


class WorldShapeError(ValueError):
    """A world field is present but the wrong type, shape, or length."""


@cache
def _schema_pin() -> dict:
    return json.loads((SCHEMA_DIR / "pins.json").read_text())


@cache
def _load_schema(name: str) -> dict:
    pin = _schema_pin()
    expected = pin["files"].get(name)
    if expected is None:
        raise WorldShapeError(f"{name} is not pinned in worlds/schema/pins.json")
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise WorldShapeError(f"missing vendored world schema {name}")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise WorldShapeError(
            f"checksum mismatch for {name}: expected {expected} "
            f"({pin['repository']} {pin['commit']}), got {got}"
        )
    return json.loads(path.read_text())


def validate_schema(world: Any) -> None:
    """Refuse a world that does not match the published livingproof-world schema.

    Unknown formats are left to the caller, so existing ``world format`` messages
    stay unchanged. Schema files are the official copies from the pinned
    livingproof-worlds release, not a hand-authored extract.
    """
    if not isinstance(world, dict):
        raise WorldShapeError("world must be an object")
    fmt = world.get("format")
    name = SCHEMA_BY_FORMAT.get(fmt) if isinstance(fmt, str) else None
    if name is None:
        return
    error = next(Draft202012Validator(_load_schema(name)).iter_errors(world), None)
    if error is None:
        return
    path = "/".join(str(part) for part in error.absolute_path)
    if path:
        raise WorldShapeError(f"{path}: {error.message}")
    raise WorldShapeError(error.message)


def require_object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise WorldShapeError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise WorldShapeError(f"{label} must be a string")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldShapeError(f"{label} must be a number")
    if not math.isfinite(value):
        raise WorldShapeError(f"{label} must be a finite number")
    return float(value)


def vector3(value: Any, label: str, *, positive: bool = False) -> list[float]:
    """A list of exactly three finite numbers, optionally all positive."""
    if not isinstance(value, list) or len(value) != 3:
        raise WorldShapeError(f"{label} must be a list of 3 numbers")
    numbers = [require_number(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if positive:
        for index, number in enumerate(numbers):
            if number <= 0:
                raise WorldShapeError(f"{label}[{index}] must be a positive number")
    return numbers


def points3(
    value: Any,
    label: str,
    *,
    count: int | None = None,
    minimum: int | None = None,
) -> list[list[float]]:
    """A list of 3-number lists, optionally of an exact or minimum length."""
    if not isinstance(value, list):
        raise WorldShapeError(f"{label} must be a list")
    if count is not None and len(value) != count:
        raise WorldShapeError(f"{label} must have exactly {count} points")
    if minimum is not None and len(value) < minimum:
        raise WorldShapeError(f"{label} must have at least {minimum} points")
    return [vector3(item, f"{label}[{index}]") for index, item in enumerate(value)]


def require_geometry_item(item: Any, *, keys: set[str], label: str) -> dict:
    """An object with the required geometry keys, string name/type, and 3-number centre/size."""
    obj = require_object(item, label)
    absent = [key for key in sorted(keys) if key not in obj]
    if absent:
        raise WorldShapeError(f"{label} has no {', '.join(repr(key) for key in absent)}")
    require_string(obj.get("name"), f"{label} name")
    require_string(obj.get("type"), f"{label} type")
    vector3(obj.get("centre"), f"{label} centre")
    vector3(obj.get("size"), f"{label} size", positive=True)
    return obj


def require_place_item(item: Any, label: str) -> dict:
    """An object; pose has a 3-number position and numeric yaw_deg; paths have 3-number points."""
    obj = require_object(item, label)
    typ = obj.get("type")
    if typ == "pose":
        vector3(obj.get("position"), f"{label} position")
        require_number(obj.get("yaw_deg"), f"{label} yaw_deg")
    elif typ == "segment":
        points3(obj.get("points"), f"{label} points", count=2)
    elif typ == "polyline":
        points3(obj.get("points"), f"{label} points", minimum=2)
    return obj
