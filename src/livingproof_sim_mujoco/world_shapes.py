# SPDX-License-Identifier: Apache-2.0
"""Type, shape and length checks for world geometry and places.

Both the hallway adapter (#89) and declared-world loading (#88) use this
module so a present-but-wrong-typed field is refused clearly instead of
crashing later in MuJoCo or compose.
"""

from __future__ import annotations

import math
from typing import Any


class WorldShapeError(ValueError):
    """A world field is present but the wrong type, shape, or length."""


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


def vector3(value: Any, label: str) -> list[float]:
    """A list of exactly three finite numbers."""
    if not isinstance(value, list) or len(value) != 3:
        raise WorldShapeError(f"{label} must be a list of 3 numbers")
    return [require_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def points3(value: Any, label: str) -> list[list[float]]:
    """A list of 3-number lists."""
    if not isinstance(value, list):
        raise WorldShapeError(f"{label} must be a list")
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
    vector3(obj.get("size"), f"{label} size")
    return obj


def require_place_item(item: Any, label: str) -> dict:
    """An object; pose has a 3-number position and numeric yaw_deg; paths have 3-number points."""
    obj = require_object(item, label)
    typ = obj.get("type")
    if typ == "pose":
        vector3(obj.get("position"), f"{label} position")
        require_number(obj.get("yaw_deg"), f"{label} yaw_deg")
    elif typ in {"segment", "polyline"}:
        points3(obj.get("points"), f"{label} points")
    return obj
