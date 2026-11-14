# SPDX-License-Identifier: Apache-2.0
"""Generic scene helpers.

``add_world`` adds a world's walls as visual-only boxes (contype 0, conaffinity 0,
so they create no contact solver pairs; wall clearance is measured geometrically)
and its named places as sites, read back with ``place_pose`` from the compiled model.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from livingproof_sim_mujoco.compat.hallway import FROZEN_WORLD_PATH as WORLD

PLACE = "place:"

__all__ = [
    "PLACE",
    "WORLD",
    "add_world",
    "actuator_table",
    "load_world",
    "place_pose",
    "to_pd_torque",
]


def load_world(root: Path, path: str | None = None) -> dict:
    """Retired. The simulator supplies the world; the robot package does not."""
    del root, path
    raise SystemExit(
        "INCOMPATIBLE: the simulator supplies the world from the vendored release named "
        "by the caller's verification lock. The robot package does not. "
        "Example: python -m livingproof_sim_mujoco.evidence "
        "<adapter> <configuration> <out> --lock path/to/lock.json"
    )


def add_world(spec: mujoco.MjSpec, world: dict) -> list[str]:
    """Add the world's walls (visual only) and places (sites); return the wall names."""
    walls = []
    for geom in world["geometry"]:
        if geom["role"] != "wall":
            continue
        g = spec.worldbody.add_geom()
        g.name, g.type = geom["name"], mujoco.mjtGeom.mjGEOM_BOX
        g.size = [s / 2 for s in geom["size"]]
        g.pos = geom["centre"]
        g.contype = g.conaffinity = 0
        g.rgba = [0.55, 0.6, 0.7, 0.45]
        walls.append(geom["name"])
    for place in world["places"]:
        site = spec.worldbody.add_site()
        site.name = PLACE + place["name"]
        site.pos = place.get("position") or place.get("reference") or [0.0, 0.0, 0.0]
        yaw = math.radians(place.get("yaw_deg", 0.0))
        site.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        site.size = [0.03, 0.03, 0.03]
        site.rgba = [0.1, 0.6, 0.2, 0.5]
    return walls


def actuator_table(model: mujoco.MjModel) -> tuple[dict[str, float], dict[str, tuple]]:
    """Published <position> gains and control ranges, by actuator name."""
    kp, ctrlrange = {}, {}
    for i in range(model.nu):
        name = model.actuator(i).name
        kp[name] = float(model.actuator_gainprm[i][0])
        ctrlrange[name] = tuple(float(v) for v in model.actuator_ctrlrange[i])
    return kp, ctrlrange


def to_pd_torque(spec: mujoco.MjSpec) -> None:
    """Turn every <position> actuator into a <motor>; torque limits stay each joint's
    actuatorfrcrange."""
    for act in spec.actuators:
        act.set_to_motor()
        act.inheritrange = 0
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE


def place_pose(model: mujoco.MjModel, name: str) -> tuple[np.ndarray, float]:
    """Position and yaw of a named world place, read from the compiled model."""
    site = model.site(PLACE + name)
    w, _, _, z = model.site_quat[site.id]
    return model.site_pos[site.id].copy(), 2 * math.atan2(z, w)
