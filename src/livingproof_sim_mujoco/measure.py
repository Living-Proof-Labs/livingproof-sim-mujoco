# SPDX-License-Identifier: Apache-2.0
"""Robot dimensions measured from a pinned model: world-frame mesh vertices and mass."""

from __future__ import annotations

import mujoco
import numpy as np


def mesh_points(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Every robot mesh vertex (bodies other than the world) in world coordinates."""
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[
            model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
        ]
        pts.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
    return np.vstack(pts)


def extents(points: np.ndarray) -> list[float]:
    """x, y, z extents in metres, rounded to millimetres."""
    return (points.max(0) - points.min(0)).round(3).tolist()


def keyframe_extents(model: mujoco.MjModel, keyframes) -> dict:
    """Width, length and mass over named start keyframes (base at the origin), largest wins."""
    d = mujoco.MjData(model)
    ext = []
    for key in keyframes:
        mujoco.mj_resetDataKeyframe(model, d, model.key(key).id)
        d.qpos[:2] = 0
        mujoco.mj_forward(model, d)
        ext.append({"keyframe": key, "extent_xyz_m": extents(mesh_points(model, d))})
    return {
        "width_m": max(e["extent_xyz_m"][1] for e in ext),
        "length_m": max(e["extent_xyz_m"][0] for e in ext),
        "mass_kg": round(float(sum(model.body_mass)), 2),
        "keyframes": ext,
    }
