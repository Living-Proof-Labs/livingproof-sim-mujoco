# SPDX-License-Identifier: Apache-2.0
"""Hallway-walk observations: evidence only, no grading.

``observe`` steps a simulation and records what happened in one trial: the base
position trace, the largest sideways offset, the lowest base height, the first exit
crossing, and every wall contact measured on the robot's real geometry. It applies
no threshold and decides no outcome; the capability verifier's evaluators do that.

Wall distance is the signed ``mj_geomDistance`` from every robot geom (primitives
exactly, meshes as their convex hulls, which enclose the mesh) to each wall geom,
capped at ``DISTMAX``. A contact is any sample at or below zero; its depth is the
deepest penetration while it lasts.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

FORMAT = "livingproof-hallway-observation/1.0.0"
DISTMAX = 0.25
COLUMNS = ["t", "x", "y", "z", "wall_distance"]


class WallDistance:
    """Signed real-geometry distance from the robot to named wall geoms."""

    def __init__(self, model: mujoco.MjModel, walls: list[str]):
        self.model = model
        self.walls = {name: model.geom(name).id for name in walls}
        self.robot = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] != 0])
        # Bounding spheres prune geoms that cannot be within DISTMAX of a wall.
        self.rbound = model.geom_rbound[self.robot]

    def __call__(self, data: mujoco.MjData) -> dict[str, float]:
        out = {}
        for name, w in self.walls.items():
            centre = data.geom_xpos[w]
            half = self.model.geom_size[w]
            rot = data.geom_xmat[w].reshape(3, 3)
            local = np.abs((data.geom_xpos[self.robot] - centre) @ rot) - half
            outside = np.linalg.norm(np.maximum(local, 0.0), axis=1)
            near = self.robot[outside - self.rbound < DISTMAX]
            out[name] = min(
                (
                    float(mujoco.mj_geomDistance(self.model, data, int(g), w, DISTMAX, None))
                    for g in near
                ),
                default=DISTMAX,
            )
        return out


def observe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: Callable[[mujoco.MjData], None],
    *,
    base_body: str,
    walls: list[str],
    exit_x: float,
    seconds: float,
    sample_hz: float = 50.0,
    centreline_y: float = 0.0,
    on_step: Callable[[int, mujoco.MjData], None] | None = None,
) -> dict:
    """Run one trial from the current state and return its observations.

    ``step(data)`` advances one physics timestep (for example a policy step).
    The trial ends when the base crosses ``exit_x``, after ``seconds``, or when the
    state becomes non-finite. ``exit_x`` ends the recording; it is not a judgement.
    """
    distance = WallDistance(model, walls)
    base = model.body(base_body).id
    every = max(1, int(round(1 / (sample_hz * model.opt.timestep))))
    samples, contacts, open_contact = [], [], None
    end, exit_time = "time_limit", None
    lateral = {"value": -1.0, "time": None}
    height = {"value": float("inf"), "time": None}
    for i in range(int(round(seconds / model.opt.timestep)) + 1):
        if on_step is not None:
            on_step(i, data)
        if i % every == 0:
            if not np.all(np.isfinite(data.qpos)):
                end = "non_finite_state"
                break
            t = round(float(data.time), 4)
            x, y, z = (float(v) for v in data.xpos[base])
            per_wall = distance(data)
            wall, d = min(per_wall.items(), key=lambda kv: kv[1])
            samples.append([t, round(x, 4), round(y, 4), round(z, 4), round(d, 4)])
            if abs(y - centreline_y) > lateral["value"]:
                lateral = {"value": round(abs(y - centreline_y), 4), "time": t}
            if z < height["value"]:
                height = {"value": round(z, 4), "time": t}
            if d <= 0.0:
                if open_contact is None:
                    open_contact = {"wall": wall, "start": t, "end": t, "depth": 0.0}
                    contacts.append(open_contact)
                open_contact["end"] = t
                open_contact["depth"] = round(max(open_contact["depth"], -d), 4)
            else:
                open_contact = None
            if x >= exit_x:
                exit_time, end = t, "crossed_exit"
                break
        step(data)
    return {
        "format": FORMAT,
        "end": end,
        "duration_s": samples[-1][0] if samples else 0.0,
        "exit_time": exit_time,
        "max_abs_lateral": lateral,
        "min_base_height": height,
        "wall_contacts": contacts,
        "wall_distance": {"method": "mj_geomDistance", "distmax": DISTMAX},
        "trace": {"columns": COLUMNS, "samples": samples},
    }
