# SPDX-License-Identifier: Apache-2.0
"""Hallway-walk observations: evidence only, no grading.

``observe`` steps a simulation and records what happened in one trial: the base
position trace, the largest sideways offset, the lowest base height, the first exit
crossing, and every wall contact measured on the robot's real geometry. It applies
no threshold and decides no outcome; the capability verifier's evaluators do that.

The walls are visual only (no contacts; see ``scene.add_world``), so a robot can move
into and through them. Each wall's inner and outer faces come from its box geom.

- Wall distance is the signed ``mj_geomDistance`` from every robot geom (primitives
  exactly, meshes as their convex hulls, which enclose the mesh) to each wall geom,
  capped at ``DISTMAX``: the clearance while the robot is clear of the wall.
- Wall intrusion is how far the farthest robot part reaches past a wall's inner face,
  measured across the hallway (y) on the robot's collision geometry only (``Reach``,
  ``collision_geoms``): what physics would touch, not visual-only meshes. It grows with how
  far the robot is in the wall, unlike a penetration depth, which against a thin wall
  peaks and falls as a body passes through. A contact is any sample with intrusion at or
  above zero; its ``depth`` is the largest intrusion while it lasts.
- The trial ends ``left_hallway`` as soon as any robot part reaches past a wall's outer
  face (any part, not only the base: by then the robot is through the wall), or when the
  base reaches the exit line outside the exit segment (|y| above the half-width). Only a
  crossing within the segment ends ``crossed_exit``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import mujoco
import numpy as np

FORMAT = "livingproof-hallway-observation/2.0.0"
DISTMAX = 0.25
COLUMNS = ["t", "x", "y", "z", "wall_distance", "wall_intrusion"]


def wall_faces(model: mujoco.MjModel, walls: list[str]) -> dict[str, dict]:
    """Each wall's side (+1 left, -1 right) and its inner and outer face distance from the
    centreline (y = 0), from its compiled box geom. Walls must be axis-aligned boxes."""
    out = {}
    for name in walls:
        g = model.geom(name)
        if int(g.type[0]) != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"wall {name} is not a box geom")
        y, half = float(model.geom_pos[g.id][1]), float(model.geom_size[g.id][1])
        if not np.allclose(model.geom_quat[g.id], [1, 0, 0, 0]) or model.geom_bodyid[g.id] != 0:
            raise ValueError(f"wall {name} must be an axis-aligned world geom")
        if abs(y) <= half:
            raise ValueError(f"wall {name} straddles the centreline")
        out[name] = {"side": 1 if y > 0 else -1, "inner": abs(y) - half, "outer": abs(y) + half}
    return out


COLLISION_RULE = (
    "contype or conaffinity nonzero, or in an explicit contact pair, "
    "plus every geom in the same geom group as those (the robot's collision class)"
)


def collision_geoms(model: mujoco.MjModel) -> tuple[list[int], dict]:
    """The robot's collision geoms, and a record of how they were chosen.

    Robots mark collision geometry in different ways: Booster T1 sets contype/conaffinity
    on its collision primitives; the Unitree G1 and Go1 (MuJoCo Playground) zero both and
    collide through explicit ``<pair>`` elements or a few foot geoms, keeping the rest of
    their collision class in one geom group (3). So a geom counts when it collides by
    contype/conaffinity or a pair, or shares a geom group with one that does. Visual-only
    geoms (their own group, contype = conaffinity = 0) never count.

    Bodies whose geoms are all visual are listed (``bodies_without_collision_geoms``) and
    warned about, not silently skipped: wall intrusion cannot see them. A robot with no
    collision geom at all fails closed (ValueError)."""
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]
    paired = {int(g) for g in (*model.pair_geom1, *model.pair_geom2)}
    marked = {g for g in robot if model.geom_contype[g] or model.geom_conaffinity[g] or g in paired}
    if not marked:
        raise ValueError("robot has no collision geoms; wall intrusion cannot be measured")
    groups = sorted({int(model.geom_group[g]) for g in marked})
    chosen = [g for g in robot if g in marked or int(model.geom_group[g]) in groups]
    covered = {int(model.geom_bodyid[g]) for g in chosen}
    uncovered = sorted(
        {model.body(int(model.geom_bodyid[g])).name for g in robot}
        - {model.body(b).name for b in covered}
    )
    if uncovered:
        warnings.warn(
            "wall intrusion ignores bodies with no collision geom: " + ", ".join(uncovered),
            stacklevel=2,
        )
    record = {
        "rule": COLLISION_RULE,
        "collision_groups": groups,
        "geoms": len(chosen),
        "bodies_without_collision_geoms": uncovered,
    }
    return chosen, record


class Reach:
    """How far the robot's collision geometry reaches across the hallway: for each side
    (+y, -y), the largest y-extent of any collision geom on that side (``collision_geoms``).
    Primitives exactly; meshes from their compiled vertices (the convex hull's extent
    equals the mesh's)."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.robot, self.record = collision_geoms(model)
        self.verts = {}
        for g in self.robot:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                self.verts[g] = np.array(model.mesh_vert[a : a + n], dtype=float)

    def extent(self, data: mujoco.MjData, g: int) -> tuple[float, float]:
        """(lowest y, highest y) of geom ``g`` in world coordinates."""
        m, t = self.model, self.model.geom_type[g]
        c, rot = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        size, row = m.geom_size[g], rot[1]  # row: geom axes' y components
        G = mujoco.mjtGeom
        if t == G.mjGEOM_MESH:
            ys = c[1] + self.verts[g] @ row
            return float(ys.min()), float(ys.max())
        if t == G.mjGEOM_SPHERE:
            r = size[0]
        elif t == G.mjGEOM_CAPSULE:
            r = abs(row[2]) * size[1] + size[0]
        elif t == G.mjGEOM_CYLINDER:
            r = abs(row[2]) * size[1] + size[0] * np.sqrt(max(0.0, 1 - row[2] ** 2))
        elif t == G.mjGEOM_BOX:
            r = float(np.abs(row) @ size)
        elif t == G.mjGEOM_ELLIPSOID:
            r = float(np.sqrt(((row * size) ** 2).sum()))
        else:  # planes, heightfields, SDFs: no finite robot extent we can measure
            raise ValueError(f"robot geom {g} has unsupported type {int(t)}")
        return float(c[1] - r), float(c[1] + r)

    def __call__(self, data: mujoco.MjData) -> dict[int, float]:
        """{+1: highest y of any robot part, -1: -(lowest y)}; both are distances from y = 0
        on that side (negative when the whole robot is on the other side)."""
        lo, hi = zip(*(self.extent(data, g) for g in self.robot))
        return {1: max(hi), -1: -min(lo)}


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
    faces = wall_faces(model, walls)
    reach = Reach(model)
    half_width = min((f["inner"] for f in faces.values()), default=float("inf"))
    left = None
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
            d = min(per_wall.values(), default=DISTMAX)
            r = reach(data)
            intrusion = {w: r[f["side"]] - f["inner"] for w, f in faces.items()}
            wall, deepest = max(intrusion.items(), key=lambda kv: kv[1], default=(None, -DISTMAX))
            samples.append(
                [t, round(x, 4), round(y, 4), round(z, 4), round(d, 4), round(deepest, 4)]
            )
            if abs(y - centreline_y) > lateral["value"]:
                lateral = {"value": round(abs(y - centreline_y), 4), "time": t}
            if z < height["value"]:
                height = {"value": round(z, 4), "time": t}
            if deepest >= 0.0:
                if open_contact is None:
                    open_contact = {"wall": wall, "start": t, "end": t, "depth": 0.0}
                    contacts.append(open_contact)
                open_contact["end"] = t
                open_contact["depth"] = round(max(open_contact["depth"], deepest), 4)
            else:
                open_contact = None
            through = [w for w, f in faces.items() if r[f["side"]] > f["outer"]]
            if through:
                left = {"time": t, "via": through[0], "x": round(x, 4), "y": round(y, 4)}
                end = "left_hallway"
                break
            if x >= exit_x:
                if abs(y - centreline_y) <= half_width:
                    exit_time, end = t, "crossed_exit"
                else:
                    left = {
                        "time": t,
                        "via": "exit-outside-segment",
                        "x": round(x, 4),
                        "y": round(y, 4),
                    }
                    end = "left_hallway"
                break
        step(data)
    return {
        "format": FORMAT,
        "end": end,
        "duration_s": samples[-1][0] if samples else 0.0,
        "exit_time": exit_time,
        "left_hallway": left,
        "max_abs_lateral": lateral,
        "min_base_height": height,
        "wall_contacts": contacts,
        "wall_distance": {"method": "mj_geomDistance", "distmax": DISTMAX},
        "wall_intrusion": {
            "method": "robot collision-geometry y-extent past each wall's inner face",
            "half_width": round(half_width, 4) if np.isfinite(half_width) else None,
            "walls": {w: {k: round(v, 4) for k, v in f.items()} for w, f in faces.items()},
            "geometry": reach.record,
        },
        "trace": {"columns": COLUMNS, "samples": samples},
    }
