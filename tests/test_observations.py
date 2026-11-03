# SPDX-License-Identifier: Apache-2.0
"""Given a body drifting into a wall, when observed, then contacts, extremes and exit are recorded, not judged."""

import json

import mujoco
import pytest

from livingproof_sim_mujoco.observations import (
    COLUMNS,
    DISTMAX,
    FORMAT,
    Reach,
    WallDistance,
    observe,
    wall_faces,
)

HALL = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <geom name="wall-left" type="box" pos="0 0.75 0.5" size="5 0.05 0.5" contype="0" conaffinity="0"/>
    <geom name="wall-right" type="box" pos="0 -0.75 0.5" size="5 0.05 0.5" contype="0" conaffinity="0"/>
    <body name="base" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.1" contype="0" conaffinity="0" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def trial(vx, vy, seconds=3.0, exit_x=10.0, xml=HALL, start=(0.0, 0.0)):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:2] = start
    d.qvel[:2] = [vx, vy]
    mujoco.mj_forward(m, d)
    return observe(
        m,
        d,
        lambda data: mujoco.mj_step(m, data),
        base_body="base",
        walls=["wall-left", "wall-right"],
        exit_x=exit_x,
        seconds=seconds,
    )


def test_real_geometry_distance_is_signed():
    """Inner wall face at 0.70 m, sphere radius 0.1 m: overlap is negative."""
    m = mujoco.MjModel.from_xml_string(HALL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert WallDistance(m, ["wall-left"])(d)["wall-left"] == pytest.approx(DISTMAX)
    d.qpos[1] = 0.65
    mujoco.mj_forward(m, d)
    assert WallDistance(m, ["wall-left"])(d)["wall-left"] == pytest.approx(-0.05, abs=1e-6)


def test_drift_into_wall_is_observed_with_time_and_depth():
    """Inner face 0.70 m, outer 0.80 m. At 0.25 m/s for 2.6 s the sphere's edge reaches
    0.75 m: 0.05 m into the wall, not yet through it."""
    o = trial(0.0, 0.25, seconds=2.6)
    assert o["format"] == FORMAT == "livingproof-hallway-observation/2.0.0"
    assert o["end"] == "time_limit" and o["exit_time"] is None and o["left_hallway"] is None
    (c,) = o["wall_contacts"]
    assert c["wall"] == "wall-left"
    assert c["start"] == pytest.approx(2.4, abs=0.03)  # 0.6 m surface gap at 0.25 m/s
    assert c["depth"] == pytest.approx(0.05, abs=0.01)  # edge at 0.75 m vs inner face 0.70 m
    assert o["max_abs_lateral"]["value"] == pytest.approx(0.65, abs=0.01)
    assert o["min_base_height"]["value"] == pytest.approx(0.3)
    assert (
        o["trace"]["columns"] == COLUMNS == ["t", "x", "y", "z", "wall_distance", "wall_intrusion"]
    )
    assert o["wall_intrusion"]["half_width"] == pytest.approx(0.7)
    assert o["wall_intrusion"]["walls"]["wall-right"] == {"side": -1, "inner": 0.7, "outer": 0.8}


def test_any_part_past_the_outer_face_ends_the_trial_left_hallway():
    """The sphere's edge passes the outer face (0.80 m) at y = 0.70 m, t = 2.8 s, while the
    base itself is still inside the wall: any robot part counts, not only the base."""
    o = trial(0.0, 0.25, seconds=5.0)
    assert o["end"] == "left_hallway" and o["exit_time"] is None
    left = o["left_hallway"]
    assert left["via"] == "wall-left" and left["time"] == pytest.approx(2.8, abs=0.03)
    assert left["y"] < 0.8  # the base had not passed the outer face
    assert o["duration_s"] == left["time"]
    (c,) = o["wall_contacts"]
    assert c["depth"] == pytest.approx(0.1, abs=0.01)  # the full wall thickness


THICK = HALL.replace('pos="0 0.75 0.5" size="5 0.05 0.5"', 'pos="0 1.2 0.5" size="5 0.5 0.5"')


def test_an_exit_outside_the_segment_is_not_crossed_exit():
    """Inner face 0.70 m, outer 1.70 m. The base reaches the exit line at y = 0.9 m, inside
    the thick wall but not through it: outside the exit segment, so it is left_hallway."""
    o = trial(1.0, 0.0, seconds=5.0, exit_x=2.0, xml=THICK, start=(1.5, 0.9))
    assert o["end"] == "left_hallway" and o["exit_time"] is None
    assert o["left_hallway"]["via"] == "exit-outside-segment"
    assert o["left_hallway"]["y"] == pytest.approx(0.9)
    inside = trial(1.0, 0.0, seconds=5.0, exit_x=2.0, xml=THICK, start=(1.5, 0.6))
    assert inside["end"] == "crossed_exit" and inside["left_hallway"] is None


SLAB = """
<mujoco>
  <worldbody>
    <geom name="wall-right" type="box" pos="0 -0.825 1.25" size="3 0.075 1.25" contype="0" conaffinity="0"/>
    <body name="base" pos="0 0 0.7">
      <freejoint/>
      <geom type="box" size="0.1 0.12 0.2" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 0 -0.2 -0.5" size="0.04" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_wall_slide_intrusion_grows_with_depth_unlike_penetration_distance():
    """The #37 reproduction: slide a body across a 0.15 m wall (inner face 0.75 m, like
    hallway-open). The signed mj_geomDistance peaks and falls as the body passes through;
    the intrusion past the inner face keeps growing."""
    m = mujoco.MjModel.from_xml_string(SLAB)
    d = mujoco.MjData(m)
    faces = wall_faces(m, ["wall-right"])
    assert faces["wall-right"] == pytest.approx({"side": -1, "inner": 0.75, "outer": 0.9})
    reach, dist = Reach(m), WallDistance(m, ["wall-right"])
    intrusions, distances = [], []
    for y in [-0.3 - 0.05 * i for i in range(20)]:
        d.qpos[1] = y
        mujoco.mj_forward(m, d)
        intrusions.append(reach(d)[-1] - 0.75)
        distances.append(dist(d)["wall-right"])
    assert all(b > a for a, b in zip(intrusions, intrusions[1:]))
    deepest = distances.index(min(distances))
    assert 0 < deepest < len(distances) - 1 and distances[-1] > min(distances)
    # Exact extents: capsule end at y + (-0.2 - 0.04); box half-width 0.12.
    d.qpos[1] = -0.5
    mujoco.mj_forward(m, d)
    assert reach(d)[-1] == pytest.approx(0.74)
    assert reach(d)[1] == pytest.approx(-0.38)


def test_mesh_extent_is_exact_from_vertices():
    xml = """
<mujoco>
  <asset><mesh name="tet" vertex="0 0 0  0.3 0 0  0 0.2 0  0 0 0.1"/></asset>
  <worldbody>
    <body name="base" pos="0 0.1 0.5" euler="0 0 90"><freejoint/>
      <geom type="mesh" mesh="tet" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>
"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] != 0)
    lo, hi = Reach(m).extent(d, g)
    # Rotated 90 degrees about z: mesh x (0..0.3) becomes world y.
    assert (lo, hi) == (pytest.approx(0.1), pytest.approx(0.4))


def test_exit_crossing_ends_the_trial_and_nothing_is_graded():
    o = trial(1.0, 0.0, seconds=5.0, exit_x=2.0)
    assert o["end"] == "crossed_exit" and o["exit_time"] == pytest.approx(2.0, abs=0.03)
    assert o["wall_contacts"] == []
    text = json.dumps(o)
    for word in ("PASS", "FAIL", "outcome", "grade", "limit"):
        assert word not in text
