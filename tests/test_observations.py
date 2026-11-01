# SPDX-License-Identifier: Apache-2.0
"""Given a body drifting into a wall, when observed, then contacts, extremes and exit are recorded, not judged."""

import json

import mujoco
import pytest

from livingproof_sim_mujoco.observations import DISTMAX, FORMAT, WallDistance, observe

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


def trial(vx, vy, seconds=3.0, exit_x=10.0):
    m = mujoco.MjModel.from_xml_string(HALL)
    d = mujoco.MjData(m)
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
    o = trial(0.0, 0.25)
    assert o["format"] == FORMAT and o["end"] == "time_limit" and o["exit_time"] is None
    (c,) = o["wall_contacts"]
    assert c["wall"] == "wall-left"
    assert c["start"] == pytest.approx(2.4, abs=0.03)  # 0.6 m surface gap at 0.25 m/s
    assert c["depth"] == pytest.approx(0.15, abs=0.01)  # surface at 0.85 m vs face at 0.70 m
    assert o["max_abs_lateral"]["value"] == pytest.approx(0.75, abs=0.01)
    assert o["max_abs_lateral"]["time"] == pytest.approx(3.0, abs=0.03)
    assert o["min_base_height"]["value"] == pytest.approx(0.3)
    assert o["trace"]["columns"] == ["t", "x", "y", "z", "wall_distance"]


def test_exit_crossing_ends_the_trial_and_nothing_is_graded():
    o = trial(1.0, 0.0, seconds=5.0, exit_x=2.0)
    assert o["end"] == "crossed_exit" and o["exit_time"] == pytest.approx(2.0, abs=0.03)
    assert o["wall_contacts"] == []
    text = json.dumps(o)
    for word in ("PASS", "FAIL", "outcome", "grade", "limit"):
        assert word not in text
