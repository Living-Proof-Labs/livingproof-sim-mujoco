# SPDX-License-Identifier: Apache-2.0
"""livingproof-worlds v1.4.0 hallway-corner 0.2.0 (livingproof-world/0.3) loads and composes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import pytest

from livingproof_sim_mujoco import compose, hallway
from livingproof_sim_mujoco.scene import place_pose
from livingproof_sim_mujoco.walk import initial_state

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "livingproof-worlds-v1.4.0"
    / "hallway-corner"
    / "0.2.0"
    / "world.json"
)
# livingproof-worlds v1.4.0, commit 971b39ee845a18c50da627a7410d22e3b98171f9
WORLD_SHA256 = "f398f2705e630750f7b7ef3035ba4b5aed076de1f417165e58540e973150678e"

FORWARD = {
    "name": "forward",
    "from": "end-west",
    "start": "spawn-west",
    "route": "route-forward",
    "finish": "end-north",
}
REVERSE = {
    "name": "reverse",
    "from": "end-north",
    "start": "spawn-north",
    "route": "route-reverse",
    "finish": "end-west",
}


def _world() -> dict:
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == WORLD_SHA256
    return json.loads(raw)


class _Scene:
    def __init__(self, model):
        self.model = model


class _Ctrl:
    def __init__(self, model):
        self.initial_qpos = model.qpos0.copy()
        self.spec = type("Spec", (), {"actuation": "pd_torque"})()


class _Robot:
    id = "world-format-0.3"

    def scene(self, name):
        del name
        spec = mujoco.MjSpec.from_string(
            '<mujoco><option timestep="0.002"/><worldbody><body name="base" pos="0 0 0.3">'
            '<freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        return _Scene(spec.compile())

    def controller(self, name, scene):
        del name
        return _Ctrl(scene.model)

    def base_body(self, name):
        del name
        return "base"

    def initial_state(self, model, data, ctrl, spawn, *, start_offset=None):
        initial_state(model, data, ctrl, spawn, start_offset=start_offset)


def test_v1_4_0_hallway_corner_0_2_0_loads_and_composes_named_places():
    """Format 0.3 with traversals is accepted; start, route and finish are in the model."""
    world = _world()
    assert world["format"] == "livingproof-world/0.3"
    assert world["id"] == "hallway-corner"
    assert world["content_version"] == "0.2.0"
    assert world["traversals"] == [FORWARD, REVERSE]
    for item in world["traversals"]:
        assert set(item) == {"name", "from", "start", "route", "finish"}

    hallway.supported(world)
    assert world["traversals"] == [FORWARD, REVERSE]

    model, data, ctrl, walls = compose.place(_Robot(), "x", world, "spawn-west")
    assert "traversals" in world
    assert world["traversals"] == [FORWARD, REVERSE]
    assert walls  # the composed model has the declared walls
    for name in (
        "spawn-west",
        "spawn-north",
        "route-forward",
        "route-reverse",
        "end-north",
        "end-west",
    ):
        position, yaw = place_pose(model, name)
        assert position is not None

    west, west_yaw = place_pose(model, "spawn-west")
    assert west[0] == pytest.approx(-2.5)
    assert west[1] == pytest.approx(0.0)
    assert west_yaw == pytest.approx(0.0)

    north, north_yaw = place_pose(model, "spawn-north")
    assert north[0] == pytest.approx(0.0)
    assert north[1] == pytest.approx(2.5)
    assert north_yaw == pytest.approx(-90.0 * 3.141592653589793 / 180.0)

    end_north, _ = place_pose(model, "end-north")
    assert end_north[0] == pytest.approx(0.0)
    assert end_north[1] == pytest.approx(3.0)

    end_west, _ = place_pose(model, "end-west")
    assert end_west[0] == pytest.approx(-3.0)
    assert end_west[1] == pytest.approx(0.0)

    base = data.xpos[model.body("base").id]
    assert base[0] == pytest.approx(-2.5)
    assert base[1] == pytest.approx(0.0)

    mujoco.mj_resetData(model, data)
    initial_state(model, data, ctrl, "spawn-north")
    base = data.xpos[model.body("base").id]
    assert base[0] == pytest.approx(0.0)
    assert base[1] == pytest.approx(2.5)
    _, yaw = place_pose(model, "spawn-north")
    # yaw from qpos
    w, _, _, z = data.qpos[3:7]
    import math

    assert 2 * math.atan2(z, w) == pytest.approx(yaw)
