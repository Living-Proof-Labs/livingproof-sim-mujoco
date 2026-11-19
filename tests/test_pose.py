# SPDX-License-Identifier: Apache-2.0
"""Tests for the livingproof_sim_mujoco Pose API and MuJoCo pose provider."""

from __future__ import annotations

import math

import mujoco
import pytest

from livingproof_sim_mujoco import hallway, scene
from livingproof_sim_mujoco.pose import (
    MujocoPoseProvider,
    PoseProvider,
    PoseSample,
    PoseUnavailableError,
    get_pose,
    quat_to_yaw,
    set_pose,
    set_pose_at_place,
    validate_pose_sample,
    yaw_to_quat,
)

# Deliberately includes a hinge-jointed sibling before and child after the floating base
# so tests prove set_pose/get_pose locate the named body's freejoint by body id, not qpos[0:7].
MULTI_BODY_XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="floor" type="plane" size="10 10 0.1"/>
<body name="anchor" pos="2 2 0.2">
  <joint name="anchor_hinge" type="hinge" axis="0 0 1"/>
  <geom type="sphere" size="0.05"/>
</body>
<body name="robot_base" pos="0 0 0.4">
  <freejoint name="root_free"/>
  <geom type="sphere" size="0.1" mass="1"/>
  <body name="arm" pos="0.1 0 0">
    <joint name="arm_hinge" type="hinge" axis="0 1 0"/>
    <geom type="sphere" size="0.04" mass="0.2"/>
  </body>
</body>
</worldbody></mujoco>"""


def _model_and_data():
    model = mujoco.MjModel.from_xml_string(MULTI_BODY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_pose_sample_constructors_properties_and_serialization():
    yaw = math.pi / 3.0
    sample = PoseSample.from_yaw(1.25, -0.5, 0.78, yaw, timestamp_s=2.5, source="unit-test")
    assert sample.xy == (1.25, -0.5)
    assert sample.position == (1.25, -0.5, 0.78)
    assert sample.xy_yaw == (1.25, -0.5, yaw)
    assert sample.frame_id == "world"
    assert sample.source == "unit-test"
    assert sample.clock == "simulation"
    assert sample.timestamp_s == 2.5
    assert quat_to_yaw(*sample.orientation) == pytest.approx(yaw)

    as_dict = sample.to_dict()
    assert as_dict["x"] == 1.25
    assert as_dict["y"] == -0.5
    assert as_dict["z"] == 0.78
    assert as_dict["yaw"] == pytest.approx(yaw)
    assert as_dict["frame_id"] == "world"
    assert as_dict["source"] == "unit-test"
    assert as_dict["clock"] == "simulation"
    assert as_dict["orientation"] == {
        "w": sample.qw,
        "x": sample.qx,
        "y": sample.qy,
        "z": sample.qz,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": float("nan"), "y": 0.0, "z": 0.0, "yaw": 0.0},
        {"x": 0.0, "y": float("inf"), "z": 0.0, "yaw": 0.0},
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "frame_id": ""},
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "source": ""},
    ],
)
def test_pose_sample_rejects_invalid_fields(kwargs):
    with pytest.raises(PoseUnavailableError):
        PoseSample(**kwargs)


def test_validate_pose_sample_checks_presence_frame_and_staleness():
    with pytest.raises(PoseUnavailableError, match="returned None"):
        validate_pose_sample(None)

    wrong_frame = PoseSample.from_yaw(0.0, 0.0, 0.5, 0.0, frame_id="odom")
    with pytest.raises(PoseUnavailableError, match="frame_id"):
        validate_pose_sample(wrong_frame, expected_frame="world")

    stale = PoseSample.from_yaw(0.0, 0.0, 0.5, 0.0, timestamp_s=1.0)
    with pytest.raises(PoseUnavailableError, match="stale"):
        validate_pose_sample(stale, expected_frame="world", max_age_s=0.1, now_s=1.25)

    fresh = validate_pose_sample(stale, expected_frame="world", max_age_s=0.5, now_s=1.25)
    assert fresh == stale


def test_set_pose_and_get_pose_round_trip_with_non_root_freejoint():
    model, data = _model_and_data()
    arm_dof = int(model.jnt_dofadr[model.joint("arm_hinge").id])
    root_dof = int(model.jnt_dofadr[model.joint("root_free").id])
    data.qvel[arm_dof] = 1.75
    data.qvel[root_dof : root_dof + 6] = [0.4, -0.2, 0.1, 0.0, 0.0, 0.5]

    placed = set_pose(model, data, "robot_base", x=-1.5, y=0.75, z=0.65, yaw=-math.pi / 2)
    assert placed.x == pytest.approx(-1.5)
    assert placed.y == pytest.approx(0.75)
    assert placed.z == pytest.approx(0.65)
    assert placed.yaw == pytest.approx(-math.pi / 2)
    assert placed.orientation == pytest.approx(yaw_to_quat(-math.pi / 2))
    assert data.qvel[root_dof : root_dof + 6] == pytest.approx([0.0] * 6)
    assert data.qvel[arm_dof] == pytest.approx(1.75)  # child joint velocity left intact

    queried = get_pose(model, data, "robot_base")
    assert queried == placed

    # Omitting z keeps existing z; zero_vel=False keeps freejoint velocity.
    data.qvel[root_dof] = 0.9
    moved = set_pose(model, data, "robot_base", x=0.25, y=-0.4, yaw=0.3, zero_vel=False)
    assert moved.z == pytest.approx(0.65)
    assert moved.yaw == pytest.approx(0.3)
    assert data.qvel[root_dof] == pytest.approx(0.9)

    # Explicit non-normalized quaternion is normalized cleanly.
    q = yaw_to_quat(0.8)
    scaled = tuple(2.0 * v for v in q)
    with_quat = set_pose(model, data, "robot_base", x=0.0, y=0.0, z=0.5, quat=scaled)
    assert with_quat.orientation == pytest.approx(q)
    assert with_quat.yaw == pytest.approx(0.8)


def test_set_pose_at_place_teleports_with_optional_offset():
    world = hallway.load_fixture("hallway-corner", "0.1.0")["world"]
    spec = mujoco.MjSpec.from_string(MULTI_BODY_XML)
    scene.add_world(spec, world)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sample = set_pose_at_place(
        model,
        data,
        "robot_base",
        "spawn-entrance",
        z=0.55,
        offset={"base_xy_m": [0.1, -0.05], "base_yaw_rad": 0.2},
    )
    site_pos, site_yaw = scene.place_pose(model, "spawn-entrance")
    assert sample.x == pytest.approx(float(site_pos[0]) + 0.1)
    assert sample.y == pytest.approx(float(site_pos[1]) - 0.05)
    assert sample.z == pytest.approx(0.55)
    assert sample.yaw == pytest.approx(site_yaw + 0.2)

    with pytest.raises(PoseUnavailableError, match="does not exist"):
        set_pose_at_place(model, data, "robot_base", "missing-place")


def test_pose_api_fails_explicitly_on_invalid_body_joint_or_state():
    model, data = _model_and_data()

    with pytest.raises(PoseUnavailableError, match="does not exist"):
        get_pose(model, data, "no_such_body")
    with pytest.raises(PoseUnavailableError, match="out of range"):
        get_pose(model, data, 999)
    with pytest.raises(PoseUnavailableError, match="None"):
        get_pose(model, None, "robot_base")
    with pytest.raises(PoseUnavailableError, match="no free joint"):
        set_pose(model, data, "arm", x=0.0, y=0.0)
    with pytest.raises(PoseUnavailableError, match="non-finite"):
        set_pose(model, data, "robot_base", x=float("nan"), y=0.0)

    data.xpos[model.body("robot_base").id, 0] = float("nan")
    with pytest.raises(PoseUnavailableError, match="Non-finite"):
        get_pose(model, data, "robot_base")


def test_mujoco_pose_provider_protocol_read_set_and_staleness():
    model, data = _model_and_data()
    provider = MujocoPoseProvider(model, "robot_base", source="sim-ideal", max_age_s=0.05)
    assert isinstance(provider, PoseProvider)

    set_sample = provider.set_pose(data, x=1.0, y=-2.0, z=0.45, yaw=1.1)
    assert set_sample.source == "sim-ideal"
    assert set_sample.frame_id == "world"

    read_sample = provider.read(data)
    assert read_sample == set_sample
    assert read_sample.xy_yaw == pytest.approx((1.0, -2.0, 1.1))
