"""Generic scene, walk and measure helpers shared by the robot packages."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from livingproof_sim_mujoco import measure, scene, walk

WORLD = {
    "geometry": [
        {"role": "wall", "name": "wall-left", "size": [4, 0.1, 1], "centre": [0, 0.85, 0.5]},
        {"role": "floor", "name": "floor", "size": [4, 2, 0.1], "centre": [0, 0, 0]},
    ],
    "places": [
        {"name": "spawn-entrance", "position": [-0.5, 0.2, 0.0], "yaw_deg": 90},
        {"name": "exit", "reference": [0.5, 0.0, 0.0]},
    ],
}
XML = """<mujoco><worldbody><body name="base" pos="0 0 0.5"><freejoint/>
<geom type="box" size="0.1 0.2 0.3"/><body pos="0 0 -0.3"><joint name="hip" type="hinge"/>
<geom type="capsule" size="0.03 0.1"/></body></body></worldbody>
<actuator><position name="hip" joint="hip" kp="40" ctrlrange="-1 1"/></actuator>
<keyframe><key name="home" qpos="0 0 0.5 1 0 0 0 0.2"/></keyframe></mujoco>"""


def compiled():
    spec = mujoco.MjSpec.from_string(XML)
    walls = scene.add_world(spec, WORLD)
    return spec, walls, spec.compile()


def test_world_walls_are_visual_only_and_places_are_sites():
    _, walls, m = compiled()
    assert walls == ["wall-left"]
    g = m.geom("wall-left")
    assert m.geom_contype[g.id] == 0 and m.geom_conaffinity[g.id] == 0
    pos, yaw = scene.place_pose(m, "spawn-entrance")
    assert np.allclose(pos, [-0.5, 0.2, 0.0]) and yaw == pytest.approx(math.pi / 2)
    assert np.allclose(scene.place_pose(m, "exit")[0], [0.5, 0, 0])


def test_actuator_table_and_pd_torque():
    spec, _, m = compiled()
    kp, ctrlrange = scene.actuator_table(m)
    assert kp == {"hip": 40.0} and ctrlrange == {"hip": (-1.0, 1.0)}
    scene.to_pd_torque(spec)
    t = spec.compile()
    assert t.actuator_gainprm[0][0] == 1.0 and t.actuator_ctrllimited[0] == 0


def test_initial_state_is_the_nominal_pose_and_simulate_records_samples():
    _, _, m = compiled()
    ctrl = SimpleNamespace(
        initial_qpos=m.key("home").qpos.copy(),
        spec=SimpleNamespace(actuation="position", actuators=["hip"], default_pose=[0.2]),
    )
    noise = dict(
        joint_pos_rad=0.1, joint_vel_rad_s=0.1, base_xy_m=0.01, base_yaw_rad=0.05, base_vel_m_s=0.0
    )
    a, b, c = (mujoco.MjData(m) for _ in range(3))
    walk.initial_state(m, a, ctrl, "spawn-entrance")
    walk.initial_state(m, b, ctrl, 3, noise, "spawn-entrance")
    assert np.array_equal(a.qpos, b.qpos) and np.all(a.qvel == 0) and a.ctrl[0] == 0.2
    pos, _yaw = scene.place_pose(m, "spawn-entrance")
    assert np.allclose(a.qpos[0:2], pos[0:2])
    walk.initial_state(m, c, ctrl, "spawn-entrance", start_offset={"base_xy_m": [0.05, -0.01]})
    assert np.allclose(c.qpos[0:2], pos[0:2] + np.array([0.05, -0.01]))
    p = {"exit_x": 0.5, "fall_height": 0.3}
    samples, end, clip = walk.simulate(
        m, a, lambda d: mujoco.mj_step(m, d), base_body="base", seconds=0.2, p=p, sample_hz=50
    )
    assert samples[0][0] == 0.0 and end == "time_limit" and clip is None
    s = walk.stand_summary("stock", "x", 0.2, samples, end, clip, 0.3)
    assert s["standing"] and "seed" not in s


def test_measure_extents_over_keyframes():
    xml = XML.replace('type="box" size="0.1 0.2 0.3"', 'type="mesh" mesh="cube"').replace(
        "<worldbody>",
        '<asset><mesh name="cube" vertex="-.1 -.2 -.3 .1 -.2 -.3 -.1 .2 -.3 .1 .2 -.3 '
        '-.1 -.2 .3 .1 -.2 .3 -.1 .2 .3 .1 .2 .3"/></asset><worldbody>',
    )
    m = mujoco.MjModel.from_xml_string(xml)
    r = measure.keyframe_extents(m, ("home",))
    assert (r["width_m"], r["length_m"]) == (0.4, 0.2) and r["keyframes"][0]["keyframe"] == "home"
