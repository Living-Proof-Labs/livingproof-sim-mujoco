# SPDX-License-Identifier: Apache-2.0
"""OnnxPolicy against real MuJoCo physics and a real onnxruntime session."""

from __future__ import annotations

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from livingproof_sim_mujoco.policy import ObsTerm, OnnxPolicy, PolicyError, PolicySpec

ARM = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base">
      <site name="imu"/>
      <body name="a"><joint name="j0" type="hinge" axis="0 1 0" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        <body name="b" pos="0.2 0 0"><joint name="j1" type="hinge" axis="0 1 0" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="m1" joint="j1" kp="20" kv="0.5"/>
    <position name="m0" joint="j0" kp="20" kv="0.5"/>
    <motor name="torque0" joint="j0"/>
  </actuator>
  <sensor><gyro name="gyro" site="imu"/></sensor>
</mujoco>
"""


def write_onnx(path, weights, bias):
    """obs [1,n] -> obs @ W + b, i.e. a real linear policy graph."""
    w = np.asarray(weights, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "W"], ["y"]),
            helper.make_node("Add", ["y", "b"], ["continuous_actions"]),
        ],
        "linear",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, w.shape[0]])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, w.shape[1]])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(np.float32(bias), "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


OBS = [
    ObsTerm("sensor", "gyro"),
    ObsTerm("gravity", "imu"),
    ObsTerm("command"),
    ObsTerm("joint_pos"),
    ObsTerm("joint_vel", scale=0.5),
    ObsTerm("last_action"),
    ObsTerm("phase"),
]
FRAME = 3 + 3 + 3 + 2 + 2 + 2 + 4


def spec(**kw):
    base = dict(
        joints=["j0", "j1"],
        actuators=["m0", "m1"],  # policy order differs from actuator order
        default_pose=[0.1, -0.2],
        action_scale=0.5,
        decimation=10,
        obs=OBS,
        gait_hz=1.5,
        gait_offsets=[0.0, np.pi],
    )
    base.update(kw)
    return PolicySpec(**base)


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(ARM)


def constant(tmp_path, values, n_in=FRAME):
    return write_onnx(tmp_path / "p.onnx", np.zeros((n_in, 2)), values)


def test_targets_follow_joint_map_scale_and_default(model, tmp_path):
    policy = OnnxPolicy(model, constant(tmp_path, [1.0, -1.0]), spec())
    data = mujoco.MjData(model)
    policy.step(data, [0.5, 0, 0])
    assert data.ctrl[model.actuator("m0").id] == pytest.approx(0.1 + 0.5)
    assert data.ctrl[model.actuator("m1").id] == pytest.approx(-0.2 - 0.5)
    assert data.ctrl[model.actuator("torque0").id] == 0


def test_policy_runs_once_per_decimation(model, tmp_path):
    policy = OnnxPolicy(model, constant(tmp_path, [0.0, 0.0]), spec())
    data = mujoco.MjData(model)
    assert policy.run(data, 1.0, [0, 0, 0]) == 500
    assert policy.policy_steps == 50
    assert policy.policy_dt == pytest.approx(0.02)


def test_pd_is_computed_inside_mj_step(model, tmp_path):
    """Targets are set at 50 Hz but the actuator force changes on every physics step."""
    policy = OnnxPolicy(model, constant(tmp_path, [1.0, 1.0]), spec())
    data = mujoco.MjData(model)
    forces = []
    for _ in range(10):  # one policy tick
        policy.step(data, [0, 0, 0])
        forces.append(data.actuator_force[model.actuator("m0").id])
    assert len(set(np.round(forces, 9))) == 10
    policy.run(data, 3.0, [0, 0, 0])
    assert data.joint("j0").qpos[0] == pytest.approx(0.6, abs=0.05)


def test_observation_layout_history_and_phase(model, tmp_path):
    policy = OnnxPolicy(model, constant(tmp_path, [0.0, 0.0], n_in=2 * FRAME), spec(history=2))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.3, 0.4]
    data.qvel[:] = [2.0, 4.0]
    mujoco.mj_forward(model, data)
    obs = policy.observation(data, [0.5, 0.0, 0.1])
    assert obs.shape == (2 * FRAME,)
    np.testing.assert_allclose(obs[:FRAME], obs[FRAME:])  # first frame repeated
    f = obs[FRAME:]
    np.testing.assert_allclose(f[3:6], [0, 0, -1], atol=1e-6)  # gravity in site frame
    np.testing.assert_allclose(f[6:9], [0.5, 0, 0.1])
    np.testing.assert_allclose(f[9:11], [0.2, 0.6], atol=1e-6)  # minus default pose
    np.testing.assert_allclose(f[11:13], [1.0, 2.0])  # scaled
    np.testing.assert_allclose(f[15:19], [1, -1, 0, 0], atol=1e-6)  # cos, sin of [0, pi]
    policy.act(data, [0, 0, 0])
    assert policy.phase[0] == pytest.approx(2 * np.pi * 1.5 * 0.02)
    assert -np.pi <= policy.phase[1] <= np.pi


def test_last_action_feeds_back(model, tmp_path):
    w = np.zeros((FRAME, 2))
    w[13, 0] = 1.0  # action0 = last_action0 + 1 (bias)
    policy = OnnxPolicy(model, write_onnx(tmp_path / "p.onnx", w, [1.0, 0.0]), spec())
    data = mujoco.MjData(model)
    policy.act(data, [0, 0, 0])
    policy.act(data, [0, 0, 0])
    assert policy.last_action[0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (dict(actuators=["torque0", "m1"]), "not a position actuator"),
        (dict(actuators=["m1", "m0"]), "does not drive"),
        (dict(joints=["nope", "j1"]), "no joint 'nope'"),
        (dict(default_pose=[0.0]), "same length"),
        (dict(obs=[ObsTerm("bogus")]), "unknown observation term"),
        (dict(history=3), "ONNX input expects"),
        (dict(decimation=0), "decimation"),
    ],
)
def test_mismatches_are_rejected(model, tmp_path, change, message):
    with pytest.raises(PolicyError, match=message):
        OnnxPolicy(model, constant(tmp_path, [0.0, 0.0]), spec(**change))


FLOAT = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <default><joint armature="0.05"/></default>
  <worldbody>
    <body name="pelvis" pos="0 0 1"><freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="a"><joint name="j0" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/></body>
      <body name="b"><joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.02" mass="0.2"/></body>
      <body name="c"><joint name="arm" type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0.2 0" size="0.02" mass="0.2"/></body>
    </body>
  </worldbody>
  <actuator>
    <motor name="t0" joint="j0"/><motor name="t1" joint="j1"/><motor name="tarm" joint="arm"/>
    <position name="p0" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""

DEPLOY = {
    "control_dt": 0.02,
    "num_actions": 2,
    "num_obs": 9 + 3 * 2 + 2,
    "kps": [30.0, 40.0],
    "kds": [1.0, 2.0],
    "default_angles": [0.1, -0.1],
    "arm_waist_target": [0.3],
    "arm_waist_kps": [20.0],
    "arm_waist_kds": [1.0],
    "ang_vel_scale": 0.25,
    "dof_pos_scale": 1.0,
    "dof_vel_scale": 0.05,
    "action_scale": 0.25,
    "cmd_scale": [2.0, 2.0, 0.25],
}


def rl_spec(**kw):
    from livingproof_sim_mujoco import adapters

    return adapters.rl_gym(
        {**DEPLOY, **kw},
        physics_dt=0.002,
        joints=["j0", "j1"],
        actuators=["t0", "t1"],
        hold_joints=["arm"],
        hold_actuators=["tarm"],
    )


def test_rl_gym_adapter_reads_the_deploy_config():
    s = rl_spec()
    assert s.decimation == 10 and s.actuation == "pd_torque" and s.phase_order == "sin_cos"
    assert list(s.kp) == [30.0, 40.0] and list(s.kd) == [1.0, 2.0]
    assert s.hold == [("arm", "tarm", 0.3, 20.0, 1.0)]
    assert s.gait_hz == pytest.approx(1.25)
    assert s.gait_offsets[0] == pytest.approx(2 * np.pi * 0.02 / 0.8)
    with pytest.raises(ValueError, match="num_obs"):
        rl_spec(num_obs=99)


def test_pd_torque_runs_every_physics_step_from_current_state(tmp_path):
    model = mujoco.MjModel.from_xml_string(FLOAT)
    s = rl_spec()
    policy = OnnxPolicy(model, write_onnx(tmp_path / "p.onnx", np.zeros((17, 2)), [1.0, 0.0]), s)
    data = mujoco.MjData(model)
    torques = []
    for _ in range(10):
        policy.step(data, [0, 0, 0])
        torques.append(data.ctrl[0])
    # ctrl was written before this step's integration, from that step's state
    assert len(set(np.round(torques, 9))) > 1
    assert data.ctrl[3] == 0  # the position actuator is untouched
    policy.run(data, 4.0, [0, 0, 0])
    # equilibrium of the torque PD (kp 30 to 0.35) against the idle position actuator
    # p0 (kp 10 to 0): 30 * (0.35 - q) = 10 * q
    assert data.joint("j0").qpos[0] == pytest.approx(30 * 0.35 / 40, abs=0.005)
    assert data.joint("arm").qpos[0] == pytest.approx(0.3, abs=0.02)


def test_rl_gym_observation_terms(tmp_path):
    model = mujoco.MjModel.from_xml_string(FLOAT)
    policy = OnnxPolicy(
        model, write_onnx(tmp_path / "p.onnx", np.zeros((17, 2)), [0, 0]), rl_spec()
    )
    data = mujoco.MjData(model)
    data.qvel[3:6] = [4.0, 0.0, 8.0]
    data.qvel[6:8] = [20.0, 40.0]
    mujoco.mj_forward(model, data)
    obs = policy.observation(data, [0.5, 0.0, 1.0])
    np.testing.assert_allclose(obs[0:3], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[6:9], [1.0, 0.0, 0.25])
    np.testing.assert_allclose(obs[9:11], [-0.1, 0.1], atol=1e-6)
    np.testing.assert_allclose(obs[11:13], [1.0, 2.0])
    first = 2 * np.pi * 0.02 / 0.8
    np.testing.assert_allclose(obs[15:17], [np.sin(first), np.cos(first)], atol=1e-6)


def test_pd_torque_rejects_position_actuators(tmp_path):
    model = mujoco.MjModel.from_xml_string(FLOAT)
    s = rl_spec()
    bad = PolicySpec(**{**s.__dict__, "actuators": ["p0", "t1"]})
    with pytest.raises(PolicyError, match="not a motor"):
        OnnxPolicy(model, write_onnx(tmp_path / "p.onnx", np.zeros((17, 2)), [0, 0]), bad)


def test_torchscript_backend_keeps_and_resets_recurrent_state(model, tmp_path):
    torch = pytest.importorskip("torch")

    class Recurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(1, 2))

        def forward(self, x):
            self.state.copy_(self.state + 1.0)
            return self.state + 0.0 * x[:, :2]

    path = tmp_path / "p.pt"
    torch.jit.script(Recurrent()).save(str(path))
    policy = OnnxPolicy(model, path, spec())
    data = mujoco.MjData(model)
    policy.act(data, [0, 0, 0])
    policy.act(data, [0, 0, 0])
    assert policy.last_action.tolist() == [2.0, 2.0]  # state carried between calls
    policy.reset()  # also undoes the size probe made at construction
    policy.act(data, [0, 0, 0])
    assert policy.last_action.tolist() == [1.0, 1.0]


def test_torchscript_wrong_observation_size_is_rejected(model, tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "p.pt"
    torch.jit.script(torch.nn.Linear(5, 2)).save(str(path))
    with pytest.raises(PolicyError, match="rejects"):
        OnnxPolicy(model, path, spec())


def test_target_range_clamps_policy_targets(model, tmp_path):
    clamp = spec(target_range=[(-0.2, 0.3), (-1.0, 1.0)])
    policy = OnnxPolicy(model, constant(tmp_path, [1.0, -1.0]), clamp)
    data = mujoco.MjData(model)
    policy.act(data, [0, 0, 0])
    assert data.ctrl[model.actuator("m0").id] == pytest.approx(0.3)
    assert data.ctrl[model.actuator("m1").id] == pytest.approx(-0.7)
