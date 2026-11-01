# SPDX-License-Identifier: Apache-2.0
"""Build a ``PolicySpec`` from a published controller's own deploy configuration.

Each adapter takes the upstream values unchanged, so the numbers stay in the upstream
file rather than being copied into code. The robot package supplies only the mapping
from upstream joint indices to model joint and actuator names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from livingproof_sim_mujoco.policy import ObsTerm, PolicySpec


def rl_gym(
    deploy: Mapping,
    *,
    physics_dt: float,
    joints: Sequence[str],
    actuators: Sequence[str],
    hold_joints: Sequence[str] = (),
    hold_actuators: Sequence[str] = (),
    gait_period: float = 0.8,
) -> PolicySpec:
    """unitree_rl_gym ``deploy_real/configs/<robot>.yaml`` plus ``deploy_mujoco.py``.

    Observation (``num_obs``): base angular velocity * ang_vel_scale, projected gravity
    from the pelvis quaternion, command * cmd_scale, (q - default) * dof_pos_scale,
    dq * dof_vel_scale, last action, then [sin, cos] of a ``gait_period`` phase
    (0.8 s in deploy_mujoco.py and deploy_real.py). Targets default + action *
    action_scale, tracked by a torque PD with ``kps``/``kds``; arm and waist joints are
    held at ``arm_waist_target`` with ``arm_waist_kps``/``arm_waist_kds``.
    """
    n = int(deploy["num_actions"])
    if len(joints) != n or len(actuators) != n:
        raise ValueError(f"num_actions is {n}; got {len(joints)} joints")
    decimation = round(float(deploy["control_dt"]) / physics_dt)
    if not math.isclose(decimation * physics_dt, float(deploy["control_dt"])):
        raise ValueError("control_dt is not a whole number of physics steps")
    hold = []
    if hold_joints:
        targets, kps, kds = (
            deploy[k] for k in ("arm_waist_target", "arm_waist_kps", "arm_waist_kds")
        )
        if not (len(hold_joints) == len(hold_actuators) == len(targets)):
            raise ValueError("hold joints must match arm_waist_target")
        hold = [
            (j, a, float(t), float(p), float(d))
            for j, a, t, p, d in zip(hold_joints, hold_actuators, targets, kps, kds)
        ]
    # deploy_mujoco.py evaluates the policy after `counter` physics steps, so the first
    # phase is one control period in: phase = (k * control_dt % period) / period, k >= 1.
    rate = 1.0 / gait_period
    first = 2 * math.pi * rate * float(deploy["control_dt"])
    spec = PolicySpec(
        joints=list(joints),
        actuators=list(actuators),
        default_pose=[float(v) for v in deploy["default_angles"]],
        action_scale=float(deploy["action_scale"]),
        decimation=decimation,
        obs=[
            ObsTerm("base_angvel", scale=float(deploy["ang_vel_scale"])),
            ObsTerm("gravity", "body:" + deploy.get("imu_body", "pelvis")),
            ObsTerm("command", scale=[float(v) for v in deploy["cmd_scale"]]),
            ObsTerm("joint_pos", scale=float(deploy["dof_pos_scale"])),
            ObsTerm("joint_vel", scale=float(deploy["dof_vel_scale"])),
            ObsTerm("last_action"),
            ObsTerm("phase"),
        ],
        gait_hz=rate,
        gait_offsets=[math.remainder(first, 2 * math.pi)],
        phase_order="sin_cos",
        actuation="pd_torque",
        kp=[float(v) for v in deploy["kps"]],
        kd=[float(v) for v in deploy["kds"]],
        hold=hold,
    )
    if 9 + 3 * n + 2 != int(deploy["num_obs"]):
        raise ValueError("num_obs does not match the rl_gym observation layout")
    return spec


def playground_joystick(
    *,
    physics_dt: float,
    ctrl_dt: float,
    joints: Sequence[str],
    actuators: Sequence[str],
    default_pose: Sequence[float],
    action_scale: float,
    gait_hz: float,
    gait_offsets: Sequence[float],
    linvel_sensor: str,
    gyro_sensor: str,
    imu_site: str,
    actuation: str = "position",
    kp: Sequence[float] = (),
    kd: Sequence[float] = (),
) -> PolicySpec:
    """MuJoCo Playground joystick policies (``experimental/sim2sim``).

    Observation: local linear velocity, gyro, gravity in the IMU site frame, command,
    q - default, dq, last action, [cos, sin] of each foot's gait phase.
    """
    decimation = round(ctrl_dt / physics_dt)
    if not math.isclose(decimation * physics_dt, ctrl_dt):
        raise ValueError("ctrl_dt is not a whole number of physics steps")
    return PolicySpec(
        joints=list(joints),
        actuators=list(actuators),
        default_pose=list(default_pose),
        action_scale=action_scale,
        decimation=decimation,
        obs=[
            ObsTerm("sensor", linvel_sensor),
            ObsTerm("sensor", gyro_sensor),
            ObsTerm("gravity", imu_site),
            ObsTerm("command"),
            ObsTerm("joint_pos"),
            ObsTerm("joint_vel"),
            ObsTerm("last_action"),
            ObsTerm("phase"),
        ],
        gait_hz=gait_hz,
        gait_offsets=list(gait_offsets),
        phase_order="cos_sin",
        actuation=actuation,
        kp=list(kp),
        kd=list(kd),
    )
