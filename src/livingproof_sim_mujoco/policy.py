# SPDX-License-Identifier: Apache-2.0
"""Run a published ONNX locomotion policy in native MuJoCo.

The policy runs every ``decimation`` physics steps, for example 50 Hz over a 500 Hz step,
and produces joint-position targets ``default_pose + action_scale * action``. The PD
loop that tracks those targets always runs at the physics rate, from current state:

- ``actuation="position"``: targets go into ``data.ctrl`` of MuJoCo position actuators,
  so ``kp * (target - q) - kv * qdot`` is evaluated inside ``mj_step``.
- ``actuation="pd_torque"``: motor actuators; each physics step runs ``mj_step1``,
  writes ``kp * (target - q) - kd * qdot`` into ``data.ctrl``, then ``mj_step2``. This
  matches a motor driver closing the loop on fresh measurements, and is the fix used in
  Living-Proof-Labs/unitree-foxglove-sdk (``MotorFirmwareBridge``). Joints the policy
  does not drive can be held at a fixed target with their own gains (``hold``).

Engine-specific but robot-neutral: the observation spec, action scale, default pose,
decimation and joint map are configuration (see ``PolicySpec``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path

import mujoco
import numpy as np


class PolicyError(ValueError):
    """The policy, model and spec do not fit together."""


class OnnxBackend:
    """onnxruntime CPU session; stateless."""

    def __init__(self, path: Path, input_name: str, output_name: str | None):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        if input_name not in inputs:
            raise PolicyError(f"ONNX model has no input {input_name!r} ({sorted(inputs)})")
        self.input_name = input_name
        self.output_name = output_name or self.session.get_outputs()[0].name
        size = inputs[input_name].shape[-1]
        self.input_size = size if isinstance(size, int) else None

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: obs[None, :]})[0][0]


class TorchScriptBackend:
    """TorchScript module on the CPU (needs the ``torch`` extra).

    Recurrent policies such as unitree_rl_gym's ``motion.pt`` keep their LSTM state in
    module buffers that ``forward`` updates in place, which an ONNX export does not
    preserve; ``reset`` reloads the module so each episode starts from its saved state.
    """

    def __init__(self, path: Path):
        import torch

        torch.set_num_threads(1)
        self._torch, self._path = torch, path
        self.input_size = None
        self.reset()

    def reset(self) -> None:
        self._module = self._torch.jit.load(str(self._path), map_location="cpu").eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            out = self._module(self._torch.from_numpy(obs).unsqueeze(0))
        return out.numpy().reshape(-1)


def load_backend(path: Path, input_name: str = "obs", output_name: str | None = None):
    """``.onnx`` runs in onnxruntime; ``.pt``/``.jit`` TorchScript runs in torch."""
    if path.suffix in (".pt", ".jit"):
        return TorchScriptBackend(path)
    return OnnxBackend(path, input_name, output_name)


@dataclass(frozen=True)
class ObsTerm:
    """One observation term, in policy order.

    kind:
      - ``sensor``: ``data.sensor(source)``
      - ``gravity``: world -z in the frame of site ``source`` (or ``body:<name>``)
      - ``base_angvel``: angular velocity of the model's free joint, body frame
      - ``command``: the command vector passed to ``step``
      - ``joint_pos``: joint positions minus ``default_pose``, in policy joint order
      - ``joint_vel``: joint velocities, in policy joint order
      - ``last_action``: the previous raw policy output
      - ``phase``: ``[cos(phase), sin(phase)]`` for each gait offset (see ``PolicySpec``)
    """

    kind: str
    source: str | None = None
    scale: float | Sequence[float] = 1.0


KINDS = {
    "sensor",
    "gravity",
    "base_angvel",
    "command",
    "joint_pos",
    "joint_vel",
    "last_action",
    "phase",
}


@dataclass(frozen=True)
class PolicySpec:
    joints: Sequence[str]  # joint names in policy order (the joint map)
    actuators: Sequence[str]  # position actuator driving each joint, same order
    default_pose: Sequence[float]  # policy order
    action_scale: float | Sequence[float]
    decimation: int  # physics steps per policy step
    obs: Sequence[ObsTerm]
    command_size: int = 3
    history: int = 1  # stacked frames, newest last (oldest repeats the first frame)
    gait_hz: float = 0.0  # phase term only
    gait_offsets: Sequence[float] = field(default_factory=tuple)
    phase_order: str = "cos_sin"  # or "sin_cos"
    actuation: str = "position"  # or "pd_torque"
    kp: Sequence[float] = field(default_factory=tuple)  # pd_torque, policy order
    kd: Sequence[float] = field(default_factory=tuple)
    hold: Sequence[tuple[str, str, float, float, float]] = field(default_factory=tuple)
    # pd_torque only: (joint, actuator, target, kp, kd) for joints outside the policy
    target_range: Sequence[tuple[float, float]] = field(default_factory=tuple)
    # optional per-joint clamp of policy targets, as a position actuator's ctrlrange does
    input_name: str = "obs"
    output_name: str | None = None  # default: the model's first output


class OnnxPolicy:
    """Bind an ONNX policy to a MuJoCo model; ``step`` advances one physics step."""

    def __init__(self, model: mujoco.MjModel, policy_path: str | Path, spec: PolicySpec):
        self.model, self.spec = model, spec
        n = len(spec.joints)
        if not (len(spec.actuators) == len(spec.default_pose) == n):
            raise PolicyError("joints, actuators and default_pose must have the same length")
        if spec.decimation < 1:
            raise PolicyError("decimation must be >= 1")
        if spec.actuation not in ("position", "pd_torque"):
            raise PolicyError(f"unknown actuation {spec.actuation!r}")
        if spec.phase_order not in ("cos_sin", "sin_cos"):
            raise PolicyError(f"unknown phase_order {spec.phase_order!r}")
        torque = spec.actuation == "pd_torque"
        if torque and not (len(spec.kp) == len(spec.kd) == n):
            raise PolicyError("pd_torque needs kp and kd for every policy joint")
        if not torque and spec.hold:
            raise PolicyError("hold joints need actuation='pd_torque'")
        for term in spec.obs:
            if term.kind not in KINDS:
                raise PolicyError(f"unknown observation term {term.kind!r}")
        self._qpos, self._qvel, self._act = [], [], []
        joints = list(spec.joints) + [h[0] for h in spec.hold]
        actuators = list(spec.actuators) + [h[1] for h in spec.hold]
        for joint, actuator in zip(joints, actuators):
            j = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint)
            a = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            if int(model.jnt_type[j]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise PolicyError(f"joint {joint!r} is not a hinge or slide joint")
            if model.actuator_trnid[a][0] != j:
                raise PolicyError(f"actuator {actuator!r} does not drive joint {joint!r}")
            if torque and not self.is_motor_actuator(model, a):
                raise PolicyError(f"actuator {actuator!r} is not a motor; pd_torque writes torque")
            if not torque and not self.is_position_actuator(model, a):
                raise PolicyError(
                    f"actuator {actuator!r} is not a position actuator; the PD must run "
                    "inside mj_step"
                )
            self._qpos.append(model.jnt_qposadr[j])
            self._qvel.append(model.jnt_dofadr[j])
            self._act.append(a)
        self.n = n
        self.default_pose = np.asarray(spec.default_pose, dtype=np.float64)
        self._kp = np.array(list(spec.kp) + [h[3] for h in spec.hold], dtype=np.float64)
        self._kd = np.array(list(spec.kd) + [h[4] for h in spec.hold], dtype=np.float64)
        if spec.target_range and len(spec.target_range) != n:
            raise PolicyError("target_range needs one (low, high) per policy joint")
        rng = np.asarray(spec.target_range, dtype=np.float64).reshape(-1, 2)
        self._lo, self._hi = (rng[:, 0], rng[:, 1]) if len(rng) else (-np.inf, np.inf)
        self._hold_target = np.array([h[2] for h in spec.hold], dtype=np.float64)
        self._free = next(
            (
                i
                for i in range(model.njnt)
                if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)
            ),
            None,
        )
        self.action_scale = np.broadcast_to(
            np.asarray(spec.action_scale, dtype=np.float64), (n,)
        ).copy()
        self.backend = load_backend(Path(policy_path), spec.input_name, spec.output_name)
        want = self.backend.input_size
        self.frame_size = sum(self._term_size(t) for t in spec.obs)
        size = self.frame_size * spec.history
        if want is not None and want != size:
            raise PolicyError(
                f"observation spec gives {self.frame_size} x {spec.history} history = "
                f"{size}, ONNX input expects {want}"
            )
        if want is None:  # TorchScript: probe once, then restore the saved state
            try:
                probe = self.backend(np.zeros(size, dtype=np.float32))
            except Exception as error:  # noqa: BLE001 - torch raises many types
                raise PolicyError(f"policy rejects a {size}-value observation: {error}") from error
            if probe.shape != (n,):
                raise PolicyError(f"policy returns {probe.shape[0]} actions, spec has {n} joints")
        self.policy_dt = spec.decimation * model.opt.timestep
        self.reset()

    @staticmethod
    def is_position_actuator(model: mujoco.MjModel, a: int) -> bool:
        """gain kp, bias [0, -kp, -kv]: MuJoCo's <position> actuator."""
        kp = model.actuator_gainprm[a][0]
        return (
            int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            and int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
            and int(model.actuator_dyntype[a]) == int(mujoco.mjtDyn.mjDYN_NONE)
            and kp > 0
            and math.isclose(model.actuator_biasprm[a][1], -kp)
        )

    @staticmethod
    def is_motor_actuator(model: mujoco.MjModel, a: int) -> bool:
        """gain 1, no bias: MuJoCo's <motor> actuator (ctrl is torque)."""
        return (
            int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            and int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE)
            and int(model.actuator_dyntype[a]) == int(mujoco.mjtDyn.mjDYN_NONE)
            and model.actuator_gainprm[a][0] == 1.0
        )

    def _id(self, kind, name: str) -> int:
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            raise PolicyError(f"model has no {kind.name.split('_')[-1].lower()} {name!r}")
        return i

    def _term_size(self, term: ObsTerm) -> int:
        n = len(self.spec.joints)
        if term.kind == "sensor":
            return int(self.model.sensor_dim[self._id(mujoco.mjtObj.mjOBJ_SENSOR, term.source)])
        if term.kind == "gravity":
            if term.source.startswith("body:"):
                self._id(mujoco.mjtObj.mjOBJ_BODY, term.source[5:])
            else:
                self._id(mujoco.mjtObj.mjOBJ_SITE, term.source)
            return 3
        if term.kind == "base_angvel":
            if self._free is None:
                raise PolicyError("base_angvel needs a free joint")
            return 3
        if term.kind == "command":
            return self.spec.command_size
        if term.kind == "phase":
            return 2 * len(self.spec.gait_offsets)
        return n

    def reset(self) -> None:
        self.backend.reset()
        self.counter = 0
        self.policy_steps = 0
        self.last_action = np.zeros(len(self.spec.joints), dtype=np.float32)
        self.target = np.concatenate([self.default_pose, self._hold_target])
        self.phase = np.asarray(self.spec.gait_offsets, dtype=np.float64)
        self._history: deque[np.ndarray] = deque(maxlen=self.spec.history)

    def frame(self, data: mujoco.MjData, command: Sequence[float]) -> np.ndarray:
        """One observation frame (no history), float32, in spec order."""
        parts = []
        for term in self.spec.obs:
            if term.kind == "sensor":
                v = data.sensor(term.source).data
            elif term.kind == "gravity":
                if term.source.startswith("body:"):
                    xmat = data.body(term.source[5:]).xmat.reshape(3, 3)
                else:
                    xmat = data.site(term.source).xmat.reshape(3, 3)
                v = xmat.T @ np.array([0.0, 0.0, -1.0])
            elif term.kind == "base_angvel":
                adr = self.model.jnt_dofadr[self._free]
                v = data.qvel[adr + 3 : adr + 6]
            elif term.kind == "command":
                v = np.asarray(command, dtype=np.float64)
                if v.shape != (self.spec.command_size,):
                    raise PolicyError(f"command must have {self.spec.command_size} values")
            elif term.kind == "joint_pos":
                v = data.qpos[self._qpos[: self.n]] - self.default_pose
            elif term.kind == "joint_vel":
                v = data.qvel[self._qvel[: self.n]]
            elif term.kind == "last_action":
                v = self.last_action
            else:
                c, s = np.cos(self.phase), np.sin(self.phase)
                v = np.concatenate([c, s] if self.spec.phase_order == "cos_sin" else [s, c])
            parts.append(np.asarray(v, dtype=np.float64) * np.asarray(term.scale))
        return np.concatenate(parts).astype(np.float32)

    def observation(self, data: mujoco.MjData, command: Sequence[float]) -> np.ndarray:
        frame = self.frame(data, command)
        if not self._history:
            self._history.extend([frame] * self.spec.history)
        else:
            self._history.append(frame)
        return np.concatenate(self._history)

    def act(self, data: mujoco.MjData, command: Sequence[float]) -> np.ndarray:
        """Run the policy once and write position targets into data.ctrl."""
        obs = self.observation(data, command)
        out = self.backend(obs)
        self.last_action = np.asarray(out, dtype=np.float32).copy()
        raw = self.default_pose + self.action_scale * self.last_action
        self.target[: self.n] = np.clip(raw, self._lo, self._hi)
        if self.spec.actuation == "position":
            data.ctrl[self._act] = self.target
        if self.spec.gait_offsets:
            step = 2 * math.pi * self.spec.gait_hz * self.policy_dt
            self.phase = np.fmod(self.phase + step + math.pi, 2 * math.pi) - math.pi
        self.policy_steps += 1
        return self.last_action

    def step(self, data: mujoco.MjData, command: Sequence[float]) -> None:
        """One physics step; the policy acts first on every ``decimation``-th step."""
        if self.counter % self.spec.decimation == 0:
            self.act(data, command)
        if self.spec.actuation == "position":
            mujoco.mj_step(self.model, data)  # position actuators: PD computed in here
        else:
            mujoco.mj_step1(self.model, data)  # state and sensors current for this step
            q, dq = data.qpos[self._qpos], data.qvel[self._qvel]
            data.ctrl[self._act] = self._kp * (self.target - q) - self._kd * dq
            mujoco.mj_step2(self.model, data)
        self.counter += 1

    def run(
        self,
        data: mujoco.MjData,
        seconds: float,
        command: Callable[[float], Sequence[float]] | Sequence[float],
        on_step: Callable[[mujoco.MjData], bool | None] | None = None,
    ) -> int:
        """Step for ``seconds`` of simulated time; ``on_step`` returning True stops early."""
        get = command if callable(command) else (lambda _t, c=command: c)
        steps = int(round(seconds / self.model.opt.timestep))
        for i in range(steps):
            self.step(data, get(data.time))
            if on_step is not None and on_step(data):
                return i + 1
        return steps
