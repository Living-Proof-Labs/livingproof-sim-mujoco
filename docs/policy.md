# Policy runner

`livingproof_sim_mujoco.policy.OnnxPolicy` runs a published ONNX locomotion policy in
native MuJoCo. It is engine-specific but robot-neutral; the robot package supplies the
configuration.

## Timing

The policy acts every `decimation` physics steps, for example 50 Hz over a 0.002 s
(500 Hz) step with `decimation=10`. Every physics step runs `mj_step1` first, so the policy observes positions, velocities and sensors of the current instant (as a MuJoCo control callback does), then the actuator inputs are written and `mj_step2` integrates. The PD loop that tracks its joint targets runs on
every physics step, from the current state:

| `actuation` | Model actuators | What happens each physics step |
|---|---|---|
| `position` | MuJoCo `<position>` | targets sit in `data.ctrl`; `kp (target - q) - kv dq` is evaluated inside `mj_step` |
| `pd_torque` | MuJoCo `<motor>` | `mj_step1`, then `ctrl = kp (target - q) - kd dq` from current state, then `mj_step2` |

`pd_torque` is the in-step PD fix from Living-Proof-Labs/unitree-foxglove-sdk
(`MotorFirmwareBridge`), which replaced a torque computed once per command from stale
sensors. Joints the policy does not drive can be held at fixed targets with their own
gains (`hold`).

## Configuration (`PolicySpec`)

| Field | Meaning |
|---|---|
| `joints`, `actuators` | joint map: model joint and actuator for each policy index, in policy order |
| `default_pose` | joint offsets, policy order; target = `default_pose + action_scale * action` |
| `action_scale` | scalar or one value per joint |
| `decimation` | physics steps per policy step |
| `first_tick` | physics step index of the first policy step (default 0; the Playground adapter uses `decimation - 1`, as its sim2sim loop acts on the 10th control callback) |
| `obs` | ordered `ObsTerm`s: `sensor`, `gravity` (site or `body:<name>`), `base_angvel`, `command`, `joint_pos`, `joint_vel`, `last_action`, `phase`; each with a scalar or per-element `scale` |
| `history` | stacked observation frames, newest last |
| `gait_hz`, `gait_offsets`, `phase_order` | phase clock advanced once per policy step and wrapped to [-pi, pi] |
| `actuation`, `kp`, `kd`, `hold` | see Timing |
| `target_range` | optional per-joint clamp of policy targets, as a position actuator's `ctrlrange` does |

Mismatches fail at construction: wrong actuator type for the actuation mode, an actuator
that does not drive its joint, unknown names, or an observation size that differs from the
ONNX input.

## Adapters

`livingproof_sim_mujoco.adapters` builds a spec from a controller family's own deploy
values, so the numbers stay in the upstream file:

- `rl_gym(deploy_yaml, ...)`: unitree_rl_gym `deploy_real/configs/<robot>.yaml` with the
  `deploy_mujoco.py` observation (47 values for G1) and a 0.8 s gait phase.
- `playground_joystick(...)`: MuJoCo Playground `experimental/sim2sim` joystick policies.

## Backends

The policy file's suffix picks the backend: `.onnx` runs in onnxruntime (CPU); `.pt` or
`.jit` TorchScript runs in torch (CPU, `pip install livingproof-sim-mujoco[torch]`).
TorchScript is run directly rather than converted: unitree_rl_gym's G1 `motion.pt` is an
LSTM whose hidden and cell state live in module buffers updated in place, and an ONNX
export drops that state (outputs differed by up to 14.8 on random inputs). `reset()`
reloads the module so every episode starts from the saved state.

## Example

```python
import mujoco
from livingproof_sim_mujoco.policy import OnnxPolicy

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
policy = OnnxPolicy(model, "policy.onnx", spec)
policy.run(data, seconds=20.0, command=[0.5, 0.0, 0.0])
```
