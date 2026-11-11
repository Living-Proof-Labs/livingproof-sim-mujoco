# SPDX-License-Identifier: Apache-2.0
"""Generic walk helpers shared by robot packages: runner settings, the nominal initial
state, a recorded simulation and the zero-command standing check. Never grades.

    python -m livingproof_sim_mujoco.walk stand <module:adapter> --configuration NAME [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from livingproof_sim_mujoco.scene import place_pose


def runner_settings(root: Path) -> dict:
    doc = json.loads((root / "runner.json").read_text())
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in doc.items()}


def joint_dofs(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel addresses of every hinge joint."""
    hinge = [
        j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
    return model.jnt_qposadr[hinge], model.jnt_dofadr[hinge]


def initial_state(model, data, ctrl, spawn, noise=None, retired_spawn=None, *, start_offset=None):
    """The controller's exact nominal pose at the world's named place. Velocities are zero.

    Certification calls ``initial_state(model, data, ctrl, spawn)``. A declared comparison
    may pass one ``start_offset`` (``base_xy_m`` in metres and/or ``base_yaw_rad``). That
    is a single named change for the run, not a sweep.

    Adapters that still call ``initial_state(model, data, ctrl, seed, noise, spawn)`` keep
    working. The seed and the noise dict are ignored, including ``initial_noise``.
    """
    if isinstance(spawn, str):
        place = spawn
        if retired_spawn is not None:
            raise TypeError(
                "initial_state(model, data, ctrl, spawn, start_offset=None). "
                f"Got an extra positional place {retired_spawn!r}. "
                "Example: initial_state(model, data, ctrl, 'spawn-entrance')"
            )
    elif isinstance(retired_spawn, str):
        place = retired_spawn  # retired (seed, noise, spawn); seed and noise are ignored
    else:
        raise TypeError(
            "initial_state(model, data, ctrl, spawn) expects the world's place name. "
            "The retired (seed, noise, spawn) form is accepted and the seed and noise "
            f"are ignored, but the place was {retired_spawn!r}. "
            "Example: initial_state(model, data, ctrl, 'spawn-entrance')"
        )
    dxy, dyaw = _start_offset(start_offset)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = ctrl.initial_qpos
    position, yaw = place_pose(model, place)
    data.qpos[0:2] = position[0:2] + dxy
    yaw += dyaw
    data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qvel[:] = 0.0
    if ctrl.spec.actuation == "position":  # hold the default pose until the first tick,
        for name, q in zip(ctrl.spec.actuators, ctrl.spec.default_pose):  # as keyframe ctrl
            data.ctrl[model.actuator(name).id] = q
    mujoco.mj_forward(model, data)


def _start_offset(start_offset) -> tuple[np.ndarray, float]:
    """One declared xy/yaw change. Absent means the exact nominal pose."""
    if start_offset is None:
        return np.zeros(2), 0.0
    if not isinstance(start_offset, dict):
        raise TypeError(
            "start_offset must be a dict with base_xy_m and/or base_yaw_rad, "
            f"got {type(start_offset).__name__}. "
            "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
        )
    unknown = set(start_offset) - {"base_xy_m", "base_yaw_rad"}
    if unknown:
        raise ValueError(
            "start_offset only accepts base_xy_m and base_yaw_rad "
            f"(one named comparison change), got {sorted(unknown)}. "
            "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
        )
    dxy = np.zeros(2)
    if "base_xy_m" in start_offset:
        raw = start_offset["base_xy_m"]
        if len(raw) != 2:
            raise ValueError(
                "start_offset base_xy_m must be [x, y] in metres. "
                "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
            )
        dxy = np.array([float(raw[0]), float(raw[1])], dtype=float)
    dyaw = float(start_offset.get("base_yaw_rad", 0.0))
    return dxy, dyaw


def simulate(
    model,
    data,
    step,
    *,
    base_body: str,
    seconds: float,
    p: dict,
    sample_hz: float,
    stop_at_exit=True,
    clip: Path | None = None,
    camera_view=(3.6, -38.0, 200.0),
):
    """Run ``step(data)`` from the current state; return (samples [t, x, y, z], end, clip)."""
    base = model.body(base_body).id
    every_sample = int(round(1 / (sample_hz * model.opt.timestep)))
    renderer = None
    if clip is not None:
        renderer = mujoco.Renderer(model, 288, 512)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = base
        camera.distance, camera.elevation, camera.azimuth = camera_view
        frames, every_frame = [], int(round(1 / (25 * model.opt.timestep)))
    samples, end = [], "time_limit"
    for i in range(int(round(seconds / model.opt.timestep)) + 1):
        if renderer is not None and i % every_frame == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())
        if i % every_sample == 0:
            if not np.all(np.isfinite(data.qpos)):
                end = "non_finite_state"
                break
            x, y, z = (float(v) for v in data.xpos[base])
            samples.append([round(float(data.time), 3), x, y, z])
            if stop_at_exit and x >= p["exit_x"]:
                end = "crossed_exit"
                break
            if z < p["fall_height"]:
                end = "fell"  # outcome decided; nothing later can change it
                break
        step(data)
    clip_name = None
    if renderer is not None:
        import imageio.v2 as imageio

        clip.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(clip, frames, fps=25, codec="libx264", quality=7, macro_block_size=8)
        renderer.close()
        clip_name = clip.name
    return samples, end, clip_name


def stand_summary(name, digest, seconds, samples, end, clip, fall_height) -> dict:
    """Zero-command standing check at the nominal pose; not part of the scenario grade."""
    zs = [s[3] for s in samples]
    last = samples[-30:]  # final 0.6 s
    return {
        "check": "zero-command-stand",
        "configuration": name,
        "configuration_sha256": digest,
        "seconds": seconds,
        "standing": end == "time_limit" and min(zs) >= fall_height,
        "end": end,
        "min_base_height": round(min(zs), 4),
        "final_base_height_mean": round(sum(s[3] for s in last) / len(last), 4),
        "drift_m": round(math.dist(samples[0][1:3], samples[-1][1:3]), 4),
        "max_abs_lateral": round(max(abs(s[2]) for s in samples), 4),
        "clip": clip,
    }


CAMERA = (
    3.6,
    -38.0,
    200.0,
)  # tracking distance, elevation, azimuth; optional adapter.camera overrides


def stand(ref, name: str, out: Path | None = None) -> dict:
    """Zero-command standing check at the nominal pose through a robot adapter
    (``module:NAME`` or the adapter object, as ``livingproof_sim_mujoco.evidence`` uses);
    not part of any grade."""
    from livingproof_sim_mujoco import evidence as ev
    from livingproof_sim_mujoco.pins import fetch

    robot = ev.adapter(ref) if isinstance(ref, str) else ref
    root = Path(robot.root)
    fetch(root)
    entry = ev.locked_configuration(ev.load_lock(root), name, root)
    p = ev.parameters(robot, name)
    rs = robot.runner_settings()
    seconds = rs["stand_seconds"]
    scene = robot.scene(name)
    model = scene.model
    ctrl = robot.controller(name, scene)
    policy = robot.policy(model, ctrl)
    data = mujoco.MjData(model)
    ev.nominal_start(robot, model, data, ctrl, p["spawn"], None)
    command = [0.0, 0.0, 0.0]
    samples, end, clip = simulate(
        model,
        data,
        lambda d: policy.step(d, command),
        base_body=robot.base_body(name),
        seconds=seconds,
        p=p,
        sample_hz=rs["sample_hz"],
        stop_at_exit=False,
        clip=Path(out) / "stand-zero-command.mp4" if out else None,
        camera_view=ev.camera(robot, CAMERA),
    )
    result = stand_summary(name, entry["sha256"], seconds, samples, end, clip, p["fall_height"])
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "stand.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stand", help="zero-command standing check at the nominal pose")
    s.add_argument("adapter", help="robot adapter, module:NAME (e.g. lpl_g1.adapter:ROBOT)")
    s.add_argument("--configuration", required=True)
    s.add_argument("--out", type=Path, default=None, help="default <robot root>/artifacts/<name>")
    a = ap.parse_args(argv)
    from livingproof_sim_mujoco import evidence as ev

    robot = ev.adapter(a.adapter)
    out = a.out or Path(robot.root) / "artifacts" / a.configuration
    print(json.dumps(stand(robot, a.configuration, out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
