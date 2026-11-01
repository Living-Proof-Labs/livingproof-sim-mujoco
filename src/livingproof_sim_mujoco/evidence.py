"""walk-hallway sim evidence for the capability verifier. Records observations; never grades.

Generic over robots: a robot package supplies a ``Robot`` adapter (robot-specific data and
code: scene/model build, controller and policy, seeded initial state, profiles) and this
module does the rest. It reads the robot repository's committed ``verification.lock.json``
(trial plan and release refs) and, for one configuration, runs every planned seed through
``observations.observe``. It writes one observation per trial and a
``livingproof-hallway-evidence/1.0.0`` manifest that echoes the lock.

    python -m livingproof_sim_mujoco.evidence <module:adapter> <configuration> <out>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol

import mujoco

from livingproof_sim_mujoco import scenario as sc
from livingproof_sim_mujoco.pins import fetch, sha256

FORMAT = "livingproof-hallway-evidence/1.0.0"
PENDING = "PENDING_RELEASE"
LOCK = "verification.lock.json"


class Robot(Protocol):
    """What a robot package provides. Everything generic lives here instead."""

    id: str  # e.g. "unitree-g1"
    root: Path  # robot repository root (lock, robot.json, pins.json, configurations/)
    width_decimals: int  # fits_width detail formatting, kept per robot for stable manifests

    def base_body(self, name: str) -> str: ...
    def scene(self, name: str) -> Any: ...  # .model (MjModel), .walls (list[str])
    def controller(self, name: str, scene: Any) -> Any: ...
    def policy(self, model: mujoco.MjModel, ctrl: Any) -> Any: ...  # .step(data, command)
    def initial_state(self, model, data, ctrl, seed: int, noise: dict, spawn: str) -> None: ...
    def runner_settings(self) -> dict: ...
    def robot(self) -> dict: ...
    def configuration(self, name: str) -> dict: ...  # raises Unsupported(precondition, detail)
    def world(self) -> dict: ...
    def environment(self, name: str) -> dict: ...  # extra manifest environment fields


def adapter(ref: str) -> Robot:
    """``package.module:NAME`` -> the adapter object (importable in worker processes)."""
    module, _, attr = ref.partition(":")
    return getattr(import_module(module), attr or "ROBOT")


def load_lock(root: Path) -> dict:
    return json.loads((Path(root) / LOCK).read_text())


def pending(lock: dict) -> list[str]:
    """Release entries that still say PENDING_RELEASE."""
    return sorted(
        name
        for name, ref in lock["releases"].items()
        if PENDING in (ref.get("tag"), ref.get("commit"))
    )


def locked_configuration(lock: dict, name: str, root: Path) -> dict:
    entry = next((c for c in lock["configurations"] if c["name"] == name), None)
    if entry is None:
        raise SystemExit(f"{name} is not in {LOCK}")
    if sha256(Path(root) / entry["path"]) != entry["sha256"]:
        raise SystemExit(f"{entry['path']} differs from {LOCK}; update the lock in the same commit")
    return entry


def parameters(robot: Robot, name: str) -> dict:
    return sc.parameters(robot.robot(), robot.configuration(name), robot.world())


def _trial(args):
    from livingproof_sim_mujoco.observations import observe

    ref, name, seed, plan, p, out = args
    robot = adapter(ref)
    rs = robot.runner_settings()
    scene = robot.scene(name)
    model = scene.model
    ctrl = robot.controller(name, scene)
    policy = robot.policy(model, ctrl)
    data = mujoco.MjData(model)
    robot.initial_state(model, data, ctrl, seed, rs["initial_noise"], p["spawn"])
    command = [plan["command_speed"], 0.0, 0.0]
    obs = observe(
        model,
        data,
        lambda d: policy.step(d, command),
        base_body=robot.base_body(name),
        walls=scene.walls,
        exit_x=p["exit_x"],
        seconds=p["time_limit"],
        sample_hz=rs["sample_hz"],
    )
    path = Path(out) / "trials" / f"seed-{seed:02d}.json"
    path.write_text(json.dumps(obs, separators=(",", ":")) + "\n")
    return {"seed": seed, "path": f"trials/{path.name}", "sha256": sha256(path)}


def evidence(
    ref: str, name: str, out: Path, *, jobs: int | None = None, allow_pending=False
) -> dict:
    robot = adapter(ref)
    root = Path(robot.root)
    lock = load_lock(root)
    waiting = pending(lock)
    if waiting and not allow_pending:
        raise SystemExit(f"{LOCK} has PENDING_RELEASE refs ({', '.join(waiting)}); run task repin")
    entry = locked_configuration(lock, name, root)
    fetch(root)
    plan = lock["plan"]
    world = robot.world()
    p = parameters(robot, name)
    # Recording length only; the verifier applies the scenario's time limit.
    p["time_limit"] = 2 * p["route_length"] / plan["command_speed"] + 5
    out = Path(out)
    (out / "trials").mkdir(parents=True, exist_ok=False)
    created = dt.datetime.now(dt.timezone.utc)
    work = [(ref, name, s, plan, p, out) for s in plan["seeds"]]
    with ProcessPoolExecutor(jobs or os.cpu_count()) as pool:
        trials = list(pool.map(_trial, work))
    conf_path = root / entry["path"]
    profile = robot.robot()
    manifest = {
        "format": FORMAT,
        "run_id": f"walk-hallway-{robot.id}-{name}-{created:%Y%m%dT%H%M%SZ}",
        "created": created.isoformat(timespec="seconds"),
        "lock": {"path": LOCK, "sha256": sha256(root / LOCK), "content": lock},
        "pending_release": waiting,
        "plan": plan,
        "preconditions": sc.preconditions(
            profile, world, name, robot.configuration, width_decimals=robot.width_decimals
        ),
        "robot": {
            "id": robot.id,
            "profile_sha256": sha256(root / "robot.json"),
            "profile": json.loads((root / "robot.json").read_text()),
        },
        "configuration": {
            "name": name,
            "version": entry["version"],
            "sha256": entry["sha256"],
            "profile": json.loads(conf_path.read_text()),
        },
        "world": {
            "id": lock["world"]["id"],
            "version": lock["world"]["version"],
            "clear_width": world["dimensions"]["clear_width"],
            "route_length": p["route_length"],
            "exit_x": p["exit_x"],
        },
        "environment": {
            "simulator": {"id": "mujoco", "version": mujoco.__version__},
            "runner": {
                "version": metadata.version("livingproof-sim-mujoco"),
                "tag": lock["releases"]["runner"]["tag"],
                "commit": lock["releases"]["runner"]["commit"],
            },
            "base_body": robot.base_body(name),
            "walls": robot.scene(name).walls,
            **robot.environment(name),
            "source_commit": sc.git_head(root)[0],
        },
        "trials": sorted(trials, key=lambda t: t["seed"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", help="robot adapter, module:NAME (e.g. lpl_g1.adapter:ROBOT)")
    ap.add_argument("configuration")
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--allow-pending", action="store_true", help="local runs before releases exist")
    a = ap.parse_args(argv)
    m = evidence(a.adapter, a.configuration, a.out, jobs=a.jobs, allow_pending=a.allow_pending)
    print(f"{len(m['trials'])} observations -> {a.out / 'manifest.json'}")
    return 0  # evidence only: no grade, so nothing to signal in the exit code


if __name__ == "__main__":
    sys.exit(main())
