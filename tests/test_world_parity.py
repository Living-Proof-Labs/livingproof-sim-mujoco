# SPDX-License-Identifier: Apache-2.0
"""One hallway rollout per graded configuration: v2.1.1 + robot v1 versus this sim + robot v2.

Numbers are compared on the same machine. A difference is a failure, not a tolerance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/livingproof-world-parity")

# Last v1 release of each robot, and the configuration livingproof-verification grades.
CASES = [
    ("livingproof-unitree-g1", "v1.0.4", "lpl_g1.adapter:ROBOT", "playground"),
    ("livingproof-unitree-g1", "v1.0.4", "lpl_g1.adapter:ROBOT", "stock"),
    ("livingproof-unitree-go1", "v1.0.3", "lpl_go1.adapter:ROBOT", "playground"),
    ("livingproof-booster-t1", "v1.0.3", "lpl_t1.adapter:ROBOT", "stock"),
]


def _run(cmd, **kwargs):
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, **kwargs)
    if result.returncode != 0:
        raise AssertionError(
            f"{cmd} exited {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
        )
    return result


def _sim_v211() -> Path:
    dest = CACHE / "sim-v2.1.1"
    if not (dest / ".git").is_dir():
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=True)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v2.1.1",
                "https://github.com/Living-Proof-Labs/livingproof-sim-mujoco.git",
                str(dest),
            ]
        )
    return dest


def _robot_v1(name: str, tag: str) -> Path:
    dest = CACHE / f"{name}-{tag}"
    if not (dest / "pyproject.toml").is_file():
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=True)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                tag,
                f"https://github.com/Living-Proof-Labs/{name}.git",
                str(dest),
            ]
        )
    return dest


def _old_trial(sim: Path, robot_dir: Path, adapter: str, configuration: str, out: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(sim / "src"), str(robot_dir / "src")])
    env["MUJOCO_GL"] = "osmesa"
    out.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "livingproof_sim_mujoco.evidence",
            adapter,
            configuration,
            str(out),
            "--trials",
            "1",
        ],
        env=env,
        cwd=str(robot_dir),
    )
    return json.loads((out / "trials" / "trial-01.json").read_text())


def _new_lock(robot, configuration: str, directory: Path) -> Path:
    from livingproof_sim_mujoco.pins import sha256

    offered = {c["name"]: c for c in robot.robot()["configurations"]}
    entry = offered[configuration]
    path = Path(robot.root) / entry["path"]
    version = entry.get("version")
    if not isinstance(version, str):
        version = "1.0.0"
    lock = {
        "scenario": "walk-hallway@0.1.0",
        "world": {"id": "hallway-open", "version": "0.1.0"},
        "robot": {
            "repo": {
                "unitree-g1": "https://github.com/Living-Proof-Labs/livingproof-unitree-g1",
                "unitree-go1": "https://github.com/Living-Proof-Labs/livingproof-unitree-go1",
                "booster-t1": "https://github.com/Living-Proof-Labs/livingproof-booster-t1",
            }[robot.id]
        },
        "plan": {"trials": {"count": 1}, "command_speed": 0.5},
        "configurations": [
            {
                "name": configuration,
                "version": version,
                "path": entry["path"],
                "sha256": sha256(path),
            }
        ],
        "releases": {
            "runner": {
                "tag": "v2.1.1",
                "commit": "6aa1017adf72ac2886ed4f407e0ac574531744d5",
            },
            "specs": {
                "tag": "v2.2.0",
                "commit": "23bec516bda29be7b26555286cbf2e8217ac5c81",
            },
        },
    }
    dest = directory / "lock.json"
    dest.write_text(json.dumps(lock))
    return dest


def _new_trial(adapter: str, configuration: str, checkout: Path, out: Path) -> dict:
    from livingproof_sim_mujoco import evidence

    robot = evidence.adapter(adapter)
    # The installed wheel does not ship pins or models. The v2.0.0 tree does.
    robot.root = checkout
    lock = _new_lock(robot, configuration, out.parent)
    evidence.evidence(adapter, configuration, out, lock=lock, trials=1)
    return json.loads((out / "trials" / "trial-01.json").read_text())


def _brief(obs: dict) -> dict:
    samples = obs.get("trace", {}).get("samples") or []
    return {
        "end": obs.get("end"),
        "exit_time": obs.get("exit_time"),
        "samples": len(samples),
        "start": samples[0][1:4] if samples else None,
        "contacts": obs.get("wall_contacts"),
        "max_abs_lateral": obs.get("max_abs_lateral"),
        "min_base_height": obs.get("min_base_height"),
    }


@pytest.mark.parametrize(("repo", "tag", "adapter", "configuration"), CASES)
def test_hallway_rollout_matches_v211_with_the_last_v1_robot(
    repo, tag, adapter, configuration, tmp_path
):
    old = _old_trial(_sim_v211(), _robot_v1(repo, tag), adapter, configuration, tmp_path / "old")
    new = _new_trial(adapter, configuration, _robot_v1(repo, "v2.0.0"), tmp_path / "new")
    if old != new:
        raise AssertionError(json.dumps({"old": _brief(old), "new": _brief(new)}, indent=2))
