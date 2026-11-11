"""walk-hallway sim evidence for the capability verifier. Records observations; never grades.

Generic over robots: a robot package supplies a ``Robot`` adapter (robot-specific data and
code: scene/model build, controller and policy, nominal initial state, profiles) and this
module does the rest. It reads the robot repository's committed ``verification.lock.json``
(release refs and the command speed) and, for one configuration, runs ``trials.count``
identical trials in sequence through ``observations.observe``. The default count is 1.
It writes one observation per trial (``trials/trial-01.json``, ...) and a
``livingproof-hallway-evidence/2.0.0`` manifest. Before the first trial it writes
``run-manifest.json`` in the ``livingproof-run-manifest/0.1`` shape.

    python -m livingproof_sim_mujoco.evidence <module:adapter> <configuration> <out> [--trials N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
from importlib import import_module, metadata
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Protocol

import mujoco

from livingproof_sim_mujoco import scenario as sc
from livingproof_sim_mujoco.cpu import host_cpus, one_torch_thread
from livingproof_sim_mujoco.pins import fetch, sha256

FORMAT = "livingproof-hallway-evidence/2.0.0"
RUN_MANIFEST_FORMAT = "livingproof-run-manifest/0.1"
PENDING = "PENDING_RELEASE"
# Evidence exits: 0 recorded (an unmet precondition is still recorded, with no trials);
# 1 a lock, plan, or manifest error (SystemExit text); 2 argparse. The physics command
# uses 2, 124, 130, and 143. The hallway fixture tool uses 0 for PASS, 1 for a non-PASS
# result, 2 for a fixture error, and 3 for UNSUPPORTED.
# 4 is only a test-system reproducibility finding, not a robot grade and not those codes.
EXIT_REPRODUCIBILITY = 4
LOCK = "verification.lock.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SCENARIO = re.compile(r"^[a-z0-9-]+@[0-9]+\.[0-9]+\.[0-9]+$")


class Robot(Protocol):
    """What a robot package provides. Everything generic lives here instead."""

    id: str  # e.g. "unitree-g1"
    root: Path  # robot repository root (lock, robot.json, pins.json, configurations/)
    width_decimals: int  # fits_width detail formatting, kept per robot for stable manifests

    def base_body(self, name: str) -> str: ...
    def scene(self, name: str) -> Any: ...  # .model (MjModel), .walls (list[str])
    def controller(self, name: str, scene: Any) -> Any: ...
    def policy(self, model: mujoco.MjModel, ctrl: Any) -> Any: ...  # .step(data, command)
    def initial_state(self, model, data, ctrl, spawn: str, *, start_offset=None) -> None: ...
    def runner_settings(self) -> dict: ...
    def robot(self) -> dict: ...
    def configuration(self, name: str) -> dict: ...  # raises Unsupported(precondition, detail)
    def world(self) -> dict: ...
    def environment(self, name: str) -> dict: ...  # extra manifest environment fields

    # Optional: ``camera = (distance, elevation, azimuth)`` for the tracking camera of the
    # stand clip and media videos; read through ``camera()``. Never affects evidence.


def camera(robot, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """The adapter's optional tracking camera (distance, elevation, azimuth), else default."""
    view = getattr(robot, "camera", None)
    if view is None:
        return tuple(default)
    view = tuple(float(v) for v in view)
    if len(view) != 3 or view[0] <= 0:
        raise ValueError(f"camera must be (distance > 0, elevation, azimuth), got {view!r}")
    return view


def adapter(ref: str) -> Robot:
    """``package.module:NAME`` -> the adapter object."""
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


def trial_count(plan: dict, override: int | None) -> int:
    """``trials.count`` for this run. The default is 1. ``plan["seeds"]`` is ignored.

    An integer ``plan["trials"]`` is the retired seed count, not the v2 count. A plan
    that already stores ``{"trials": {"count": N}}`` is used when ``--trials`` is omitted.
    """
    if override is not None:
        count = override
    else:
        raw = plan.get("trials", 1)
        count = raw.get("count", 1) if isinstance(raw, dict) else 1
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SystemExit(
            f"trials.count must be an integer >= 1, got {count!r}. "
            "Example: python -m livingproof_sim_mujoco.evidence "
            "<adapter> <configuration> <out> --trials 3"
        )
    return count


def declared_offset(plan: dict) -> dict | None:
    """One named start offset from the plan, or none for the exact nominal pose.

    This is a hook for a declared comparison change. It does not sweep offsets.
    """
    if "start_offset" not in plan or plan["start_offset"] is None:
        return None
    offset = plan["start_offset"]
    if not isinstance(offset, dict):
        raise SystemExit(
            "plan start_offset must be an object with base_xy_m and/or base_yaw_rad, "
            f"got {offset!r}. Example: "
            '\'"start_offset": {"base_xy_m": [0.0, 0.05]}\''
        )
    return offset


def nominal_start(robot, model, data, ctrl, spawn: str, start_offset: dict | None) -> None:
    """Place the trial at the nominal pose. Does not read runner ``initial_noise``.

    Adapters published before the trial model still take ``(seed, noise, spawn)`` and
    forward those to ``walk.initial_state``, which ignores both. A declared
    ``start_offset`` needs the current ``(spawn, start_offset=...)`` signature.
    """
    params = inspect.signature(robot.initial_state).parameters
    if "noise" in params or "seed" in params:
        if start_offset:
            raise SystemExit(
                "start_offset is one declared comparison change and needs "
                "initial_state(model, data, ctrl, spawn, start_offset=...). "
                f"{getattr(robot, 'id', robot)} still takes the retired seed and noise "
                "arguments, so the offset was not applied. "
                "Example: initial_state(model, data, ctrl, 'spawn-entrance', "
                "start_offset={'base_xy_m': [0.0, 0.05]})"
            )
        robot.initial_state(model, data, ctrl, 0, None, spawn)
        return
    if start_offset is None:
        robot.initial_state(model, data, ctrl, spawn)
    else:
        robot.initial_state(model, data, ctrl, spawn, start_offset=start_offset)


def run_trial(ref: str, name: str, plan: dict, p: dict, on_step=None, *, start_offset=None) -> dict:
    """One trial at the nominal pose through ``observations.observe``.

    ``on_step`` (for example ``livingproof_sim_mujoco.media.Recorder``) only reads
    ``data``, so it cannot change the observation. Every trial in the run uses this
    same call.
    """
    one_torch_thread()
    from livingproof_sim_mujoco.observations import observe

    robot = adapter(ref)
    rs = robot.runner_settings()
    scene = robot.scene(name)
    model = scene.model
    ctrl = robot.controller(name, scene)
    policy = robot.policy(model, ctrl)
    data = mujoco.MjData(model)
    nominal_start(robot, model, data, ctrl, p["spawn"], start_offset)
    command = [plan["command_speed"], 0.0, 0.0]
    hook = on_step(model, robot.base_body(name)) if on_step is not None else None
    obs = observe(
        model,
        data,
        lambda d: policy.step(d, command),
        base_body=robot.base_body(name),
        walls=scene.walls,
        exit_x=p["exit_x"],
        seconds=p["time_limit"],
        sample_hz=rs["sample_hz"],
        on_step=hook,
    )
    if hook is not None:
        hook.close()
    return obs


def observation_bytes(obs: dict) -> bytes:
    """The exact bytes written to ``trials/trial-NN.json`` (their sha256 is the trial's)."""
    return (json.dumps(obs, separators=(",", ":")) + "\n").encode()


def trajectory_sha256(obs: dict) -> str:
    """sha256 of the recorded trace. Identical trials share this hash."""
    body = json.dumps(obs.get("trace"), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def _write_trial(out: Path, index: int, obs: dict, wall_clock_s: float) -> dict:
    path = out / "trials" / f"trial-{index:02d}.json"
    path.write_bytes(observation_bytes(obs))
    return {
        "index": index,
        "path": f"trials/{path.name}",
        "sha256": sha256(path),
        "trajectory_sha256": trajectory_sha256(obs),
        "outcome": obs.get("end"),
        "wall_clock_s": round(wall_clock_s, 3),
    }


def reproducibility(results: list[dict]) -> dict:
    """Compare identical trials. A disagreement is a test-system finding, not a robot grade.

    Count 1 is not a comparison. Outcome here is the observation's recorded end reason
    (for example ``crossed_exit``). The capability verifier grades later.
    """
    if len(results) < 2:
        return {"compared": False, "identical": None, "finding": None}
    same = (
        len({t["sha256"] for t in results}) == 1
        and len({t["trajectory_sha256"] for t in results}) == 1
        and len({t["outcome"] for t in results}) == 1
    )
    if same:
        return {"compared": True, "identical": True, "finding": None}
    return {
        "compared": True,
        "identical": False,
        "finding": {
            "kind": "test-system",
            "name": "reproducibility",
            "detail": (
                "Identical trials disagreed on trajectory hash or outcome. "
                "This is a test-system reproducibility finding, not a robot FAIL."
            ),
            "trajectory_sha256": [t["trajectory_sha256"] for t in results],
            "outcomes": [t["outcome"] for t in results],
        },
    }


def unmet_required(preconditions: list) -> dict | None:
    """The first required precondition that is not met, else None.

    An unmet one means the robot cannot attempt the scenario: no simulator is
    started and no trial is recorded (the verifier grades it UNSUPPORTED).
    """
    for c in preconditions:
        if c.get("required", True) and not c["met"]:
            return c
    return None


def recording(robot: Robot, lock: dict, name: str) -> tuple[dict, dict]:
    """(plan, scenario parameters) exactly as every evidence trial is recorded."""
    plan = lock["plan"]
    p = parameters(robot, name)
    # Recording length only; the verifier applies the scenario's time limit.
    p["time_limit"] = 2 * p["route_length"] / plan["command_speed"] + 5
    return plan, p


def executed_plan(plan: dict, count: int) -> dict:
    """The plan this run used: ``trials.count``. Seeds and ``required_passes`` are omitted."""
    used = {
        k: v
        for k, v in plan.items()
        if k not in {"seeds", "trials", "start_offset", "required_passes"}
    }
    used["trials"] = {"count": count}
    return used


def _require(label: str, value: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise SystemExit(
            f"run manifest {label} must match {pattern.pattern}, got {value!r}. "
            "Example: a 40-character commit or a sha256 hex digest, as "
            "specs/robots/run-manifest.schema.json requires."
        )
    return value


def world_sha256(root: Path, lock: dict) -> str:
    """sha256 of the world file: the lock, else the robot's pins.json."""
    world = lock["world"]
    declared = world.get("sha256")
    if isinstance(declared, str) and _SHA256.fullmatch(declared):
        return declared
    ident = f"{world['id']}/{world['version']}"
    pins_path = Path(root) / "pins.json"
    if pins_path.is_file():
        pins = json.loads(pins_path.read_text())
        for group in pins.get("groups", {}).values():
            for path, entry in group.get("files", {}).items():
                digest = entry.get("sha256") if isinstance(entry, dict) else None
                if ident in path and path.endswith("world.json") and isinstance(digest, str):
                    return _require(f"world {ident} sha256", digest, _SHA256)
    raise SystemExit(
        f"world {ident} has no sha256 in {LOCK} or pins.json. "
        "Add world.sha256 or pin the world file. "
        'Example: "world": {"id": "hallway-open", "version": "0.1.0", "sha256": "<64 hex>"}'
    )


def robot_repo(root: Path, lock: dict) -> str:
    declared = lock.get("robot", {}).get("repo") if isinstance(lock.get("robot"), dict) else None
    if isinstance(declared, str) and declared:
        return declared
    remote = subprocess.run(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    url = remote.stdout.strip()
    if remote.returncode == 0 and url:
        return url
    raise SystemExit(
        "run manifest robot.repo needs the robot repository URL. "
        "Set lock robot.repo or git remote origin. "
        'Example: "robot": {"repo": "https://github.com/Living-Proof-Labs/livingproof-unitree-g1"}'
    )


def _valid_commit(value: object) -> str | None:
    return value if isinstance(value, str) and _COMMIT.fullmatch(value) else None


def _git_text(cwd: Path, *args: str) -> str | None:
    try:
        found = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if found.returncode != 0:
        return None
    return found.stdout.strip()


def _source_tree_root() -> Path | None:
    """Git toplevel when this module is the package source, else None.

    An installed copy lives under ``site-packages``. Git would walk up from there
    to whatever repository encloses the virtualenv; that commit is not this package.
    Accept the toplevel only when it contains this package's source (the module
    directory is ``src/livingproof_sim_mujoco`` or ``livingproof_sim_mujoco``) and,
    when ``pyproject.toml`` is present, that file names livingproof-sim-mujoco.
    """
    package_dir = Path(__file__).resolve().parent
    top = _git_text(package_dir, "rev-parse", "--show-toplevel")
    if not top:
        return None
    root = Path(top).resolve()
    if package_dir not in {
        (root / "src" / "livingproof_sim_mujoco").resolve(),
        (root / "livingproof_sim_mujoco").resolve(),
    }:
        return None
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and "livingproof-sim-mujoco" not in pyproject.read_text(
        encoding="utf-8", errors="replace"
    ):
        return None
    return root


def _build_provenance_commit() -> str | None:
    """``identity.commit`` written into the wheel by the hatch build hook."""
    path = Path(__file__).resolve().parent / "_build_provenance.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        commit = saved.get("identity", {}).get("commit")
    except (OSError, ValueError, AttributeError):
        return None
    return _valid_commit(commit)


def _direct_url_commit() -> str | None:
    """PEP 610 ``direct_url.json`` ``vcs_info.commit_id`` of the installed distribution."""
    try:
        text = metadata.distribution("livingproof-sim-mujoco").read_text("direct_url.json")
    except Exception:
        return None
    if not text:
        return None
    try:
        commit = json.loads(text).get("vcs_info", {}).get("commit_id")
    except (ValueError, AttributeError):
        return None
    return _valid_commit(commit)


def adapter_commit() -> tuple[str | None, str]:
    """(commit, provenance) of livingproof-sim-mujoco for the run manifest.

    A source checkout uses its own git HEAD (``git``), so a leftover build snapshot
    cannot hide the checkout and an enclosing repository is never consulted.
    An installed copy reads ``identity.commit`` from the packaged
    ``_build_provenance.json`` (``package``), then PEP 610 ``direct_url.json``
    ``vcs_info.commit_id`` (``direct-url``). When none of those is a 40-character
    commit, the commit is null and provenance is ``unknown``; evidence still records
    the run. The Debian package and the runtime image import this function. The
    Debian payload drops ``direct_url.json``, so that install uses the build snapshot.
    """
    root = _source_tree_root()
    if root is not None:
        commit = _valid_commit(_git_text(root, "rev-parse", "HEAD"))
        if commit:
            return commit, "git"
    commit = _build_provenance_commit()
    if commit:
        return commit, "package"
    commit = _direct_url_commit()
    if commit:
        return commit, "direct-url"
    return None, "unknown"


def build_run_manifest(
    robot: Robot,
    lock: dict,
    name: str,
    entry: dict,
    p: dict,
    count: int,
    run_id: str,
    profile_sha256: str,
    robot_commit: str,
    adapter_source: tuple[str | None, str],
) -> dict:
    """``livingproof-run-manifest/0.1``, written before the first trial."""
    adapter_commit_value, adapter_provenance = adapter_source
    specs = lock.get("releases", {}).get("specs")
    if not isinstance(specs, dict):
        raise SystemExit(
            f"{LOCK} releases.specs needs a tag and commit. "
            'Example: "specs": {"tag": "v2.0.2", "commit": "<40 hex>"}'
        )
    scenario = lock.get("scenario", sc.SCENARIO)
    parameters_out = {}
    for key, value in p.items():
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            parameters_out[key] = value
        else:
            raise SystemExit(
                f"parameter {key} must be a number, string, or boolean for the run manifest, "
                f"got {type(value).__name__}."
            )
    return {
        "format": RUN_MANIFEST_FORMAT,
        "run_id": run_id,
        "specification": {
            "tag": specs["tag"],
            "commit": _require("specification.commit", specs.get("commit", ""), _COMMIT),
        },
        "scenario": _require("scenario", scenario, _SCENARIO),
        "world": {
            "id": lock["world"]["id"],
            "version": lock["world"]["version"],
            "sha256": world_sha256(robot.root, lock),
        },
        "simulator": {
            "engine": "mujoco",
            "version": mujoco.__version__,
            "adapter": "livingproof-sim-mujoco",
            # Null when the commit is unknown. A known commit is still 40 hex characters.
            # ``provenance`` says which source supplied it (git, package, direct-url, unknown).
            "adapter_commit": (
                _require("simulator.adapter_commit", adapter_commit_value, _COMMIT)
                if adapter_commit_value
                else None
            ),
            "provenance": adapter_provenance,
        },
        "robot": {
            "id": robot.id,
            "repo": robot_repo(robot.root, lock),
            "commit": _require("robot.commit", robot_commit, _COMMIT),
            "profile_sha256": _require("robot.profile_sha256", profile_sha256, _SHA256),
        },
        "configuration": {
            "name": name,
            "version": _require("configuration.version", entry["version"], _SEMVER),
            "sha256": _require("configuration.sha256", entry["sha256"], _SHA256),
        },
        "parameters": parameters_out,
        "trials": {"count": count},
    }


def evidence(
    ref: str, name: str, out: Path, *, trials: int | None = None, allow_pending=False
) -> dict:
    robot = adapter(ref)
    root = Path(robot.root)
    lock = load_lock(root)
    waiting = pending(lock)
    if waiting and not allow_pending:
        raise SystemExit(f"{LOCK} has PENDING_RELEASE refs ({', '.join(waiting)}); run task repin")
    entry = locked_configuration(lock, name, root)
    fetch(root)
    world = robot.world()
    plan, p = recording(robot, lock, name)
    count = trial_count(plan, trials)
    offset = declared_offset(plan)
    out = Path(out)
    (out / "trials").mkdir(parents=True, exist_ok=False)
    created = dt.datetime.now(dt.timezone.utc)
    run_id = f"walk-hallway-{robot.id}-{name}-{created:%Y%m%dT%H%M%SZ}"
    profile = robot.robot()
    checked = sc.preconditions(
        profile, world, name, robot.configuration, width_decimals=robot.width_decimals
    )
    unmet = unmet_required(checked)
    profile_sha = sha256(root / "robot.json")
    robot_commit = sc.git_head(root)[0]
    adapter_source = adapter_commit()
    results = []
    started = time.perf_counter()
    if unmet is None:
        document = build_run_manifest(
            robot,
            lock,
            name,
            entry,
            p,
            count,
            run_id,
            profile_sha,
            robot_commit,
            adapter_source,
        )
        # Before the first trial, as the run-manifest schema requires.
        (out / "run-manifest.json").write_text(json.dumps(document, indent=2) + "\n")
        for index in range(1, count + 1):
            t0 = time.perf_counter()
            obs = run_trial(ref, name, plan, p, start_offset=offset)
            results.append(_write_trial(out, index, obs, time.perf_counter() - t0))
    wall_clock_s = round(time.perf_counter() - started, 3)
    report = reproducibility(results)
    manifest = {
        "format": FORMAT,
        "run_id": run_id,
        "created": created.isoformat(timespec="seconds"),
        "lock": {"path": LOCK, "sha256": sha256(root / LOCK), "content": lock},
        "pending_release": waiting,
        "plan": executed_plan(plan, count),
        "start": {"spawn": p["spawn"], "pose": "nominal", "offset": offset},
        "preconditions": checked,
        "robot": {
            "id": robot.id,
            "profile_sha256": profile_sha,
            "profile": json.loads((root / "robot.json").read_text()),
        },
        "configuration": {
            "name": name,
            "version": entry["version"],
            "sha256": entry["sha256"],
            "profile": json.loads((root / entry["path"]).read_text()),
        },
        "world": {
            "id": lock["world"]["id"],
            "version": lock["world"]["version"],
            "clear_width": world["dimensions"]["clear_width"],
            "route_length": p["route_length"],
            "exit_x": p["exit_x"],
        },
        "environment": {
            "simulator": {
                "id": "mujoco",
                "version": mujoco.__version__,
                "adapter": "livingproof-sim-mujoco",
                "adapter_commit": adapter_source[0],
                "provenance": adapter_source[1],
            },
            "runner": {
                "version": metadata.version("livingproof-sim-mujoco"),
                "tag": lock["releases"]["runner"]["tag"],
                "commit": lock["releases"]["runner"]["commit"],
            },
            **(
                {}
                if unmet is not None
                else {
                    "base_body": robot.base_body(name),
                    "walls": robot.scene(name).walls,
                    **robot.environment(name),
                }
            ),
            "source_commit": robot_commit,
        },
        "host": {"wall_clock_s": wall_clock_s, "cpu": host_cpus()},
        "trials": {"count": count, "results": results, "reproducibility": report},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", help="robot adapter, module:NAME (e.g. lpl_g1.adapter:ROBOT)")
    ap.add_argument("configuration")
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--trials",
        type=int,
        default=None,
        help="identical trials to run (trials.count). Default 1, unless the lock plan "
        "already sets trials.count. Example: --trials 3",
    )
    ap.add_argument("--allow-pending", action="store_true", help="local runs before releases exist")
    ap.epilog = (
        "Exit 0: observations recorded. Exit 1: the lock, plan, or manifest is unusable. "
        "Exit 2: the command line is unusable. "
        f"Exit {EXIT_REPRODUCIBILITY}: trials.count > 1 and the trials disagree. "
        "That exit is a test-system reproducibility finding recorded in the manifest, "
        "not a robot grade. Example: python -m livingproof_sim_mujoco.evidence "
        "lpl_g1.adapter:ROBOT stock out --trials 3"
    )
    a = ap.parse_args(argv)
    m = evidence(a.adapter, a.configuration, a.out, trials=a.trials, allow_pending=a.allow_pending)
    unmet = unmet_required(m["preconditions"])
    if unmet is not None:
        print(f"UNSUPPORTED: precondition {unmet['name']} unmet: {unmet['detail']}; no trials run")
        return 0
    finding = m["trials"]["reproducibility"]["finding"]
    print(
        f"{m['trials']['count']} observations in {m['host']['wall_clock_s']}s "
        f"-> {a.out / 'manifest.json'}"
    )
    if finding is not None:
        print(finding["detail"], file=sys.stderr)
        return EXIT_REPRODUCIBILITY
    return 0


if __name__ == "__main__":
    sys.exit(main())
