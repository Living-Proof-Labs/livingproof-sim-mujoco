"""Sim evidence for the capability verifier. Records observations; never grades.

Generic over robots: a robot package supplies a ``Robot`` adapter (the robot model,
controller and policy, nominal initial pose, profiles) and this module does the rest.
The caller passes the verification lock (``--lock``). That lock selects the scenario
and the world. Wall geometry and the named start place come from the world's release
vendored in this package, not from the robot package. For one configuration it runs
``trials.count`` identical trials in sequence through ``observations.observe``. The
default count is 1. It writes one observation per trial (``trials/trial-01.json``, ...)
and a ``livingproof-hallway-evidence/2.0.0`` manifest. Before the first trial it writes
``run-manifest.json`` in the ``livingproof-run-manifest/0.1`` shape.

    python -m livingproof_sim_mujoco.evidence <module:adapter> <configuration> <out> \\
        --lock <verification-lock.json> [--trials N]
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

from livingproof_scenario_parameters import ScenarioParameterError

from livingproof_sim_mujoco import compose, scenario as sc, sources
from livingproof_sim_mujoco.compat import hallway as frozen
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
    def scene(self, name: str) -> Any: ...  # robot model only; the sim adds the world
    def controller(self, name: str, scene: Any) -> Any: ...
    def policy(self, model: mujoco.MjModel, ctrl: Any) -> Any: ...  # .step(data, command)
    def initial_state(self, model, data, ctrl, spawn: str, *, start_offset=None) -> None: ...
    def runner_settings(self) -> dict: ...
    def robot(self) -> dict: ...
    def configuration(self, name: str) -> dict: ...  # raises Unsupported(precondition, detail)
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


def read_lock(path: Path) -> dict:
    """The caller's verification lock. The path is whatever the caller passed."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"INCOMPATIBLE: verification lock {path} is not a file. "
            "Pass it with --lock. Example: python -m livingproof_sim_mujoco.evidence "
            "<adapter> <configuration> <out> --lock path/to/lock.json"
        )
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"INCOMPATIBLE: {path} is not JSON ({exc}).") from exc
    if not isinstance(document, dict):
        raise SystemExit(
            f"INCOMPATIBLE: {path} must be a JSON object. "
            'Example: {"scenario": "<scenario>@<version>", "world": {"id": "<world>", "version": "0.1.0"}}'
        )
    return document


_LOCK_EXAMPLE = {
    "world": '"world": {"id": "<world>", "version": "0.1.0"}',
    "releases": (
        '"releases": {"specs": {"tag": "v2.2.0", "commit": "<40 hex>"}, '
        '"runner": {"tag": "v2.1.1", "commit": "<40 hex>"}}'
    ),
    "configurations": (
        '"configurations": [{"name": "<name>", "version": "1.0.0", '
        '"path": "<file>", "sha256": "<64 hex>"}]'
    ),
    "plan": '"plan": {"trials": {"count": 1}}',
}


def lock_field(lock: dict, field: str, kind: type):
    """The lock's ``field``, or an exit-1 error that names it.

    A missing ``releases``, ``configurations``, ``plan`` or ``world`` is a caller
    mistake. It is reported as that field, not as a ``KeyError``.
    """
    if not isinstance(lock, dict) or field not in lock or lock[field] is None:
        raise SystemExit(
            f"INCOMPATIBLE: the verification lock is missing {field}. "
            f"The caller lock needs {field}. Example: {_LOCK_EXAMPLE[field]}"
        )
    if not isinstance(lock[field], kind):
        expected = "a JSON object" if kind is dict else "a JSON array"
        raise SystemExit(
            f"INCOMPATIBLE: the verification lock {field} must be {expected}, "
            f"got {type(lock[field]).__name__}. Example: {_LOCK_EXAMPLE[field]}"
        )
    return lock[field]


def require_lock(lock: dict) -> None:
    """Refuse a caller lock that lacks the fields evidence and stand read."""
    lock_field(lock, "world", dict)
    lock_field(lock, "releases", dict)
    lock_field(lock, "configurations", list)
    lock_field(lock, "plan", dict)


def pending(lock: dict) -> list[str]:
    """Release entries that still say PENDING_RELEASE."""
    releases = lock_field(lock, "releases", dict)
    return sorted(
        name
        for name, ref in releases.items()
        if isinstance(ref, dict) and PENDING in (ref.get("tag"), ref.get("commit"))
    )


def locked_configuration(lock: dict, name: str, root: Path) -> dict:
    configurations = lock_field(lock, "configurations", list)
    entry = next((c for c in configurations if isinstance(c, dict) and c.get("name") == name), None)
    if entry is None:
        raise SystemExit(f"{name} is not in the verification lock")
    if sha256(Path(root) / entry["path"]) != entry["sha256"]:
        raise SystemExit(
            f"{entry['path']} differs from the verification lock; update the lock in the same commit"
        )
    return entry


def parameters(robot: Robot, name: str, lock: dict) -> dict:
    return prepare_run(robot, name, lock).recorded


def prepare_run(robot: Robot, name: str, lock: dict | None = None) -> sc.Prepared:
    """Parameters, command and world for one configuration.

    The world is the vendored release named by ``lock["world"]``. Scenario
    parameter bytes still follow ``sources``: a lock entry, else the robot
    pins, else the vendored parameter file.
    """
    if lock is None:
        raise SystemExit(
            "INCOMPATIBLE: prepare_run needs the caller's verification lock. "
            "Example: prepare_run(robot, 'playground', lock)"
        )
    root = Path(robot.root)
    document, _source = sources.parameters_document_for(root, lock)
    world, digest, relative = compose.vendored_world(lock)
    # sources.world does not select the simulated world. A sha256 that does not
    # match the file it names is still an error, the same as world.sha256.
    if isinstance(lock.get("sources"), dict) and "world" in lock["sources"]:
        sources.world_source(root, lock)
    selected = sources.WorldSource(world, "vendored", digest, relative)
    refusal = frozen.unsupported_world(world)
    if refusal:
        raise SystemExit(refusal)
    try:
        prepared = sc.prepare(robot.robot(), robot.configuration(name), world, document)
    except ScenarioParameterError as exc:
        text = str(exc)
        raise SystemExit(text if text.startswith("ERROR") else f"ERROR: {text}") from exc
    # The world that was loaded and checked. The manifests name this one (#39).
    prepared.world = world
    prepared.world_source = selected
    return prepared


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


def run_trial(
    ref: str,
    name: str,
    plan: dict,
    p: dict,
    world: dict,
    on_step=None,
    *,
    start_offset=None,
    command: list[float] | None = None,
    route: list | None = None,
    exit_place: dict | None = None,
) -> dict:
    """One trial at the nominal pose through ``observations.observe``.

    ``on_step`` (for example ``livingproof_sim_mujoco.media.Recorder``) only reads
    ``data``, so it cannot change the observation. Every trial in the run uses this
    same call.
    """
    one_torch_thread()
    from livingproof_sim_mujoco.observations import observe

    robot = adapter(ref)
    rs = robot.runner_settings()
    model, data, ctrl, walls = compose.place(robot, name, world, p["spawn"], start_offset)
    policy = robot.policy(model, ctrl)
    if command is None:
        command = [p["command_speed"], 0.0, 0.0]
    hook = on_step(model, robot.base_body(name)) if on_step is not None else None
    obs = observe(
        model,
        data,
        lambda d: policy.step(d, command),
        base_body=robot.base_body(name),
        walls=walls,
        exit_x=p["exit_x"],
        seconds=p["time_limit"],
        sample_hz=rs["sample_hz"],
        on_step=hook,
        route=route,
        exit_place=exit_place,
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


def recording(robot: Robot, lock: dict, name: str) -> tuple[dict, sc.Prepared]:
    """(plan, prepared scenario) exactly as every evidence trial is recorded.

    ``time_limit`` is the parameter file's expression. The frozen scenario
    records it unrounded; the spec-rounded value is what ``resolve`` returns.
    """
    return lock_field(lock, "plan", dict), prepare_run(robot, name, lock)


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


def world_identity(root: Path, lock: dict, selected: sources.WorldSource) -> dict:
    """``{"id", "version", "sha256"}`` of the vendored world this run loaded.

    Lock ``world.id`` and ``world.version`` must agree with that file. The
    sha256 is the digest of the vendored ``world.json``. A lock ``world.sha256``
    that disagrees is an ERROR.
    """
    legacy = lock.get("world", {})
    if not isinstance(legacy, dict):
        raise SystemExit(
            f"ERROR: the verification lock world must be an object, got {legacy!r}. "
            'Example: "world": {"id": "<world>", "version": "0.1.0"}'
        )
    where = {
        "lock": "lock sources.world",
        "pins": "pinned world",
        "adapter": "the adapter's world",
        "vendored": "the vendored world",
    }[selected.origin]
    document = selected.document if isinstance(selected.document, dict) else {}
    ident = {}
    for key, field in (("id", "id"), ("version", "content_version")):
        declared, named = legacy.get(key), document.get(field)
        if declared is not None and named is not None and declared != named:
            raise SystemExit(
                f"ERROR: the verification lock world.{key} is {declared!r} but {where} has "
                f"{field} {named!r}. The run manifest would name a different world "
                f"from the one loaded. Make world.{key} match the file or remove it."
            )
        ident[key] = declared if declared is not None else named
        if not isinstance(ident[key], str) or not ident[key]:
            raise SystemExit(
                f"ERROR: world {key} is missing from the verification lock world and from {where}. "
                'Example: "world": {"id": "<world>", "version": "0.1.0"}'
            )
    if not isinstance(selected.sha256, str) or _SHA256.fullmatch(selected.sha256) is None:
        raise SystemExit(
            "INCOMPATIBLE: the loaded world has no sha256. The simulator loads the vendored "
            "world.json named by the lock. "
            'Example: "world": {"id": "<world>", "version": "0.1.0"}'
        )
    declared = legacy.get("sha256")
    if declared is not None and declared != selected.sha256:
        raise SystemExit(
            f"ERROR: the verification lock world.sha256 is {declared} but {where} is "
            f"{selected.sha256}. The run manifest would name a different world from "
            "the one loaded. Remove world.sha256 or set it to the loaded file's sha256."
        )
    ident["sha256"] = selected.sha256
    return ident


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


def _build_provenance_commit() -> tuple[str | None, str]:
    """``identity.commit`` from the packaged snapshot, and why it was not usable."""
    path = Path(__file__).resolve().parent / "_build_provenance.json"
    if not path.is_file():
        return None, "file is missing, so identity.commit was not read"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        commit = saved.get("identity", {}).get("commit") if isinstance(saved, dict) else None
    except (OSError, ValueError, AttributeError):
        return None, "file is not JSON, so identity.commit was not read"
    found = _valid_commit(commit)
    if found:
        return found, ""
    if commit in (None, ""):
        return None, "identity.commit is missing"
    return None, f"identity.commit {commit!r} is not 40 hex characters"


def _direct_url_commit() -> tuple[str | None, str]:
    """PEP 610 ``vcs_info.commit_id``, and why it was not usable."""
    try:
        text = metadata.distribution("livingproof-sim-mujoco").read_text("direct_url.json")
    except Exception:
        return (
            None,
            "the installed distribution has no direct_url.json, so vcs_info.commit_id was not read",
        )
    if not text:
        return None, "file is absent, so vcs_info.commit_id was not read"
    try:
        commit = json.loads(text).get("vcs_info", {}).get("commit_id")
    except (ValueError, AttributeError):
        return None, "vcs_info.commit_id is missing"
    found = _valid_commit(commit)
    if found:
        return found, ""
    if commit in (None, ""):
        return None, "vcs_info.commit_id is missing"
    return None, f"vcs_info.commit_id {commit!r} is not 40 hex characters"


def _git_source_commit() -> tuple[str | None, str]:
    """HEAD when this module is the package source tree, and why it was not usable."""
    root = _source_tree_root()
    if root is None:
        return None, "no source tree containing this package"
    found = _valid_commit(_git_text(root, "rev-parse", "HEAD"))
    if found:
        return found, ""
    return None, "source tree HEAD is not a 40-character commit"


def adapter_commit() -> str:
    """40-character commit of livingproof-sim-mujoco.

    When this module is the package's own source tree (a checkout or an editable
    install) and git HEAD resolves, that live HEAD is used, so a later commit is
    seen without reinstalling. If git does not resolve there, or this is an
    installed copy, use the packaged ``_build_provenance.json`` ``identity.commit``,
    then the installed distribution's PEP 610 ``direct_url.json``
    ``vcs_info.commit_id``. An enclosing repository is never used. When none
    yields a 40-character commit, exit 1 and name each source. The Debian package
    and the runtime image call this function; the Debian payload drops
    ``direct_url.json`` and keeps the snapshot.
    """
    git_commit, git_why = _git_source_commit()
    if git_commit:
        return git_commit
    package_commit, package_why = _build_provenance_commit()
    if package_commit:
        return package_commit
    direct_commit, direct_why = _direct_url_commit()
    if direct_commit:
        return direct_commit
    failed = [
        f"package provenance (_build_provenance.json): {package_why}",
        f"direct_url.json vcs_info: {direct_why}",
        f"git: {git_why}",
    ]
    raise SystemExit(
        "run manifest simulator.adapter_commit needs the 40-character git commit of "
        "livingproof-sim-mujoco. Every source failed:\n" + "\n".join(f"- {line}" for line in failed)
    )


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
    adapter_sha: str,
    world: dict | None = None,
) -> dict:
    """``livingproof-run-manifest/0.1``, written before the first trial.

    ``world`` is ``world_identity`` of the loaded world. Without it the world
    is resolved again from the lock, the same way ``prepare_run`` does.
    """
    if world is None:
        world = world_identity(robot.root, lock, prepare_run(robot, name, lock).world_source)
    specs = lock.get("releases", {}).get("specs")
    if not isinstance(specs, dict):
        raise SystemExit(
            "the verification lock releases.specs needs a tag and commit. "
            'Example: "specs": {"tag": "v2.2.0", "commit": "<40 hex>"}'
        )
    scenario = lock.get("scenario", frozen.SCENARIO)
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
            "id": world["id"],
            "version": world["version"],
            "sha256": _require("world.sha256", world["sha256"], _SHA256),
        },
        "simulator": {
            "engine": "mujoco",
            "version": mujoco.__version__,
            "adapter": "livingproof-sim-mujoco",
            "adapter_commit": _require("simulator.adapter_commit", adapter_sha, _COMMIT),
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
    ref: str,
    name: str,
    out: Path,
    *,
    lock: Path,
    trials: int | None = None,
    allow_pending=False,
) -> dict:
    # Before any output directory or manifest. An unknown commit exits 1.
    adapter_sha = adapter_commit()
    robot = adapter(ref)
    root = Path(robot.root)
    lock_path = Path(lock)
    lock = read_lock(lock_path)
    require_lock(lock)
    waiting = pending(lock)
    if waiting and not allow_pending:
        raise SystemExit(
            f"the verification lock has PENDING_RELEASE refs ({', '.join(waiting)}); run task repin"
        )
    entry = locked_configuration(lock, name, root)
    fetch(root)
    plan, prepared = recording(robot, lock, name)
    p = prepared.recorded
    world = prepared.world
    # Before any output: a conflicting or unknown world identity writes nothing.
    world_id = world_identity(root, lock, prepared.world_source)
    count = trial_count(plan, trials)
    offset = declared_offset(plan)
    profile = robot.robot()
    checked = sc.preconditions(
        profile,
        world,
        name,
        robot.configuration,
        prepared.document,
        width_decimals=robot.width_decimals,
    )
    unmet = unmet_required(checked)
    # A missing wall or an origin start writes nothing. An unmet precondition
    # does not build a model: the robot cannot attempt the scenario.
    if unmet is None:
        compose.place(robot, name, world, p["spawn"], offset)
    out = Path(out)
    (out / "trials").mkdir(parents=True, exist_ok=False)
    created = dt.datetime.now(dt.timezone.utc)
    run_id = frozen.evidence_run_id(lock.get("scenario", frozen.SCENARIO), robot.id, name, created)
    profile_sha = sha256(root / "robot.json")
    robot_commit = sc.git_head(root)[0]
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
            adapter_sha,
            world_id,
        )
        # Before the first trial, as the run-manifest schema requires.
        (out / "run-manifest.json").write_text(json.dumps(document, indent=2) + "\n")
        for index in range(1, count + 1):
            t0 = time.perf_counter()
            geometry = sc.route_geometry(world, prepared.document)
            route, exit_place = geometry if geometry else (None, None)
            obs = run_trial(
                ref,
                name,
                plan,
                p,
                world,
                start_offset=offset,
                command=prepared.command,
                route=route,
                exit_place=exit_place,
            )
            results.append(_write_trial(out, index, obs, time.perf_counter() - t0))
    wall_clock_s = round(time.perf_counter() - started, 3)
    report = reproducibility(results)
    manifest = {
        "format": FORMAT,
        "run_id": run_id,
        "created": created.isoformat(timespec="seconds"),
        "lock": {"path": lock_path.name, "sha256": sha256(lock_path), "content": lock},
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
            "id": world_id["id"],
            "version": world_id["version"],
            "clear_width": world["dimensions"]["clear_width"],
            "route_length": p["route_length"],
            "exit_x": p["exit_x"],
        },
        "environment": {
            "simulator": {
                "id": "mujoco",
                "version": mujoco.__version__,
                "adapter": "livingproof-sim-mujoco",
                "adapter_commit": adapter_sha,
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
                    "walls": compose.wall_names(world),
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
        "--lock",
        type=Path,
        required=True,
        help="caller verification lock (scenario, world id and version, plan, configurations). "
        "Example: --lock verifications/unitree-g1-playground.lock.json",
    )
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
        "not a robot grade. An INCOMPATIBLE or UNSUPPORTED world also exits 1. "
        "Example: python -m livingproof_sim_mujoco.evidence "
        "lpl_g1.adapter:ROBOT playground out --lock path/to/lock.json --trials 3"
    )
    a = ap.parse_args(argv)
    m = evidence(
        a.adapter,
        a.configuration,
        a.out,
        lock=a.lock,
        trials=a.trials,
        allow_pending=a.allow_pending,
    )
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
