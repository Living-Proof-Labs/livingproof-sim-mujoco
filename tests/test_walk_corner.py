# SPDX-License-Identifier: Apache-2.0
"""walk-corner 0.1.0 on hallway-corner 0.1.0 (robot-capability-specs#40, step 11a).

The parameter file has no command. A robot whose manifest declares walk-corner is
driven by its own adapter (``drive``); the runner sends no velocity and records what
happens: the route observation (``livingproof-route-observation/0.1.0``) with the
exit crossing, the places passed through, the route distance, the base height and
every wall contact. An undeclared robot stays UNSUPPORTED before any trial, and a
declared adapter with no ``drive`` is a test-system ERROR that writes nothing.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import pytest
import yaml

from livingproof_sim_mujoco import compose, evidence, hallway, observations, pins, sources

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/livingproof_sim_mujoco"
SPECS = ROOT / "tests/fixtures/robot-capability-specs-v2.3.0"
WORLDS_V130 = "12915e34e6ad1dfe376ec2fcb49d5afb957caff9"
SPECS_V230 = "57317e617014a6df97bb7fba9d92a890af8d42e5"
ROBOT = {
    "class": "humanoid",
    "dimensions": {"width": {"value": 0.4, "unit": "m"}},
    "simulators": ["mujoco"],
}
CONF = {
    "locomotion": {"fall_base_height": {"value": 0.05, "unit": "m"}},
    "controller": {"interface": "velocity_command"},
}
# The robot scene brings its own floor; the composed world's walls are measured, not
# collided with, as in the frozen scenario.
XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="floor" type="plane" size="10 10 0.1"/>
<body name="base" pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
</worldbody></mujoco>"""


class Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)


class Glide:
    """A holonomic toy: ``step(data, command)`` sets the planar velocity."""

    def __init__(self, model):
        self.model = model

    def step(self, data, command):
        data.qvel[0], data.qvel[1] = command[0], command[1]
        mujoco.mj_step(self.model, data)


def _follow(policy, route, speed=1.0):
    """Head for each route point in turn (the toy's own way around the corner)."""
    points = [p[:2] for p in route["points"]]
    end = points[-1]
    # Past the last point, keep going the way the last segment points.
    beyond = [2 * end[0] - points[-2][0], 2 * end[1] - points[-2][1]]
    targets = points[1:] + [beyond]
    state = {"i": 0}

    def step(data):
        x, y = float(data.qpos[0]), float(data.qpos[1])
        tx, ty = targets[state["i"]]
        if state["i"] < len(targets) - 1 and (tx - x) ** 2 + (ty - y) ** 2 < 0.05**2:
            state["i"] += 1
            tx, ty = targets[state["i"]]
        dx, dy = tx - x, ty - y
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        policy.step(data, [speed * dx / norm, speed * dy / norm, 0.0])

    return step


class Corner:
    """A toy robot that declares walk-corner and drives it its own way."""

    id = "toy-corner"
    root = None
    width_decimals = 2
    straight = False

    def base_body(self, name):
        return "base"

    def scene(self, name):
        return Scene()

    def controller(self, name, scene):
        return None

    def policy(self, model, ctrl):
        return Glide(model)

    def initial_state(self, model, data, ctrl, spawn, *, start_offset=None):
        mujoco.mj_resetData(model, data)
        site = model.site("place:" + spawn)
        data.qpos[0], data.qpos[1] = float(site.pos[0]), float(site.pos[1])
        data.qpos[2] = 0.1
        mujoco.mj_forward(model, data)

    def runner_settings(self):
        return {"sample_hz": 50.0}

    def robot(self):
        return ROBOT

    def configuration(self, name):
        return CONF

    def environment(self, name):
        return {"model": "toy"}

    def drive(self, name, model, ctrl, policy, places):
        self.places = places
        if self.straight:
            return lambda data: policy.step(data, [1.0, 0.0, 0.0])
        return _follow(policy, places["route"])


class NoDrive(Corner):
    id = "toy-no-drive"
    drive = None


CORNER = Corner()
NO_DRIVE = NoDrive()


def _manifest(scenarios: list[str]) -> dict:
    return {
        "format": "livingproof-robot-manifest/0.1",
        "robot": "toy-corner",
        "source": {"repository": "Example/toy-corner", "commit": "a" * 40},
        "profile": {"path": "robot.json", "sha256": "b" * 64},
        "adapter": {"entry_point": "test_walk_corner:CORNER", "package": "toy-corner"},
        "configurations": [
            {"name": "stock", "version": "1.0.0", "path": "stock.json", "sha256": "c" * 64}
        ],
        "assets": [
            {
                "repository": "Example/toy-corner",
                "commit": "d" * 40,
                "path": "robot.json",
                "sha256": "e" * 64,
                "bytes": 1,
            }
        ],
        "features": {},
        "scenarios": scenarios,
    }


def _repo(tmp_path, monkeypatch, scenarios=("walk-corner",), cls=Corner) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pins.json").write_text(json.dumps({"groups": {}}))
    (repo / "robot.json").write_text(json.dumps(ROBOT))
    (repo / "stock.json").write_text(json.dumps(CONF))
    (repo / "manifest.json").write_text(json.dumps(_manifest(list(scenarios))))
    world = PACKAGE / "worlds/hallway-corner/0.1.0/world.json"
    lock = {
        "plan": {"trials": 1},
        "scenario": "walk-corner@0.1.0",
        "world": {"id": "hallway-corner", "version": "0.1.0", "sha256": pins.sha256(world)},
        "configurations": [
            {
                "name": "stock",
                "path": "stock.json",
                "version": "1.0.0",
                "sha256": pins.sha256(repo / "stock.json"),
            }
        ],
        "releases": {
            "runner": {"tag": "v3.2.0", "commit": "a" * 40},
            "specs": {"tag": "v2.3.0", "commit": SPECS_V230},
        },
    }
    (repo / "caller.lock.json").write_text(json.dumps(lock))
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "t"], check=True)
    subprocess.run([*git, "remote", "add", "origin", "https://example.test/toy.git"], check=True)
    monkeypatch.setattr(cls, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return repo


def _run(repo: Path, out: Path, entry="test_walk_corner:CORNER", **kwargs) -> dict:
    return evidence.evidence(entry, "stock", out, lock=repo / "caller.lock.json", **kwargs)


def _support(manifest: dict) -> dict:
    return next(p for p in manifest["preconditions"] if p["name"] == "scenario_supported")


def test_vendored_corner_world_is_worlds_v1_3_0():
    entry = json.loads((PACKAGE / "worlds/pins.json").read_text())
    text = json.dumps(entry)
    assert "hallway-corner" in text and WORLDS_V130 in text
    loaded = hallway.load_fixture("hallway-corner", "0.1.0")
    assert loaded["world"]["format"] == "livingproof-world/0.2"
    world, digest, _relative = compose.vendored_world(
        {"world": {"id": "hallway-corner", "version": "0.1.0"}}
    )
    assert digest == pins.sha256(PACKAGE / "worlds/hallway-corner/0.1.0/world.json")
    assert {p["name"] for p in world["places"]} >= {
        "spawn-entrance",
        "exit",
        "corner",
        "route-centreline",
    }


def test_vendored_walk_corner_parameters_are_specs_v2_3_0(tmp_path):
    catalog = json.loads((PACKAGE / "scenarios/catalog.json").read_text())
    entry = catalog["walk-corner@0.1.0"]
    assert entry["tag"] == "v2.3.0" and entry["commit"] == SPECS_V230
    raw = (PACKAGE / "scenarios" / entry["file"]).read_bytes()
    assert raw == (SPECS / "walk-corner-parameters.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    document, origin = sources.parameters_document_for(tmp_path, {"scenario": "walk-corner@0.1.0"})
    assert origin == "vendored" and document["scenario"]["id"] == "walk-corner"
    assert "command" not in document


def test_declared_robot_is_driven_around_the_corner_and_observed(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    out = tmp_path / "out"
    recorded = _run(repo, out)
    assert _support(recorded) == {
        "name": "scenario_supported",
        "met": True,
        "detail": "walk-corner",
    }
    assert recorded["world"]["id"] == "hallway-corner"
    assert "exit_x" not in recorded["world"]
    (trial,) = recorded["trials"]["results"]
    assert trial["outcome"] == "crossed_exit"
    obs = json.loads((out / trial["path"]).read_text())
    assert obs["format"] == observations.ROUTE_FORMAT
    assert obs["end"] == "crossed_exit" and obs["left_world"] is None
    assert obs["places"] == {"route": "route-centreline", "exit": "exit", "through": ["corner"]}
    corner = obs["places_entered"]["corner"]
    assert corner is not None and corner < obs["exit_time"]
    assert obs["wall_contacts"] == []
    assert obs["max_route_distance"]["value"] < 0.1
    assert obs["min_base_height"]["value"] > 0.05
    assert obs["trace"]["columns"] == observations.ROUTE_COLUMNS
    # The adapter was handed the world's own places.
    assert CORNER.places["through"][0]["type"] == "region"
    assert (out / "run-manifest.json").is_file()


def test_robot_that_drives_straight_meets_the_wall_and_it_is_recorded(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(Corner, "straight", True)
    out = tmp_path / "out"
    (trial,) = _run(repo, out)["trials"]["results"]
    obs = json.loads((out / trial["path"]).read_text())
    assert obs["end"] == "left_world" and obs["exit_time"] is None
    assert obs["left_world"]["via"] == "wall-outer-second"
    assert obs["wall_contacts"], "the outer wall of the turn is touched"
    assert obs["wall_contacts"][0]["wall"] == "wall-outer-second"
    assert obs["wall_contacts"][0]["start"] < obs["left_world"]["time"]
    assert obs["places_entered"]["corner"] is not None
    assert obs["max_route_distance"]["value"] > 0.5


def test_undeclared_robot_is_unsupported_with_no_trial(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch, scenarios=("walk-hallway",))
    out = tmp_path / "out"
    recorded = _run(repo, out)
    assert _support(recorded)["met"] is False
    assert recorded["trials"]["results"] == []
    assert not (out / "run-manifest.json").exists()


def test_declared_adapter_without_drive_is_an_error_and_writes_nothing(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch, cls=NoDrive)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=r"^ERROR: .*no drive\(\)"):
        _run(repo, out, entry="test_walk_corner:NO_DRIVE")
    assert not out.exists()


def test_route_wall_faces_put_each_inner_face_toward_the_route():
    world, _digest, _relative = compose.vendored_world(
        {"world": {"id": "hallway-corner", "version": "0.1.0"}}
    )
    model, _data, _ctrl, walls = compose.place(CORNER, "stock", world, "spawn-entrance")
    route = next(p for p in world["places"] if p["name"] == "route-centreline")["points"]
    faces = observations.route_wall_faces(model, walls, route)
    inner = {name: (f["axis"], f["side"], f["inner"]) for name, f in faces.items()}
    assert inner == {
        "wall-outer-first": ("y", 1, -0.75),
        "wall-outer-second": ("x", -1, 0.75),
        "wall-inner-first": ("y", -1, 0.75),
        "wall-inner-second": ("x", 1, -0.75),
    }


def test_ci_test_task_runs_walk_corner():
    """CI runs ``task test``; this file must be in it (Vera, review of PR #68)."""
    tasks = yaml.safe_load((ROOT / "Taskfile.runtime.yml").read_text())["tasks"]
    commands = " ".join(str(c) for c in tasks["test"]["cmds"])
    assert "tests/test_walk_corner.py" in commands
