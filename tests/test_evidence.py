"""Generic walk-hallway evidence: pins, scenario and the robot-adapter loop (toy robot)."""

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import pytest

from livingproof_sim_mujoco import evidence, observations, pins, scenario

WORLD = {
    "dimensions": {"clear_width": 1.6},
    "places": [
        {"name": "spawn-entrance", "position": [-0.5, 0.0, 0.0]},
        {"name": "exit", "reference": [0.5, 0.0, 0.0]},
    ],
}
ROBOT = {
    "class": "humanoid",
    "dimensions": {"width": {"value": 0.4}},
    "simulators": ["mujoco"],
}
CONF = {
    "locomotion": {"fall_base_height": {"value": 0.1}},
    "controller": {"interface": "velocity_command"},
}
XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="left" type="box" size="2 0.05 0.5" pos="0 0.85 0.5" contype="0" conaffinity="0"/>
<body name="base" pos="-0.5 0 0.3"><freejoint/><geom type="sphere" size="0.1"/></body>
</worldbody></mujoco>"""


class Unsupported(Exception):
    def __init__(self, precondition, detail):
        super().__init__(detail)
        self.precondition, self.detail = precondition, detail


class Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.walls = ["left"]


class Push:
    def step(self, data, command):
        data.qvel[0] = command[0] * 4  # glide forward; toy "controller"
        mujoco.mj_step(Scene.model_cache, data)


class Toy:
    id = "toy-robot"
    root = None
    width_decimals = 2

    def base_body(self, name):
        return "base"

    def scene(self, name):
        s = Scene()
        Scene.model_cache = s.model
        return s

    def controller(self, name, scene):
        return None

    def policy(self, model, ctrl):
        return Push()

    def initial_state(self, model, data, ctrl, spawn, *, start_offset=None):
        mujoco.mj_resetData(model, data)
        if start_offset and "base_xy_m" in start_offset:
            data.qpos[0] = start_offset["base_xy_m"][0]
        mujoco.mj_forward(model, data)

    def runner_settings(self):
        return {"sample_hz": 50.0}

    def robot(self):
        return ROBOT

    def configuration(self, name):
        if name != "stock":
            raise Unsupported("configuration_offered", f"{name} is not offered")
        return CONF

    def world(self):
        return WORLD

    def environment(self, name):
        return {"model": "toy"}


TOY = Toy()


def test_scenario_parameters_and_preconditions():
    p = scenario.parameters(ROBOT, CONF, WORLD)
    assert p["route_length"] == 1.0 and p["exit_x"] == 0.5 and p["lateral_limit"] == 0.6
    assert p["time_limit"] == 2 * 1.0 / 0.5 + 5 and p["wall_clearance"] == 0.05
    pre = scenario.preconditions(ROBOT, WORLD, "other", TOY.configuration)
    assert [c["name"] for c in pre] == ["fits_width", "configuration_offered", "simulator_declared"]
    assert pre[0]["detail"] == "0.4 + 2 x clearance = 0.50 m vs 1.6 m" and not pre[1]["met"]
    assert (
        "0.500 m"
        in scenario.preconditions(ROBOT, WORLD, "stock", TOY.configuration, width_decimals=3)[0][
            "detail"
        ]
    )


def test_pins_fetch_verifies_and_never_accepts_a_wrong_download(tmp_path):
    data = b"pinned"
    good = hashlib.sha256(data).hexdigest()
    doc = {
        "groups": {
            "g": {
                "repository": "o/r",
                "commit": "c",
                "dest": ".cache/g",
                "files": {"a.txt": {"sha256": good}},
            }
        }
    }
    (tmp_path / "pins.json").write_text(json.dumps(doc))
    seen = []

    class R:
        def __init__(self, b):
            self.b = b

        def read(self):
            return self.b

    def opener(url):
        seen.append(url)
        return R(data)

    assert pins.fetch(tmp_path, opener) == [tmp_path / ".cache/g/a.txt"]
    assert seen == ["https://raw.githubusercontent.com/o/r/c/a.txt"]
    pins.fetch(tmp_path, opener)
    assert len(seen) == 1  # cached and verified
    doc["groups"]["g"]["files"]["a.txt"]["sha256"] = "0" * 64
    (tmp_path / "pins.json").write_text(json.dumps(doc))
    with pytest.raises(pins.ChecksumError):
        pins.fetch(tmp_path, lambda url: R(data))


def _repo(tmp_path):
    (tmp_path / "pins.json").write_text(json.dumps({"groups": {}}))
    (tmp_path / "robot.json").write_text(json.dumps(ROBOT))
    (tmp_path / "stock.json").write_text(json.dumps(CONF))
    lock = {
        "plan": {"trials": 20, "seeds": [1, 2], "command_speed": 0.5, "required_passes": 2},
        "scenario": "walk-hallway@0.1.0",
        "world": {
            "id": "hallway-open",
            "version": "0.1.0",
            "sha256": "ab" * 32,
        },
        "configurations": [
            {
                "name": "stock",
                "path": "stock.json",
                "version": "1.0.0",
                "sha256": pins.sha256(tmp_path / "stock.json"),
            }
        ],
        "releases": {
            "runner": {"tag": "v1.5.0", "commit": "a" * 40},
            "specs": {
                "tag": "v2.0.2",
                "commit": "16524cbfeef18ff0454ebf125668b010d38a3993",
            },
        },
    }
    (tmp_path / "verification.lock.json").write_text(json.dumps(lock))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "t",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://example.test/toy-robot.git",
        ],
        check=True,
    )
    return lock


def test_evidence_runs_identical_trials_and_ignores_seeds(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    one = evidence.evidence("test_evidence:TOY", "stock", tmp_path / "one")
    assert one["trials"]["count"] == 1  # plan trials: 20 and seeds are not the count
    assert evidence.FORMAT == "livingproof-hallway-evidence/2.0.0"
    assert one["plan"] == {"command_speed": 0.5, "trials": {"count": 1}}
    assert "seeds" not in one["plan"] and "required_passes" not in one["plan"]
    assert one["trials"]["reproducibility"]["compared"] is False
    m = evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out", trials=2)
    assert m["format"] == evidence.FORMAT and m["robot"]["id"] == "toy-robot"
    assert [t["index"] for t in m["trials"]["results"]] == [1, 2]
    assert m["trials"]["count"] == 2
    assert {t["outcome"] for t in m["trials"]["results"]} == {"crossed_exit"}
    assert m["trials"]["reproducibility"] == {
        "compared": True,
        "identical": True,
        "finding": None,
    }
    assert m["trials"]["results"][0]["sha256"] == m["trials"]["results"][1]["sha256"]
    assert (
        m["trials"]["results"][0]["trajectory_sha256"]
        == m["trials"]["results"][1]["trajectory_sha256"]
    )
    assert m["environment"]["walls"] == ["left"] and m["environment"]["model"] == "toy"
    assert m["start"] == {"spawn": "spawn-entrance", "pose": "nominal", "offset": None}
    obs = json.loads((tmp_path / "out" / "trials" / "trial-01.json").read_text())
    assert obs["end"] == "crossed_exit"
    run = json.loads((tmp_path / "out" / "run-manifest.json").read_text())
    assert run["format"] == evidence.RUN_MANIFEST_FORMAT
    assert run["trials"] == {"count": 2}
    assert run["scenario"] == "walk-hallway@0.1.0"
    assert run["configuration"]["name"] == "stock"
    assert set(run) == {
        "format",
        "run_id",
        "specification",
        "scenario",
        "world",
        "simulator",
        "robot",
        "configuration",
        "parameters",
        "trials",
    }
    with pytest.raises(SystemExit, match="is not in"):
        evidence.evidence("test_evidence:TOY", "other", tmp_path / "out2")
    assert (
        evidence.main(["test_evidence:TOY", "stock", str(tmp_path / "agree"), "--trials", "2"]) == 0
    )


def test_unmet_precondition_never_starts_the_simulator_or_a_worker(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    calls = []

    def spy(what):
        def f(*a, **k):
            calls.append(what)
            raise AssertionError(f"{what} called for an unsupported robot")

        return f

    monkeypatch.setattr(evidence, "run_trial", spy("run_trial"))
    monkeypatch.setattr(observations, "observe", spy("observe"))
    monkeypatch.setattr(evidence.mujoco, "MjData", spy("MjData"))
    unmet = {"name": "fits_width", "met": False, "detail": "1.41 m vs 1.4 m"}
    monkeypatch.setattr(
        evidence.sc,
        "preconditions",
        lambda *a, **k: [unmet, {"name": "simulator_declared", "met": True, "detail": "x"}],
    )
    m = evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out", trials=2)
    assert calls == []
    assert m["trials"]["results"] == [] and m["trials"]["count"] == 2
    assert m["preconditions"][0] == unmet
    assert list((tmp_path / "out" / "trials").iterdir()) == []
    assert not (tmp_path / "out" / "run-manifest.json").exists()
    written = json.loads((tmp_path / "out" / "manifest.json").read_text())["trials"]
    assert written["results"] == []
    assert "walls" not in m["environment"]


def test_optional_unmet_precondition_still_runs_the_trials(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    real = evidence.sc.preconditions
    optional = {"name": "nice_to_have", "met": False, "detail": "x", "required": False}
    monkeypatch.setattr(evidence.sc, "preconditions", lambda *a, **k: real(*a, **k) + [optional])
    m = evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out", trials=2)
    assert [t["index"] for t in m["trials"]["results"]] == [1, 2]


def test_evidence_refuses_pending_refs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    lock = _repo(repo)
    lock["releases"]["runner"]["tag"] = evidence.PENDING
    (repo / "verification.lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    with pytest.raises(SystemExit, match="PENDING_RELEASE refs \\(runner\\)"):
        evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stand_runs_the_zero_command_check_through_the_adapter(tmp_path, monkeypatch):
    from livingproof_sim_mujoco import walk

    repo = tmp_path / "repo"
    repo.mkdir()
    lock = _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.setattr(
        Toy, "runner_settings", lambda self: {"sample_hz": 50.0, "stand_seconds": 0.2}
    )
    result = walk.stand(TOY, "stock", tmp_path / "out")
    assert result["check"] == "zero-command-stand" and "seed" not in result
    assert result["configuration_sha256"] == lock["configurations"][0]["sha256"]
    assert result["seconds"] == 0.2 and result["clip"] == "stand-zero-command.mp4"
    assert json.loads((tmp_path / "out" / "stand.json").read_text()) == result


def test_stand_and_media_use_the_optional_adapter_camera(tmp_path, monkeypatch):
    from livingproof_sim_mujoco import walk

    seen = []
    monkeypatch.setattr(
        walk, "simulate", lambda *a, camera_view, **k: seen.append(camera_view) or ([], None, None)
    )
    monkeypatch.setattr(walk, "stand_summary", lambda *a: {})
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.setattr(
        Toy, "runner_settings", lambda self: {"sample_hz": 50.0, "stand_seconds": 0.2}
    )
    walk.stand(TOY, "stock")
    monkeypatch.setattr(Toy, "camera", (2.2, -35, 200), raising=False)
    walk.stand(TOY, "stock")
    assert seen == [walk.CAMERA, (2.2, -35.0, 200.0)]
    monkeypatch.setattr(Toy, "camera", (0, 1), raising=False)
    with pytest.raises(ValueError, match="camera must be"):
        evidence.camera(TOY, walk.CAMERA)


def test_robot_protocol_lists_every_adapter_method():
    members = {n for n in vars(evidence.Robot) if not n.startswith("_")}
    assert {"configuration", "world", "environment", "robot"} <= members
    assert "camera" not in members  # optional attribute, read through evidence.camera()


def test_disagreeing_trials_are_a_test_system_finding(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    real = evidence.run_trial
    calls = {"n": 0}

    def vary(*args, **kwargs):
        obs = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            obs = {**obs, "end": "fell"}
        return obs

    monkeypatch.setattr(evidence, "run_trial", vary)
    m = evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out", trials=2)
    finding = m["trials"]["reproducibility"]["finding"]
    assert m["trials"]["reproducibility"]["identical"] is False
    assert finding["kind"] == "test-system" and finding["name"] == "reproducibility"
    assert "not a robot FAIL" in finding["detail"]
    assert finding["outcomes"] == ["crossed_exit", "fell"]
    assert all("status" not in t for t in m["trials"]["results"])
    code = evidence.main(["test_evidence:TOY", "stock", str(tmp_path / "cli"), "--trials", "2"])
    assert code == evidence.EXIT_REPRODUCIBILITY == 4
    assert code not in {0, 1, 2}
    written = json.loads((tmp_path / "cli" / "manifest.json").read_text())
    assert written["trials"]["reproducibility"]["finding"]["kind"] == "test-system"
    assert written["trials"]["reproducibility"]["finding"]["name"] == "reproducibility"
    assert "not a robot FAIL" in written["trials"]["reproducibility"]["finding"]["detail"]


def test_nominal_start_accepts_a_retired_adapter_and_refuses_its_offset():
    seen = []

    class Legacy:
        id = "legacy"

        def initial_state(self, model, data, ctrl, seed, noise, spawn):
            seen.append((seed, noise, spawn))

    evidence.nominal_start(Legacy(), None, None, None, "spawn-entrance", None)
    assert seen == [(0, None, "spawn-entrance")]
    with pytest.raises(SystemExit, match="start_offset"):
        evidence.nominal_start(
            Legacy(), None, None, None, "spawn-entrance", {"base_xy_m": [0.0, 0.05]}
        )


def test_trials_are_not_dispatched_on_a_cpu_count_pool():
    from livingproof_sim_mujoco import media

    evidence_src = Path(evidence.__file__).read_text()
    media_src = Path(media.__file__).read_text()
    assert "ProcessPool" not in evidence_src and "cpu_count" not in evidence_src
    assert "ProcessPool" not in media_src and "cpu_count" not in media_src
    assert "pool" not in evidence_src
