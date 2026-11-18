"""Generic walk-hallway evidence: pins, scenario and the robot-adapter loop (toy robot)."""

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import warnings

import mujoco
import pytest

from livingproof_sim_mujoco import evidence, observations, pins, scenario

WORLD = {
    "format": "livingproof-world/0.1",
    "dimensions": {"clear_width": 1.6},
    "places": [
        {"name": "spawn-entrance", "type": "pose", "position": [-0.5, 0.0, 0.0], "yaw_deg": 0.0},
        {
            "name": "exit",
            "type": "segment",
            "points": [[0.5, -0.8, 0.0], [0.5, 0.8, 0.0]],
            "reference": [0.5, 0.0, 0.0],
        },
        {
            "name": "passage-centreline",
            "type": "polyline",
            "points": [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        },
    ],
}
ROBOT = {
    "class": "humanoid",
    "dimensions": {"width": {"value": 0.4, "unit": "m"}},
    "simulators": ["mujoco"],
}
CONF = {
    "locomotion": {"fall_base_height": {"value": 0.1, "unit": "m"}},
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
    def __init__(self, model):
        self.model = model

    def step(self, data, command):
        data.qvel[0] = command[0] * 4  # glide forward; toy controller
        mujoco.mj_step(self.model, data)


class Toy:
    id = "toy-robot"
    root = None
    width_decimals = 2

    def base_body(self, name):
        return "base"

    def scene(self, name):
        return Scene()

    def controller(self, name, scene):
        return None

    def policy(self, model, ctrl):
        return Push(model)

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


CALLER_LOCK = "caller.lock.json"
TOY = Toy()


def _vendored_world_sha() -> str:
    from livingproof_sim_mujoco import hallway

    return pins.sha256(hallway.WORLDS / "hallway-open" / "0.1.0" / "world.json")


def _lock(repo: Path) -> Path:
    return repo / CALLER_LOCK


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
                "files": {"a.txt": {"sha256": good, "bytes": len(data)}},
            }
        }
    }
    (tmp_path / "pins.json").write_text(json.dumps(doc))
    seen = []

    class R:
        def __init__(self, b):
            self.b = b

        def read(self, _size=-1):
            data, self.b = self.b, b""
            return data

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
            "sha256": _vendored_world_sha(),
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
    parameters = (
        Path(__file__).resolve().parents[1] / "src/livingproof_sim_mujoco/compat/parameters.json"
    )
    dest = tmp_path / "parameters.json"
    dest.write_bytes(parameters.read_bytes())
    lock["sources"] = {
        "parameters": {"path": "parameters.json", "sha256": pins.sha256(dest)},
    }
    (tmp_path / CALLER_LOCK).write_text(json.dumps(lock))
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
    one = evidence.evidence("test_evidence:TOY", "stock", tmp_path / "one", lock=_lock(repo))
    assert one["trials"]["count"] == 1  # plan trials: 20 and seeds are not the count
    assert evidence.FORMAT == "livingproof-hallway-evidence/2.0.0"
    assert one["plan"] == {"command_speed": 0.5, "trials": {"count": 1}}
    assert "seeds" not in one["plan"] and "required_passes" not in one["plan"]
    assert one["trials"]["reproducibility"]["compared"] is False
    m = evidence.evidence(
        "test_evidence:TOY", "stock", tmp_path / "out", trials=2, lock=_lock(repo)
    )
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
    assert (
        m["environment"]["walls"] == ["wall-left", "wall-right"]
        and m["environment"]["model"] == "toy"
    )
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
        evidence.evidence("test_evidence:TOY", "other", tmp_path / "out2", lock=_lock(repo))
    assert (
        evidence.main(
            [
                "test_evidence:TOY",
                "stock",
                str(tmp_path / "agree"),
                "--lock",
                str(_lock(repo)),
                "--trials",
                "2",
            ]
        )
        == 0
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
    m = evidence.evidence(
        "test_evidence:TOY", "stock", tmp_path / "out", trials=2, lock=_lock(repo)
    )
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
    m = evidence.evidence(
        "test_evidence:TOY", "stock", tmp_path / "out", trials=2, lock=_lock(repo)
    )
    assert [t["index"] for t in m["trials"]["results"]] == [1, 2]


def test_evidence_refuses_pending_refs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    lock = _repo(repo)
    lock["releases"]["runner"]["tag"] = evidence.PENDING
    (repo / CALLER_LOCK).write_text(json.dumps(lock))
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    with pytest.raises(SystemExit, match="PENDING_RELEASE refs \\(runner\\)"):
        evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out", lock=_lock(repo))
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
    result = walk.stand(TOY, "stock", tmp_path / "out", lock=_lock(repo))
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
    walk.stand(TOY, "stock", lock=_lock(repo))
    monkeypatch.setattr(Toy, "camera", (2.2, -35, 200), raising=False)
    walk.stand(TOY, "stock", lock=_lock(repo))
    assert seen == [walk.CAMERA, (2.2, -35.0, 200.0)]
    monkeypatch.setattr(Toy, "camera", (0, 1), raising=False)
    with pytest.raises(ValueError, match="camera must be"):
        evidence.camera(TOY, walk.CAMERA)


def test_robot_protocol_lists_every_adapter_method():
    members = {n for n in vars(evidence.Robot) if not n.startswith("_")}
    assert {"configuration", "environment", "robot"} <= members
    assert "world" not in members  # the simulator supplies the world
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
    m = evidence.evidence(
        "test_evidence:TOY", "stock", tmp_path / "out", trials=2, lock=_lock(repo)
    )
    finding = m["trials"]["reproducibility"]["finding"]
    assert m["trials"]["reproducibility"]["identical"] is False
    assert finding["kind"] == "test-system" and finding["name"] == "reproducibility"
    assert "not a robot FAIL" in finding["detail"]
    assert finding["outcomes"] == ["crossed_exit", "fell"]
    assert all("status" not in t for t in m["trials"]["results"])
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(tmp_path / "cli"),
            "--lock",
            str(_lock(repo)),
            "--trials",
            "2",
        ]
    )
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


@pytest.mark.parametrize("missing", ["path", "sha256"])
def test_configuration_entry_missing_field_names_it_and_writes_nothing(
    tmp_path, monkeypatch, missing
):
    """A configuration entry without path or sha256 names that field and writes nothing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    lock = _repo(repo)
    del lock["configurations"][0][missing]
    (repo / CALLER_LOCK).write_text(json.dumps(lock))
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=missing) as raised:
        evidence.evidence("test_evidence:TOY", "stock", out, lock=_lock(repo), trials=1)
    message = str(raised.value)
    assert "INCOMPATIBLE" in message
    assert "KeyError" not in message
    assert not out.exists()


def test_empty_releases_exits_and_leaves_no_output_directory(tmp_path, monkeypatch):
    """``releases: {}`` exits 1 and does not leave the output directory behind."""
    repo = tmp_path / "repo"
    repo.mkdir()
    lock = _repo(repo)
    lock["releases"] = {}
    (repo / CALLER_LOCK).write_text(json.dumps(lock))
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="releases.specs") as raised:
        evidence.evidence("test_evidence:TOY", "stock", out, lock=_lock(repo), trials=1)
    assert "INCOMPATIBLE" in str(raised.value) or "releases.specs" in str(raised.value)
    assert not out.exists()


def test_trials_are_not_dispatched_on_a_cpu_count_pool():
    from livingproof_sim_mujoco import media

    evidence_src = Path(evidence.__file__).read_text()
    media_src = Path(media.__file__).read_text()
    assert "ProcessPool" not in evidence_src and "cpu_count" not in evidence_src
    assert "ProcessPool" not in media_src and "cpu_count" not in media_src
    assert "pool" not in evidence_src


# Vera review of #77 (#38): the zero-command stand now runs in CI on EGL. These make it
# fail when the adapter runs but does not stand, ignores the adapter's policy, or
# writes no clip, not only when the adapter raises.


def _stand(tmp_path, monkeypatch):
    from livingproof_sim_mujoco import walk

    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.setattr(
        Toy, "runner_settings", lambda self: {"sample_hz": 50.0, "stand_seconds": 0.2}
    )
    return walk.stand(TOY, "stock", tmp_path / "out", lock=_lock(repo))


def test_zero_command_stand_records_a_standing_robot_and_a_real_clip(tmp_path, monkeypatch):
    commands = []
    policy = Toy.policy

    def recording(self, model, ctrl):
        inner = policy(self, model, ctrl)

        class Recording:
            def step(self, data, command):
                commands.append(list(command))
                return inner.step(data, command)

        return Recording()

    monkeypatch.setattr(Toy, "policy", recording)
    result = _stand(tmp_path, monkeypatch)
    assert result["standing"] is True and result["end"] == "time_limit"
    assert commands and all(command == [0.0, 0.0, 0.0] for command in commands)
    clip = tmp_path / "out" / result["clip"]
    assert clip.read_bytes()[4:8] == b"ftyp"  # an MP4 the renderer actually wrote


def _falls(self, model, data, ctrl, spawn, *, start_offset=None):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = -1.0  # base below any fall height
    mujoco.mj_forward(model, data)


class _NonFinite:
    def step(self, data, command):
        data.qpos[:] = float("nan")


@pytest.mark.parametrize(
    ("method", "broken", "end"),
    [
        ("initial_state", _falls, "fell"),
        ("policy", lambda self, model, ctrl: _NonFinite(), "non_finite_state"),
    ],
)
def test_zero_command_stand_reports_an_adapter_that_runs_but_cannot_stand(
    tmp_path, monkeypatch, method, broken, end
):
    monkeypatch.setattr(Toy, method, broken)
    result = _stand(tmp_path, monkeypatch)
    assert result["standing"] is False and result["end"] == end


XML_WITH_VISUAL_FALLBACK = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="left" type="box" size="2 0.05 0.5" pos="0 0.85 0.5" contype="0" conaffinity="0"/>
<body name="base" pos="-0.5 0 0.3"><freejoint/><geom type="sphere" size="0.1"/>
  <body name="arm" pos="0 0.15 0">
    <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0" group="1"/>
  </body>
</body>
</worldbody></mujoco>"""


class FallbackScene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML_WITH_VISUAL_FALLBACK)
        self.walls = ["left"]


class FallbackToy(Toy):
    pass


FALLBACK_TOY = FallbackToy()


def test_visual_fallback_logs_once_per_run_and_records_manifest_summary(
    tmp_path, monkeypatch, caplog, capsys
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(FallbackToy, "root", repo)
    monkeypatch.setattr(FallbackToy, "scene", lambda self, name: FallbackScene())
    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    out = tmp_path / "out"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO, logger="livingproof_sim_mujoco"):
            m = evidence.evidence(
                "test_evidence:FALLBACK_TOY", "stock", out, trials=3, lock=_lock(repo)
            )
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert info_messages == [
        "wall intrusion: 1 body has no collision geom and uses its visual geoms "
        "(see manifest wall_intrusion.visual_fallback_bodies)"
    ]
    assert m["wall_intrusion"] == {
        "visual_fallback_count": 1,
        "visual_fallback_bodies": ["arm"],
    }
    assert json.loads((out / "manifest.json").read_text())["wall_intrusion"] == m["wall_intrusion"]
    for index in (1, 2, 3):
        trial = json.loads((out / "trials" / f"trial-{index:02d}.json").read_text())
        assert trial["wall_intrusion"]["geometry"]["visual_fallback_bodies"] == ["arm"]
    assert evidence.unresolved_hashes(out) == []

    # Direct CLI subprocess exercises _configure_logging() on stderr and summary on stdout.
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "fallback_shim.py").write_text(
        "from pathlib import Path\n"
        "from test_evidence import FallbackScene, FallbackToy\n"
        "ROBOT = FallbackToy()\n"
        f"ROBOT.root = Path({str(repo)!r})\n"
        "ROBOT.scene = lambda name: FallbackScene()\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim), str(Path(__file__).parent), env.get("PYTHONPATH", "")]
    )
    cli_out = tmp_path / "cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "livingproof_sim_mujoco.evidence",
            "fallback_shim:ROBOT",
            "stock",
            str(cli_out),
            "--lock",
            str(_lock(repo)),
            "--trials",
            "3",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "UserWarning" not in proc.stderr
    assert (
        proc.stderr.count(
            "INFO: wall intrusion: 1 body has no collision geom and uses its visual geoms "
            "(see manifest wall_intrusion.visual_fallback_bodies)"
        )
        == 1
    )
    assert "(1 visual-fallback body)" in proc.stdout


def test_zero_visual_fallback_records_zero_count_and_no_fallback_log(
    tmp_path, monkeypatch, caplog, capsys
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="livingproof_sim_mujoco"):
        code = evidence.main(
            ["test_evidence:TOY", "stock", str(out), "--lock", str(_lock(repo)), "--trials", "2"]
        )
    assert code == 0
    assert not [r for r in caplog.records if "visual" in r.getMessage()]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["wall_intrusion"] == {
        "visual_fallback_count": 0,
        "visual_fallback_bodies": [],
    }
    captured = capsys.readouterr()
    assert "visual-fallback" not in captured.out
