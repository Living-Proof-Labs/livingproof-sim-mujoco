# SPDX-License-Identifier: Apache-2.0
"""The simulator, not the robot package, supplies walls and the start place."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import pytest

from livingproof_sim_mujoco import compose, hallway

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "livingproof_sim_mujoco"


def test_package_does_not_read_a_robot_lock_or_scene_walls():
    """Nothing in this package reads the robot repository's lock or scene walls."""
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        assert "verification.lock.json" not in text, path
        assert "scene.walls" not in text, path
        assert re.search(r"\.scene\([^)]*\)\.walls", text) is None, path


def test_missing_walls_are_incompatible():
    xml = """<mujoco><worldbody><body name="base" pos="0 0 0.3"><freejoint/>
    <geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = {
        "geometry": [
            {
                "name": "wall-left",
                "role": "wall",
                "type": "box",
                "centre": [0, 0.8, 1],
                "size": [1, 0.1, 2],
            }
        ],
        "places": [
            {"name": "spawn-entrance", "type": "pose", "position": [-2.5, 0.0, 0.0], "yaw_deg": 0}
        ],
    }
    with pytest.raises(SystemExit, match="INCOMPATIBLE") as caught:
        compose.require_composed(model, data, world, "spawn-entrance", "base")
    assert "wall-left" in str(caught.value)


def test_origin_start_is_incompatible_when_the_place_is_not():
    xml = """<mujoco><worldbody>
    <geom name="wall-left" type="box" pos="0 0.8 1" size="0.5 0.05 1"/>
    <geom name="wall-right" type="box" pos="0 -0.8 1" size="0.5 0.05 1"/>
    <body name="base" pos="0 0 0.3"><freejoint/><geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert abs(float(data.xpos[model.body("base").id][0])) < 1e-6
    world = hallway.load_fixture("hallway-open", "0.1.0")["world"]
    with pytest.raises(SystemExit, match="INCOMPATIBLE") as caught:
        compose.require_composed(model, data, world, "spawn-entrance", "base")
    assert "origin" in str(caught.value)


def test_unknown_world_is_unsupported():
    with pytest.raises(SystemExit, match="UNSUPPORTED"):
        compose.vendored_world({"world": {"id": "not-a-world", "version": "0.1.0"}})


def _uv_revision(package: str) -> str:
    """Commit uv.lock records for the v2.0.0 tag of a robot package."""
    text = (ROOT / "uv.lock").read_text()
    found = re.search(rf"{re.escape(package)}\?tag=v2\.0\.0#([0-9a-f]{{40}})", text)
    if not found:
        raise AssertionError(f"uv.lock has no v2.0.0 source revision for {package}")
    return found.group(1)


def _checkout(package: str) -> Path:
    """Source tree of the v2.0.0 tag. The installed wheel does not ship pins or models.

    The checkout commit has to be the revision ``uv.lock`` records for that tag.
    """
    import shutil
    import subprocess

    dest = Path("/tmp/livingproof-world-parity") / f"{package}-v2.0.0"
    expected = _uv_revision(package)

    def head() -> str:
        if not dest.exists():
            return ""
        found = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return found.stdout.strip() if found.returncode == 0 else ""

    if not (dest / "pins.json").is_file() or head() != expected:
        if dest.exists():
            shutil.rmtree(dest)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v2.0.0",
                f"https://github.com/Living-Proof-Labs/{package}.git",
                str(dest),
            ],
            check=True,
        )
    got = head()
    if got != expected:
        raise AssertionError(f"{package} checkout {got} is not the uv.lock revision {expected}")
    return dest


@pytest.mark.parametrize(
    ("adapter", "package"),
    [
        ("lpl_g1.adapter:ROBOT", "livingproof-unitree-g1"),
        ("lpl_go1.adapter:ROBOT", "livingproof-unitree-go1"),
        ("lpl_t1.adapter:ROBOT", "livingproof-booster-t1"),
    ],
)
def test_v2_robot_starts_at_the_world_place_between_the_walls(adapter, package):
    """Each v2.0.0 robot: hallway walls are in the model and the base is not at the origin."""
    from livingproof_sim_mujoco.evidence import adapter as load
    from livingproof_sim_mujoco.pins import fetch

    robot = load(adapter)
    robot.root = _checkout(package)
    fetch(robot.root)
    world, _digest, _path = compose.vendored_world(
        {"world": {"id": "hallway-open", "version": "0.1.0"}}
    )
    place = next(p for p in world["places"] if p["name"] == "spawn-entrance")
    for offered in robot.robot()["configurations"]:
        name = offered["name"]
        model, data, _ctrl, walls = compose.place(robot, name, world, "spawn-entrance", None)
        assert walls == ["wall-left", "wall-right"]
        for wall_name in walls:
            geom = model.geom(wall_name)
            assert int(model.geom_contype[geom.id]) == 0
            assert int(model.geom_conaffinity[geom.id]) == 0
        base = data.xpos[model.body(robot.base_body(name)).id]
        assert base[0] == pytest.approx(place["position"][0])
        assert base[1] == pytest.approx(place["position"][1])
        assert abs(float(base[0])) > 1.0  # spawn-entrance is not the origin


def test_world_tag_or_commit_that_is_not_the_vendored_file_is_an_error():
    """A tag or commit that is not a release of the vendored file exits like world.sha256."""
    base = {"id": "hallway-open", "version": "0.1.0"}
    with pytest.raises(SystemExit, match="world.tag") as caught:
        compose.vendored_world({"world": {**base, "tag": "v9.9.9"}})
    assert "Traceback" not in str(caught.value)
    with pytest.raises(SystemExit, match="world.commit") as caught:
        compose.vendored_world({"world": {**base, "commit": "ab" * 20}})
    assert "Traceback" not in str(caught.value)
    # The published locks name v1.1.0. That world.json is the vendored file.
    world, digest, _path = compose.vendored_world(
        {
            "world": {
                **base,
                "tag": "v1.1.0",
                "commit": "9b3e7119aa93a3370383227fb30bd39a9dd72b4b",
            }
        }
    )
    assert world["id"] == "hallway-open"
    assert digest == "d60bb8aa6a72c8403fb6674cc9d803962c1484bc3e163527988e8c57ccec775c"


def test_compose_restores_compile_and_ignores_another_thread():
    """Composition does not leave MjSpec.compile replaced, and another thread's compile is its own."""
    import threading

    original = mujoco.MjSpec.compile
    world = hallway.load_fixture("hallway-open", "0.1.0")["world"]

    class _Scene:
        def __init__(self, model):
            self.model = model

    class _Ctrl:
        def __init__(self, model):
            self.initial_qpos = model.qpos0.copy()
            self.spec = type("Spec", (), {"actuation": "pd_torque"})()

    class _Robot:
        id = "review-plain"

        def scene(self, name):
            spec = mujoco.MjSpec.from_string(
                '<mujoco><option timestep="0.002"/><worldbody><body name="base" pos="0 0 0.3">'
                '<freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
            )
            return _Scene(spec.compile())

        def controller(self, name, scene):
            return _Ctrl(scene.model)

        def base_body(self, name):
            return "base"

        def initial_state(self, model, data, ctrl, spawn, *, start_offset=None):
            from livingproof_sim_mujoco.walk import initial_state

            initial_state(model, data, ctrl, spawn, start_offset=start_offset)

    errors = []
    stop = threading.Event()

    def other():
        spec = mujoco.MjSpec.from_string(
            '<mujoco><option timestep="0.007"/><worldbody><body name="base" pos="0 0 0.2">'
            '<freejoint/><geom type="sphere" size="0.05"/></body></worldbody></mujoco>'
        )
        while not stop.is_set():
            model = spec.compile()
            if float(model.opt.timestep) != pytest.approx(0.007):
                errors.append(float(model.opt.timestep))
                return

    thread = threading.Thread(target=other)
    thread.start()
    try:
        model, _data, _ctrl, walls = compose.place(_Robot(), "x", world, "spawn-entrance")
    finally:
        stop.set()
        thread.join(timeout=5)
    assert mujoco.MjSpec.compile is original
    assert errors == []
    assert thread.is_alive() is False
    assert walls == ["wall-left", "wall-right"]
    assert float(model.opt.timestep) == pytest.approx(0.002)
