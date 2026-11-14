# SPDX-License-Identifier: Apache-2.0
"""Independent-review checks for the caller --lock interface (PR #42).

These turn the reviewer's manual experiments into tests:

* the command line refuses a missing, absent, or malformed --lock and writes nothing;
* the real caller lock shapes (livingproof-verification-lock/0.2.0 and the
  capability-verifier 1.0.0 projection) drive a released v2.0.0 robot end to end
  through ``python -m livingproof_sim_mujoco.evidence`` and record the vendored world;
* a v1 robot adapter is refused with INCOMPATIBLE (no traceback, nothing written);
* composition keeps changes an adapter made to the compiled model, or refuses.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

from livingproof_sim_mujoco import compose, hallway
from livingproof_sim_mujoco.pins import sha256

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/livingproof-world-parity")
VENDORED = ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json"


def _uv_revision(package: str) -> str:
    text = (ROOT / "uv.lock").read_text()
    found = re.search(rf"{re.escape(package)}\?tag=v2\.0\.0#([0-9a-f]{{40}})", text)
    if not found:
        raise AssertionError(f"uv.lock has no v2.0.0 source revision for {package}")
    return found.group(1)


def _head(dest: Path) -> str:
    found = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return found.stdout.strip() if found.returncode == 0 else ""


def _clone(package: str, tag: str) -> Path:
    dest = CACHE / f"{package}-{tag}"
    expected = _uv_revision(package) if tag == "v2.0.0" else None
    if not (dest / "pins.json").is_file() or (expected is not None and _head(dest) != expected):
        subprocess.run(["rm", "-rf", str(dest)], check=True)
        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                tag,
                f"https://github.com/Living-Proof-Labs/{package}.git",
                str(dest),
            ],
            check=True,
        )
    if expected is not None and _head(dest) != expected:
        raise AssertionError(
            f"{package} checkout {_head(dest)} is not the uv.lock revision {expected}"
        )
    return dest


def _shim(directory: Path, checkout: Path, *, use_checkout_src: bool = False) -> str:
    """An adapter module whose ROBOT.root is ``checkout`` (the wheel has no pins/models)."""
    prefix = (
        f"import sys; sys.path.insert(0, {str(checkout / 'src')!r})\n" if use_checkout_src else ""
    )
    (directory / "review_shim.py").write_text(
        prefix
        + "from pathlib import Path\n"
        + "from lpl_g1.adapter import ROBOT\n"
        + f"ROBOT.root = Path({str(checkout)!r})\n"
    )
    return "review_shim:ROBOT"


def _run(directory: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(directory), env.get("PYTHONPATH", "")])
    env.setdefault("MUJOCO_GL", "osmesa")
    return subprocess.run(
        [sys.executable, "-m", "livingproof_sim_mujoco.evidence", *args],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )


def _lock_v020(checkout: Path, configuration: str) -> dict:
    """livingproof-verification/verifications/unitree-g1-playground-mujoco-hallway.lock.json,
    with the configuration digest taken from the robot checkout under test."""
    entry = {
        c["name"]: c for c in json.loads((checkout / "robot.json").read_text())["configurations"]
    }[configuration]
    return {
        "format": "livingproof-verification-lock/0.2.0",
        "procedure": "walk-hallway",
        "scenario": "walk-hallway@0.1.0",
        "robot": {
            "repository": "Living-Proof-Labs/livingproof-unitree-g1",
            "ref": "v2.0.0",
            "configurations": [configuration],
            "adapter": "lpl_g1.adapter:ROBOT",
        },
        "releases": {
            "specs": {
                "repository": "Living-Proof-Labs/robot-capability-specs",
                "tag": "v2.2.0",
                "commit": "23bec516bda29be7b26555286cbf2e8217ac5c81",
            },
            "runner": {
                "repository": "Living-Proof-Labs/livingproof-sim-mujoco",
                "tag": "v2.1.1",
                "commit": "6aa1017adf72ac2886ed4f407e0ac574531744d5",
            },
        },
        # The real lock names livingproof-worlds v1.1.0; the package vendors v1.2.0 (same bytes).
        "world": {
            "id": "hallway-open",
            "version": "0.1.0",
            "repository": "Living-Proof-Labs/livingproof-worlds",
            "tag": "v1.1.0",
            "commit": "9b3e7119aa93a3370383227fb30bd39a9dd72b4b",
        },
        "configurations": [
            {
                "name": configuration,
                "version": entry.get("version", "1.0.0"),
                "path": entry["path"],
                "sha256": sha256(checkout / entry["path"]),
            }
        ],
        "plan": {"trials": {"count": 3}, "command_speed": 0.5},
    }


def _projection(lock: dict) -> dict:
    """capability-verifier evidence_lock.evidence_document: the 1.0.0 projection."""
    keys = ("scenario", "releases", "world", "configurations", "plan")
    return {"format": "livingproof-verification-lock/1.0.0", **{k: lock[k] for k in keys}}


@pytest.fixture(scope="module")
def g1_v2() -> Path:
    return _clone("livingproof-unitree-g1", "v2.0.0")


# --- command line refusals --------------------------------------------------------


def test_missing_lock_is_exit_2_and_writes_nothing(tmp_path, g1_v2):
    ref = _shim(tmp_path, g1_v2)
    out = tmp_path / "out"
    result = _run(tmp_path, ref, "playground", str(out))
    assert result.returncode == 2, result.stderr
    assert "--lock" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    ("make", "shown"),
    [
        (lambda d: d / "absent.json", "is not a file"),
        (lambda d: (d / "bad.json").write_text("{not json") and d / "bad.json", "is not JSON"),
        (lambda d: (d / "list.json").write_text("[]") and d / "list.json", "JSON object"),
    ],
)
def test_unusable_lock_is_exit_1_with_a_teaching_message(tmp_path, g1_v2, make, shown):
    ref = _shim(tmp_path, g1_v2)
    out = tmp_path / "out"
    lock = make(tmp_path)
    result = _run(tmp_path, ref, "playground", str(out), "--lock", str(lock))
    assert result.returncode == 1, result.stdout + result.stderr
    assert shown in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("missing", ["world", "releases", "configurations", "plan"])
def test_lock_missing_a_required_field_is_named_not_a_traceback(tmp_path, g1_v2, missing):
    """A caller lock without a field this command reads: exit 1, the field is named."""
    ref = _shim(tmp_path, g1_v2)
    lock = _lock_v020(g1_v2, "playground")
    del lock[missing]
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    out = tmp_path / "out"
    result = _run(tmp_path, ref, "playground", str(out), "--lock", str(path), "--trials", "1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr[-600:]
    assert missing in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    ("world", "shown"),
    [
        ({"id": "hallway-open", "version": "0.2.0"}, "UNSUPPORTED"),
        ({"id": "hallway-open", "version": "0.1.0", "sha256": "ab" * 32}, "world.sha256"),
    ],
)
def test_unvendored_or_mismatched_world_via_cli_writes_nothing(tmp_path, g1_v2, world, shown):
    ref = _shim(tmp_path, g1_v2)
    lock = _lock_v020(g1_v2, "playground")
    lock["world"] = world
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    out = tmp_path / "out"
    result = _run(tmp_path, ref, "playground", str(out), "--lock", str(path), "--trials", "1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert shown in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


# --- real lock shapes end to end -------------------------------------------------


@pytest.mark.parametrize("shape", ["0.2.0", "1.0.0-projection"])
def test_real_caller_lock_drives_a_v2_robot_in_the_vendored_hallway(tmp_path, g1_v2, shape):
    ref = _shim(tmp_path, g1_v2)
    lock = _lock_v020(g1_v2, "playground")
    if shape != "0.2.0":
        lock = _projection(lock)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    out = tmp_path / "out"
    result = _run(tmp_path, ref, "playground", str(out), "--lock", str(path), "--trials", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    run = json.loads((out / "run-manifest.json").read_text())
    assert run["world"] == {"id": "hallway-open", "version": "0.1.0", "sha256": sha256(VENDORED)}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["lock"]["content"] == lock
    assert manifest["environment"]["walls"] == ["wall-left", "wall-right"]
    trial = json.loads((out / "trials" / "trial-01.json").read_text())
    first = trial["trace"]["samples"][0]
    assert first[1] == pytest.approx(-2.5, abs=1e-6)  # spawn-entrance x, not the origin
    assert first[2] == pytest.approx(0.0, abs=1e-6)


# --- v1 robots are refused, not crashed ------------------------------------------


def test_v1_robot_is_refused_with_incompatible_and_writes_nothing(tmp_path):
    checkout = _clone("livingproof-unitree-g1", "v1.0.4")
    ref = _shim(tmp_path, checkout, use_checkout_src=True)
    entry = {
        c["name"]: c for c in json.loads((checkout / "robot.json").read_text())["configurations"]
    }["playground"]
    lock = _lock_v020(checkout, "playground")
    lock["configurations"][0].update(path=entry["path"], sha256=sha256(checkout / entry["path"]))
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    out = tmp_path / "out"
    result = _run(tmp_path, ref, "playground", str(out), "--lock", str(path), "--trials", "1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "INCOMPATIBLE" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


# --- composition must not silently drop adapter changes --------------------------


class _Scene:
    def __init__(self, model):
        self.model = model


class _Spec:
    actuation = "pd_torque"


class _Ctrl:
    def __init__(self, model):
        self.initial_qpos = model.qpos0.copy()
        self.spec = _Spec()


class _PostCompileAdapter:
    """Compiles its spec, then edits the compiled model (a common MuJoCo idiom)."""

    id = "review-toy"

    def scene(self, name):
        spec = mujoco.MjSpec.from_string(
            '<mujoco><option timestep="0.002"/><worldbody><body name="base" pos="0 0 0.3">'
            '<freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        model = spec.compile()
        model.opt.timestep = 0.001  # set after compile, as reference loaders do
        model.geom_friction[0, 0] = 0.3
        return _Scene(model)

    def controller(self, name, scene):
        return _Ctrl(scene.model)

    def base_body(self, name):
        return "base"

    def initial_state(self, model, data, ctrl, spawn, *, start_offset=None):
        from livingproof_sim_mujoco.walk import initial_state

        initial_state(model, data, ctrl, spawn, start_offset=start_offset)


def test_compose_keeps_or_refuses_post_compile_model_changes():
    world = hallway.load_fixture("hallway-open", "0.1.0")["world"]
    robot = _PostCompileAdapter()
    try:
        model, _data, _ctrl, _walls = compose.place(robot, "x", world, "spawn-entrance")
    except SystemExit as exc:  # refusing is acceptable; silently changing physics is not
        assert "INCOMPATIBLE" in str(exc)
        return
    assert float(model.opt.timestep) == pytest.approx(0.001)
    assert float(model.geom_friction[0, 0]) == pytest.approx(0.3)
