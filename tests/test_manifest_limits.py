# SPDX-License-Identifier: Apache-2.0
"""The run manifest records every input the scenario's limits are resolved from (#153).

livingproof-evaluator-scenario (PR 31) takes each limit from the run manifest only. A limit it
cannot resolve leaves its step NOT_EVALUATED (``missing_threshold``). These tests read the manifest
a real ``evidence()`` run writes and resolve the limits from it, with the real runner when it is
importable and with the formulas as stated in the issue otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_evidence import Toy, _lock, _repo  # noqa: E402

from livingproof_sim_mujoco import evidence  # noqa: E402

# limit -> (formula over the manifest inputs, the inputs it reads), from the scenario parameter file
# (walk-hallway, robot-capability-specs v2.2.0); rounded to 4 decimals, half-even, except fall_height.
LIMITS = {
    "time_limit": ("2 * route_length / command_speed + 5", ("route_length", "command_speed")),
    "lateral_limit": ("clear_width / 2 - robot_width / 2", ("clear_width", "robot_width")),
    "half_width": ("clear_width / 2", ("clear_width",)),
    "fall_height": ("fall_base_height", ("fall_base_height",)),
}


def _number(value):
    return value["value"] if isinstance(value, dict) else value


def inputs(manifest: dict) -> dict:
    """Each input the limits read, taken from where the scenario reads it. Absent => missing."""
    plan = manifest.get("plan") or {}
    world = manifest.get("world") or {}
    robot = (manifest.get("robot") or {}).get("profile") or {}
    config = (manifest.get("configuration") or {}).get("profile") or {}
    found = {
        "command_speed": plan.get("command_speed"),
        "route_length": world.get("route_length"),
        "clear_width": world.get("clear_width"),
        "robot_width": (robot.get("dimensions") or {}).get("width"),
        "fall_base_height": (config.get("locomotion") or {}).get("fall_base_height"),
    }
    return {k: _number(v) for k, v in found.items() if v is not None}


SCENARIO_REPO = "https://github.com/Living-Proof-Labs/livingproof-evaluator-scenario"
SCENARIO_TAG, SCENARIO_COMMIT = "v1.5.0", "3fab127372208b92c532bbd880aa10ae275f7565"
_THRESHOLDS = (
    "import json, sys\n"
    "from livingproof_evaluator_scenario import media\n"
    "print(json.dumps(media.thresholds_from_manifest(json.load(sys.stdin))))\n"
)
_SOURCE: list[Path] = []


def _scenario_source() -> Path:
    """The released scenario evaluator's source, fetched once at its tag and checked by commit."""
    if not _SOURCE:
        import tempfile

        target = Path(tempfile.mkdtemp(prefix="scenario-")) / "src"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "--branch", SCENARIO_TAG, SCENARIO_REPO, str(target)],
            check=True, capture_output=True, timeout=300,
        )  # fmt: skip
        head = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert head == SCENARIO_COMMIT, (
            f"{SCENARIO_TAG} is {head}, not the pinned {SCENARIO_COMMIT}"
        )
        _SOURCE.append(target)
    return _SOURCE[0]


class _FromSource:
    """The released scenario runner (v1.5.0, pinned by commit), run in a subprocess.

    Its package metadata says Python 3.12 and this project supports 3.11 and keeps its
    dependencies minimal, so it is not a dependency of the project: its source is fetched at the
    tag and run with this environment's Python and the packages it already has (mcap, the
    scenario parameter library).
    """

    @staticmethod
    def thresholds_from_manifest(manifest: dict):
        env = {**os.environ, "PYTHONPATH": str(_scenario_source() / "src")}
        done = subprocess.run(
            [sys.executable, "-c", _THRESHOLDS],
            input=json.dumps(manifest), capture_output=True, text=True, timeout=300, env=env,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr[-2000:]
        thresholds, missing = json.loads(done.stdout.splitlines()[-1])
        return thresholds, missing


def scenario_media():
    """The real scenario runner: imported when installed, else fetched at its release tag.

    With REQUIRE_SCENARIO=1 (CI), failing to get it is a failure, not a skip.
    """
    try:
        from livingproof_evaluator_scenario import media

        return media
    except ImportError:
        pass
    if shutil.which("git") is not None:
        return _FromSource
    if os.environ.get("REQUIRE_SCENARIO") == "1":
        pytest.fail("REQUIRE_SCENARIO=1 but livingproof-evaluator-scenario is not available")
    pytest.skip("livingproof-evaluator-scenario is not available")


@pytest.fixture
def run(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    def go(mutate=None):
        lock = json.loads(_lock(repo).read_text())
        if mutate:
            mutate(lock)
        _lock(repo).write_text(json.dumps(lock))
        out = tmp_path / "ev"
        evidence.evidence("test_evidence:TOY", "stock", out, trials=1, lock=_lock(repo))
        return json.loads((out / "manifest.json").read_text())

    return go


def test_every_input_a_limit_reads_is_in_the_manifest(run):
    manifest = run()
    assert set(inputs(manifest)) == {
        "command_speed",
        "route_length",
        "clear_width",
        "robot_width",
        "fall_base_height",
    }
    needed = {name for _, names in LIMITS.values() for name in names}
    assert needed <= set(inputs(manifest))


def test_the_command_speed_is_recorded_even_when_the_lock_plan_does_not_list_it(run):
    def drop(lock):
        del lock["plan"]["command_speed"]

    manifest = run(drop)
    assert manifest["plan"]["command_speed"] == 0.5  # the speed the run used (parameter file)


def test_a_lock_speed_equal_to_the_parameter_files_passes(run):
    def same(lock):
        lock["plan"]["command_speed"] = 0.5

    assert run(same)["plan"]["command_speed"] == 0.5


def test_a_lock_speed_that_differs_from_the_parameter_files_is_refused_before_the_run(run):
    def faster(lock):
        lock["plan"]["command_speed"] = 0.25

    with pytest.raises(SystemExit, match="command_speed"):
        run(faster)


def test_a_missing_lock_speed_records_the_parameter_file_speed(run):
    def drop(lock):
        del lock["plan"]["command_speed"]

    assert run(drop)["plan"]["command_speed"] == 0.5


def test_time_limit_follows_the_speed_the_run_used(run):
    manifest = run()
    speed = manifest["plan"]["command_speed"]
    assert 2 * manifest["world"]["route_length"] / speed + 5 == 27.0


def test_the_limits_resolve_from_the_manifest_alone(run):
    manifest = run()
    known = inputs(manifest)
    assert 2 * known["route_length"] / known["command_speed"] + 5 == 27.0
    assert known["clear_width"] / 2 == 0.75


def test_the_real_runner_resolves_every_limit_with_none_missing(run):
    media = scenario_media()
    thresholds, missing = media.thresholds_from_manifest(run())
    assert missing == {}
    assert set(thresholds) == set(LIMITS)
    assert thresholds["fall_height"] == 0.1 and thresholds["time_limit"] == 27.0


def test_the_real_runner_resolves_them_when_the_lock_plan_has_no_speed(run):
    media = scenario_media()

    def drop(lock):
        del lock["plan"]["command_speed"]

    thresholds, missing = media.thresholds_from_manifest(run(drop))
    assert missing == {} and thresholds["time_limit"] == 27.0


def test_the_real_runner_is_required_when_the_switch_is_set(monkeypatch):
    monkeypatch.setitem(sys.modules, "livingproof_evaluator_scenario.media", None)
    monkeypatch.delitem(sys.modules, "livingproof_evaluator_scenario", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("REQUIRE_SCENARIO", "1")
    with pytest.raises(pytest.fail.Exception, match="REQUIRE_SCENARIO=1"):
        scenario_media()
    monkeypatch.delenv("REQUIRE_SCENARIO")
    with pytest.raises(pytest.skip.Exception):
        scenario_media()
