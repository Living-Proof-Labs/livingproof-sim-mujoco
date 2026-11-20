# SPDX-License-Identifier: Apache-2.0
"""The speed in the manifest is the speed the simulator commanded (Vera, verification of PR 154).

The scenario resolves ``time_limit`` from ``plan.command_speed`` in the manifest alone, so that
number has to be the one the run drove at. The tests record the command each policy step receives
and compare it with the manifest, with and without a speed in the lock's plan. The frozen hallway runs only the constant command
0.5 m/s, so the parameter file's speed cannot be varied and a recorded constant 0.5 is not told
apart from the parameter file's here.

One case is a strict xfail: the simulator takes its speed from the parameter file only, never from
the lock's plan, so a lock that lists a different speed is recorded as given but not driven.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evidence import CALLER_LOCK, Push, Toy, _lock, _repo

from livingproof_sim_mujoco import evidence


@pytest.fixture
def drive(tmp_path, monkeypatch):
    commanded: list[float] = []
    real = Push.step

    def recording(self, data, command):
        commanded.append(command[0])
        return real(self, data, command)

    monkeypatch.setattr(Push, "step", recording)
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    def go(*, lock_speed: float | None):
        lock = json.loads(_lock(repo).read_text())
        if lock_speed is None:
            lock["plan"].pop("command_speed")
        else:
            lock["plan"]["command_speed"] = lock_speed
        (repo / CALLER_LOCK).write_text(json.dumps(lock))
        commanded.clear()
        out = tmp_path / "ev"
        evidence.evidence("test_evidence:TOY", "stock", out, trials=1, lock=_lock(repo))
        return json.loads((out / "manifest.json").read_text()), set(commanded)

    return go


def test_a_lock_without_a_speed_records_the_speed_the_parameter_file_drove(drive) -> None:
    manifest, commanded = drive(lock_speed=None)
    assert commanded == {0.5}
    assert manifest["plan"]["command_speed"] == 0.5


def test_a_lock_speed_equal_to_the_parameter_files_is_recorded_unchanged(drive) -> None:
    manifest, commanded = drive(lock_speed=0.5)
    assert commanded == {0.5}
    assert manifest["plan"]["command_speed"] == 0.5


@pytest.mark.xfail(
    strict=True,
    reason="the simulator ignores plan.command_speed in the lock; the manifest records it as given",
)
def test_a_lock_speed_that_is_not_the_parameter_files_is_not_recorded_as_driven(drive) -> None:
    manifest, commanded = drive(lock_speed=0.25)
    assert commanded == {0.5}  # what the policy was given
    assert manifest["plan"]["command_speed"] == 0.5  # recorded 0.25 today
