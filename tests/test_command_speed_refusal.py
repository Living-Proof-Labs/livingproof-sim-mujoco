# SPDX-License-Identifier: Apache-2.0
"""Gaps in the refusal of a lock speed the run does not drive (Vera, verification of PR 154).

The refusal has to happen before any output exists, so a refused run leaves no output directory,
not only no manifest. It has to be exact, so a number that is merely close, or a string or boolean
that only looks equal, is refused too. These are my own tests; the mutation rerun is the check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evidence import CALLER_LOCK, Toy, _lock, _repo

from livingproof_sim_mujoco import evidence


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    out = tmp_path / "ev"

    def go(lock_speed):
        lock = json.loads(_lock(repo).read_text())
        lock["plan"]["command_speed"] = lock_speed
        (repo / CALLER_LOCK).write_text(json.dumps(lock))
        evidence.evidence("test_evidence:TOY", "stock", out, trials=1, lock=_lock(repo))

    go.out = out
    return go


@pytest.mark.parametrize("speed", [0.25, 2.0])
def test_a_refused_lock_speed_leaves_no_output_directory(attempt, speed) -> None:
    with pytest.raises(SystemExit):
        attempt(speed)
    assert not attempt.out.exists()


@pytest.mark.parametrize("speed", [0.5001, 0.4999999, 0.5000001, "0.5", True, 1, 0])
def test_only_the_exact_speed_is_accepted(attempt, speed) -> None:
    with pytest.raises(SystemExit) as refused:
        attempt(speed)
    assert "command_speed" in str(refused.value)
    assert not attempt.out.exists()
