# SPDX-License-Identifier: Apache-2.0
"""``unresolved_hashes`` on what robots actually write (Vera review of #74).

The fixtures are real walk-corner runs from released robot tags (PROVENANCE.json):
every declared sha256 must be classified, and every artifact and trajectory hash
must still recompute.
"""

import json
import shutil
from pathlib import Path

import pytest

from livingproof_sim_mujoco import evidence

REAL = Path(__file__).parent / "fixtures" / "real-runs"
RUNS = ("g1-stock", "t1-stock", "t1-playground", "go1-playground")


@pytest.mark.parametrize("run", RUNS)
def test_a_real_run_has_no_unresolved_hash(run):
    assert evidence.unresolved_hashes(REAL / run) == []


@pytest.mark.parametrize("run", RUNS)
def test_profile_hashes_of_a_real_run_are_sources(run):
    profile = [
        item
        for item in evidence.declared_hashes(REAL / run)
        if item["key"].startswith(("robot.profile.", "configuration.profile."))
    ]
    assert profile, "a real robot profile cites pinned files by sha256"
    assert {item["covers"] for item in profile} == {"source"}


def test_a_real_run_still_catches_a_changed_trial(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(REAL / "g1-stock", run)
    trial = run / "trials" / "trial-01.json"
    trial.write_text(trial.read_text().replace('"end"', '"end" ', 1))
    assert any("trials.results[0].sha256" in p for p in evidence.unresolved_hashes(run))


def test_lock_source_hashes_are_sources(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(REAL / "g1-stock", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["lock"]["content"]["sources"] = {
        "parameters": {"path": "parameters.json", "sha256": "ab" * 32}
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    [item] = [
        i for i in evidence.declared_hashes(run) if i["key"].startswith("lock.content.sources.")
    ]
    assert item["key"] == "lock.content.sources.parameters.sha256"
    assert item["covers"] == "source"
    assert evidence.unresolved_hashes(run) == []


def test_an_unlisted_hash_outside_those_prefixes_is_still_reported(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(REAL / "g1-stock", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["environment"]["stray_sha256"] = "ab" * 32
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert any(
        "environment.stray_sha256 is not documented" in p for p in evidence.unresolved_hashes(run)
    )
