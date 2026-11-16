# SPDX-License-Identifier: Apache-2.0
"""Every sha256 an evidence run declares resolves (capability-verifier#86).

``trajectory_sha256`` names no file: it is the sha256 of the observation's ``trace``
as compact, key-sorted JSON. ``evidence.trajectory_sha256_of`` recomputes it from a
published ``trials/trial-NN.json``. ``evidence.declared_hashes`` lists every hash the
manifests declare and what it covers, and ``evidence.unresolved_hashes`` is empty only
when each one resolves.
"""

import hashlib
import json
from pathlib import Path

import pytest

from livingproof_sim_mujoco import evidence
from test_evidence import TOY, Toy, _lock, _repo  # noqa: F401  (TOY is the adapter)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tests/fixtures/lpv-run-43402628666"


@pytest.mark.parametrize(
    ("trial", "declared"),
    [
        # Values from the evidence manifests of run 43402628666, quoted in the issue.
        (
            "go1-playground-trial-01.json",
            "8856b0a877952ca0b14406f540a9b6bede9bd3e7c3794dbc9c12ad087bdd2f2b",
        ),
        (
            "t1-stock-trial-01.json",
            "61418b3bb0d848227ff6832fc81e5caa6f9d262ebedafadedd91ece7b98d8bc2",
        ),
    ],
)
def test_published_trajectory_hash_recomputes_from_the_trial_file(trial, declared):
    assert evidence.trajectory_sha256_of(RUN / trial) == declared


def _run(tmp_path, monkeypatch, trials=2):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    out = tmp_path / "out"
    evidence.evidence("test_evidence:TOY", "stock", out, trials=trials, lock=_lock(repo))
    return out


def test_every_declared_hash_in_a_written_run_resolves(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    declared = evidence.declared_hashes(out)
    kinds = {(item["file"], item["key"]): item["covers"] for item in declared}
    assert kinds[("manifest.json", "trials.results[0].sha256")] == "artifact"
    assert kinds[("manifest.json", "trials.results[0].trajectory_sha256")] == "trajectory"
    assert kinds[("manifest.json", "robot.profile_sha256")] == "source"
    assert kinds[("run-manifest.json", "world.sha256")] == "source"
    assert evidence.unresolved_hashes(out) == []


def test_a_trajectory_hash_that_does_not_recompute_is_reported(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, trials=1)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["trials"]["results"][0]["trajectory_sha256"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(manifest))
    (problem,) = evidence.unresolved_hashes(out)
    assert "trials.results[0].trajectory_sha256" in problem
    assert "trials/trial-01.json" in problem


def test_an_artifact_hash_that_does_not_match_its_file_is_reported(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, trials=1)
    trial = out / "trials/trial-01.json"
    trial.write_bytes(trial.read_bytes() + b" ")
    problems = evidence.unresolved_hashes(out)
    assert any("trials.results[0].sha256" in p for p in problems)


def test_an_undocumented_hash_field_is_reported(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, trials=1)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["trials"]["results"][0]["video_sha256"] = hashlib.sha256(b"x").hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest))
    (problem,) = evidence.unresolved_hashes(out)
    assert "trials.results[0].video_sha256" in problem and "not documented" in problem


def test_the_source_hashes_match_the_files_they_name(tmp_path, monkeypatch):
    """``source`` hashes name pinned inputs outside the artifact; here they are on disk."""
    out = _run(tmp_path, monkeypatch, trials=1)
    repo = tmp_path / "repo"
    manifest = json.loads((out / "manifest.json").read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert manifest["robot"]["profile_sha256"] == digest(repo / "robot.json")
    configuration = manifest["lock"]["content"]["configurations"][0]
    assert configuration["sha256"] == digest(repo / configuration["path"])


def test_the_evidence_doc_says_how_to_recompute_the_trajectory_hash():
    text = (ROOT / "docs/evidence.md").read_text()
    assert "trajectory_sha256_of" in text and "sort_keys" in text
