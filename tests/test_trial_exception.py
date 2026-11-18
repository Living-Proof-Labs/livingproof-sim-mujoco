# SPDX-License-Identifier: Apache-2.0
"""A trial that raises an exception still writes manifest.json and exits non-zero (#105)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from livingproof_sim_mujoco import evidence
from test_evidence import CALLER_LOCK, Toy, _lock, _repo

ROOT = Path(__file__).resolve().parents[1]


def _setup_toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return repo


def test_step_exception_on_second_trial_writes_manifest_and_exits_nonzero(tmp_path, monkeypatch):
    repo = _setup_toy(tmp_path, monkeypatch)
    real_policy = Toy.policy
    attempts = {"count": 0}

    def failing_policy(self, model, ctrl):
        inner = real_policy(self, model, ctrl)
        attempts["count"] += 1
        trial_number = attempts["count"]

        class StepSometimes:
            def __init__(self):
                self.steps = 0

            def step(self, data, command):
                self.steps += 1
                if trial_number == 2 and self.steps == 3:
                    raise RuntimeError("controller step exploded")
                return inner.step(data, command)

        return StepSometimes()

    monkeypatch.setattr(Toy, "policy", failing_policy)
    out = tmp_path / "out"
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(out),
            "--lock",
            str(_lock(repo)),
            "--trials",
            "2",
        ]
    )
    assert code != 0
    assert code != evidence.EXIT_REPRODUCIBILITY
    assert code == evidence.EXIT_TRIAL_EXCEPTION

    assert (out / "run-manifest.json").is_file()
    manifest_path = out / "manifest.json"
    assert manifest_path.is_file()
    assert (out / "trials" / "trial-01.json").is_file()
    assert not (out / "trials" / "trial-02.json").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["trials"]["count"] == 2
    results = manifest["trials"]["results"]
    assert len(results) == 2

    first, second = results
    assert first["index"] == 1
    assert first["path"] == "trials/trial-01.json"
    assert first["outcome"] == "crossed_exit"
    assert isinstance(first["sha256"], str) and len(first["sha256"]) == 64
    assert isinstance(first["trajectory_sha256"], str) and len(first["trajectory_sha256"]) == 64
    assert "exception" not in first

    assert second["index"] == 2
    assert second["exception"] == {
        "type": "RuntimeError",
        "message": "controller step exploded",
    }
    assert isinstance(second["wall_clock_s"], (int, float)) and second["wall_clock_s"] >= 0
    for forbidden in ("outcome", "path", "sha256", "trajectory_sha256", "status"):
        assert forbidden not in second

    assert manifest["trials"]["reproducibility"] == {
        "compared": False,
        "identical": None,
        "finding": None,
    }
    assert evidence.unresolved_hashes(out) == []


def test_exception_stops_before_later_trials(tmp_path, monkeypatch):
    repo = _setup_toy(tmp_path, monkeypatch)
    calls = {"count": 0}

    def broken_policy(self, model, ctrl):
        calls["count"] += 1
        raise ValueError("policy init failed")

    monkeypatch.setattr(Toy, "policy", broken_policy)
    out = tmp_path / "out"
    manifest = evidence.evidence(
        "test_evidence:TOY", "stock", out, trials=3, lock=repo / CALLER_LOCK
    )
    assert calls["count"] == 1
    assert manifest["trials"]["count"] == 3
    assert len(manifest["trials"]["results"]) == 1
    entry = manifest["trials"]["results"][0]
    assert entry["index"] == 1
    assert entry["exception"] == {"type": "ValueError", "message": "policy init failed"}
    assert "outcome" not in entry
    assert manifest["trials"]["reproducibility"] == {
        "compared": False,
        "identical": None,
        "finding": None,
    }
    assert evidence.unresolved_hashes(out) == []


class _AdapterCustomError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc_factory", "expected_type", "expected_sub"),
    [
        (
            lambda: _AdapterCustomError("line1\nline2  sha256 " + "x" * 400),
            f"{__name__}._AdapterCustomError",
            "line1 line2 sha256 ",
        ),
        (
            lambda: RuntimeError(),
            "RuntimeError",
            "RuntimeError",
        ),
    ],
)
def test_exception_type_and_short_single_line_message(
    tmp_path, monkeypatch, exc_factory, expected_type, expected_sub
):
    repo = _setup_toy(tmp_path, monkeypatch)

    def raise_policy(self, model, ctrl):
        raise exc_factory()

    monkeypatch.setattr(Toy, "policy", raise_policy)
    out = tmp_path / "out"
    manifest = evidence.evidence(
        "test_evidence:TOY", "stock", out, trials=1, lock=repo / CALLER_LOCK
    )
    recorded = manifest["trials"]["results"][0]["exception"]
    assert recorded["type"] == expected_type
    assert "\n" not in recorded["message"]
    assert 0 < len(recorded["message"]) <= 240
    assert expected_sub in recorded["message"]
    assert evidence.unresolved_hashes(out) == []


def test_disagreeing_completed_trials_followed_by_exception_do_not_exit_4(tmp_path, monkeypatch):
    repo = _setup_toy(tmp_path, monkeypatch)
    real_run_trial = evidence.run_trial
    calls = {"n": 0}

    def vary_then_crash(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("third trial crashed")
        obs = real_run_trial(*args, **kwargs)
        if calls["n"] == 2:
            obs = {**obs, "end": "fell"}
        return obs

    monkeypatch.setattr(evidence, "run_trial", vary_then_crash)
    out = tmp_path / "out"
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(out),
            "--lock",
            str(_lock(repo)),
            "--trials",
            "3",
        ]
    )
    assert code == evidence.EXIT_TRIAL_EXCEPTION
    assert code != evidence.EXIT_REPRODUCIBILITY
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["trials"]["results"]) == 3
    assert manifest["trials"]["results"][2]["exception"]["type"] == "RuntimeError"
    assert manifest["trials"]["reproducibility"]["compared"] is True
    assert manifest["trials"]["reproducibility"]["identical"] is False


def test_evidence_doc_documents_trial_exception_record_and_exit_code():
    text = (ROOT / "docs/evidence.md").read_text(encoding="utf-8")
    assert f"Exit {evidence.EXIT_TRIAL_EXCEPTION}" in text
    assert "trials.results[]" in text
    assert "exception" in text
