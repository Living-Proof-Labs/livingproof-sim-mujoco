"""MCAP recordings and screencasts of walk-hallway trials (toy robot from test_evidence)."""

import json
from pathlib import Path

import pytest

from livingproof_sim_mujoco import evidence, media

pytest.importorskip("mcap")
pytest.importorskip("imageio_ffmpeg")

from test_evidence import Toy, _repo  # noqa: E402


def _verification(statuses):
    return {
        "trials": [
            {"seed": s, "status": st, "evidence": {"sha256": "x"}} for s, st in statuses.items()
        ]
    }


def test_pick_first_pass_and_first_fail_by_seed():
    assert media.pick(_verification({3: "FAIL", 2: "PASS", 1: "FAIL", 4: "PASS"})) == {
        2: "PASS",
        1: "FAIL",
    }
    assert media.pick(_verification({1: "PASS", 2: "PASS"})) == {1: "PASS"}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", jobs=1)
    v = tmp_path / "verification.json"
    v.write_text(json.dumps(_verification({1: "PASS", 2: "FAIL"})))
    return tmp_path, v


def test_every_seed_gets_an_mcap_and_only_picked_seeds_a_video(recorded):
    from mcap.reader import make_reader

    tmp_path, v = recorded
    doc = media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m", jobs=1)
    out = tmp_path / "m"
    assert doc["format"] == media.FORMAT
    manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    for t, e in zip(doc["trials"], manifest["trials"]):
        assert t["observation_sha256"] == e["sha256"]  # the recording is of the graded trial
        assert (out / t["mcap"]["path"]).stat().st_size == t["mcap"]["bytes"]
    assert [(t["seed"], t.get("status"), t["video"]["path"]) for t in doc["trials"]] == [
        (1, "PASS", "video/seed-01-pass.mp4"),
        (2, "FAIL", "video/seed-02-fail.mp4"),
    ]
    with (out / "mcap" / "seed-01.mcap").open("rb") as f:
        topics = {c.topic for _, c, _ in make_reader(f).iter_messages()}
    assert topics == {"/base_pose", "/joint_states"}
    assert json.loads((out / "media.json").read_text()) == doc


def test_a_rerun_that_differs_from_the_evidence_keeps_nothing(recorded):
    tmp_path, v = recorded
    m = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    m["trials"][0]["sha256"] = "0" * 64
    (tmp_path / "ev" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="differs from the evidence"):
        media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m", jobs=1)
    assert not (tmp_path / "m" / "mcap" / "seed-01.mcap").exists()
