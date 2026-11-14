"""MCAP recordings and screencasts of walk-hallway trials (toy robot from test_evidence)."""

import json
from pathlib import Path

import pytest

from livingproof_sim_mujoco import evidence, media

pytest.importorskip("mcap")
pytest.importorskip("imageio_ffmpeg")

from test_evidence import Toy, _lock, _repo  # noqa: E402


def _verification(statuses):
    return {
        "trials": [
            {"index": s, "status": st, "evidence": {"sha256": "x"}} for s, st in statuses.items()
        ]
    }


def test_pick_first_pass_and_first_fail_by_index():
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
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", trials=2, lock=_lock(repo))
    v = tmp_path / "verification.json"
    v.write_text(json.dumps(_verification({1: "PASS", 2: "FAIL"})))
    return tmp_path, v


def test_every_trial_gets_an_mcap_and_only_picked_trials_a_video(recorded):
    from mcap.reader import make_reader

    tmp_path, v = recorded
    doc = media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")
    out = tmp_path / "m"
    assert doc["format"] == media.FORMAT == "livingproof-hallway-media/2.0.0"
    manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    for t, e in zip(doc["trials"], manifest["trials"]["results"]):
        assert t["observation_sha256"] == e["sha256"]  # the recording is of the graded trial
        assert (out / t["mcap"]["path"]).stat().st_size == t["mcap"]["bytes"]
    assert [(t["index"], t.get("status"), t["video"]["path"]) for t in doc["trials"]] == [
        (1, "PASS", "video/trial-01-pass.mp4"),
        (2, "FAIL", "video/trial-02-fail.mp4"),
    ]
    with (out / "mcap" / "trial-01.mcap").open("rb") as f:
        topics = {c.topic for _, c, _ in make_reader(f).iter_messages()}
    assert topics == {"/base_pose", "/joint_states"}
    assert json.loads((out / "media.json").read_text()) == doc


def test_a_rerun_that_differs_from_the_evidence_keeps_nothing(recorded):
    tmp_path, v = recorded
    m = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    m["trials"]["results"][0]["sha256"] = "0" * 64
    (tmp_path / "ev" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="differs from the evidence"):
        media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")
    assert not (tmp_path / "m" / "mcap").exists()  # no partial files from any trial
    assert not (tmp_path / "m" / ".media-partial").exists()


def test_a_rerun_into_an_existing_folder_replaces_its_media(recorded):
    tmp_path, v = recorded
    out = tmp_path / "m"
    first = media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, out)
    (out / "video" / "stale.mp4").write_bytes(b"old")
    (out / "keep.txt").write_text("not ours")
    again = media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, out)
    assert again == first
    assert not (out / "video" / "stale.mp4").exists() and (out / "keep.txt").exists()
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt", "mcap", "media.json", "video"]


def test_media_fetches_the_pinned_files_itself(recorded, monkeypatch):
    tmp_path, v = recorded
    fetched = []
    monkeypatch.setattr(media, "fetch", lambda root: fetched.append(root))
    media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")
    assert fetched == [Toy.root]


def test_media_fetches_before_the_adapter_loads_from_a_fresh_checkout(recorded, monkeypatch):
    """A fresh checkout has no pinned files; media must fetch them before loading anything."""
    tmp_path, v = recorded
    pinned = Toy.root / ".cache" / "toy" / "profile.json"
    assert not pinned.exists()  # nothing fetched: a fresh checkout
    calls = []

    def needs_pins(method):
        def load(self, *args):
            calls.append(method.__name__)
            if not pinned.exists():
                raise FileNotFoundError(pinned)
            return method(self, *args)

        return load

    def fetch(root):
        calls.append("fetch")
        pinned.parent.mkdir(parents=True, exist_ok=True)
        pinned.write_text("{}")

    for name in ("robot", "configuration", "world", "scene"):
        monkeypatch.setattr(Toy, name, needs_pins(getattr(Toy, name)))
    monkeypatch.setattr(media, "fetch", fetch)
    doc = media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")
    assert calls[0] == "fetch" and calls.count("fetch") == 1
    assert len(doc["trials"]) == 2


def test_a_replay_that_really_differs_is_still_refused(recorded, monkeypatch):
    # Identical trials hash the same. A replay that is actually a different run is still refused.
    tmp_path, v = recorded
    real = evidence.run_trial

    def differ(*args, **kwargs):
        obs = real(*args, **kwargs)
        return {**obs, "end": "fell"}

    monkeypatch.setattr(evidence, "run_trial", differ)
    with pytest.raises(RuntimeError, match="differs from the evidence"):
        media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")
    assert not (tmp_path / "m" / "mcap").exists()
