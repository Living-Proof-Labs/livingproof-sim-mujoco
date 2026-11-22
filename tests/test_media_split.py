"""Robot-reported topics and sim ground-truth topics go to separate MCAP files (#108)."""

import json
import os
from pathlib import Path

# Before livingproof_sim_mujoco, which imports MuJoCo and locks the GL platform.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import pytest

from livingproof_sim_mujoco import media, robot_topics

pytest.importorskip("mcap")
pytest.importorskip("imageio_ffmpeg")

import test_robot_topics as base  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

SIM_ONLY = {"/base_pose", "/joint_states", "/wall_contacts"}


def _topics(path: Path) -> set[str]:
    with path.open("rb") as f:
        return {c.topic for _, c, _ in make_reader(f).iter_messages()}


def _read_all(path: Path):
    with path.open("rb") as f:
        return list(make_reader(f).iter_messages())


def _count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in make_reader(f).iter_messages())


@pytest.fixture
def hooked(tmp_path, monkeypatch):
    doc = base._hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")
    return tmp_path / "m", doc


@pytest.fixture
def unhooked(tmp_path, monkeypatch):
    doc = base._hallway(tmp_path, monkeypatch, "test_evidence:TOY")
    return tmp_path / "m", doc


def test_the_sim_file_has_no_robot_channels_and_the_robot_file_no_sim_channels(hooked):
    out, doc = hooked
    for t in doc["trials"]:
        sim, robot = _topics(out / t["mcap"]["path"]), _topics(out / t["robot_mcap"]["path"])
        assert "/estimated_pose" not in sim and sim <= SIM_ONLY
        assert robot == {"/estimated_pose"}
        for sim_topic in ("/base_pose", "/joint_states"):
            assert sim_topic not in robot


def test_a_robot_file_with_no_robot_messages_is_still_written_and_valid(unhooked):
    out, doc = unhooked
    for t in doc["trials"]:
        path = out / t["robot_mcap"]["path"]
        assert path.is_file() and _count(path) == 0
        assert {"/base_pose", "/joint_states"} <= _topics(out / t["mcap"]["path"])


def test_media_json_names_both_files_with_size_and_sha256_under_a_new_minor_format(hooked):
    out, doc = hooked
    assert doc["format"] == media.FORMAT == "livingproof-hallway-media/2.2.0"
    assert json.loads((out / "media.json").read_text()) == doc
    for t in doc["trials"]:
        assert t["mcap"]["path"] == f"mcap/trial-{t['index']:02d}.mcap"  # compat field: sim file
        assert t["robot_mcap"]["path"] == f"mcap/trial-{t['index']:02d}.robot.mcap"
        for key in ("mcap", "robot_mcap"):
            entry = t[key]
            path = out / entry["path"]
            assert entry["bytes"] == path.stat().st_size
            assert entry["sha256"] == media.sha256(path)


def test_the_binding_still_describes_the_sim_recording(hooked):
    out, doc = hooked
    for t in doc["trials"]:
        binding = json.loads((out / t["binding"]["path"]).read_text())
        assert binding["recording"] == {
            "uri": Path(t["mcap"]["path"]).name,
            "sha256": t["mcap"]["sha256"],
        }


def test_the_split_does_not_change_the_graded_trial(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = base._hallway(tmp_path / "a", monkeypatch, "test_robot_topics:PUB")
    two = base._hallway(tmp_path / "b", monkeypatch, "test_evidence:TOY")
    assert [t["observation_sha256"] for t in one["trials"]] == [
        t["observation_sha256"] for t in two["trials"]
    ]  # media() also refuses any trial whose re-run hash differs from the evidence
    sim = [
        [
            (m.log_time, m.data)
            for _, _, m in _read_all(tmp_path / d / "m" / doc["trials"][0]["mcap"]["path"])
        ]
        for d, doc in (("a", one), ("b", two))
    ]
    assert sim[0] == sim[1]  # publishing no longer touches the sim file at all


def test_a_rerun_replaces_both_files_and_a_mismatch_keeps_neither(tmp_path, monkeypatch):
    doc = base._hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")
    out = tmp_path / "m"
    stale = out / "mcap" / "trial-99.robot.mcap"
    stale.write_bytes(b"old")
    again = media.media(
        "test_robot_topics:PUB", "stock", tmp_path / "ev", tmp_path / "verification.json", out
    )
    assert again == doc and not stale.exists()
    manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    manifest["trials"]["results"][0]["sha256"] = "0" * 64
    (tmp_path / "ev" / "manifest.json").write_text(json.dumps(manifest))
    shutil_out = tmp_path / "bad"
    with pytest.raises(RuntimeError, match="differs from the evidence"):
        media.media(
            "test_robot_topics:PUB",
            "stock",
            tmp_path / "ev",
            tmp_path / "verification.json",
            shutil_out,
        )
    assert not list(shutil_out.rglob("*.mcap"))


def test_a_recorder_given_a_hook_but_no_robot_file_is_refused(tmp_path):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'><freejoint/><geom size='.1'/></body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="robot_mcap_path"):
        media.Recorder(model, "b", mcap_path=tmp_path / "s.mcap", publish=lambda step: [])
    assert robot_topics  # the hook's module is the one the recorder uses
