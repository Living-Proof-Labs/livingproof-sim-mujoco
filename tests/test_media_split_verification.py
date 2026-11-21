"""Independent checks of the MCAP split (#108): the 2.1.0 reader contract, an empty robot file
as a complete MCAP, and a mismatch into an existing output keeping the earlier files."""

import json
import os
import shutil
import subprocess
from pathlib import Path

# Before livingproof_sim_mujoco, which imports MuJoCo and locks the GL platform.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import pytest

from livingproof_sim_mujoco import media

pytest.importorskip("mcap")
pytest.importorskip("imageio_ffmpeg")

import test_robot_topics as base  # noqa: E402
from mcap.reader import make_reader  # noqa: E402

V210_TRIAL_KEYS = {"index", "observation_sha256", "mcap", "binding", "status", "video"}


def test_a_2_1_0_reader_still_finds_every_field_it_used(tmp_path, monkeypatch):
    doc = base._hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")
    assert {"format", "configuration", "run_id", "trials"} <= set(doc)
    for t in doc["trials"]:
        assert V210_TRIAL_KEYS <= set(t)
        assert set(t["mcap"]) == {"path", "bytes", "sha256"}  # unchanged shape
        assert t["mcap"]["path"] != t["robot_mcap"]["path"]
        assert t["mcap"]["sha256"] != t["robot_mcap"]["sha256"]


def test_an_empty_robot_file_is_a_complete_mcap(tmp_path, monkeypatch):
    doc = base._hallway(tmp_path, monkeypatch, "test_evidence:TOY")
    path = tmp_path / "m" / doc["trials"][0]["robot_mcap"]["path"]
    with path.open("rb") as f:
        summary = make_reader(f).get_summary()
    assert summary is not None and summary.statistics.message_count == 0
    cli = shutil.which("mcap")
    if cli is not None:
        done = subprocess.run([cli, "doctor", str(path)], capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr


def test_a_mismatch_into_an_existing_output_keeps_the_earlier_files_and_no_staging(
    tmp_path, monkeypatch
):
    base._hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")
    out = tmp_path / "m"
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    manifest = json.loads((tmp_path / "ev" / "manifest.json").read_text())
    manifest["trials"]["results"][0]["sha256"] = "0" * 64
    (tmp_path / "ev" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="differs from the evidence"):
        media.media(
            "test_robot_topics:PUB", "stock", tmp_path / "ev", tmp_path / "verification.json", out
        )
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    assert not (out / ".media-partial").exists()
