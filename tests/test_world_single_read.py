# SPDX-License-Identifier: Apache-2.0
"""Every runner entry point reads the world file once and uses those bytes (#110)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import test_scenario_sources as t

from livingproof_sim_mujoco import compose, evidence, hallway, media, sources, walk

VENDORED = t.ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json"


def _declared_world_repo(tmp_path, monkeypatch, *, clear_width: float = 1.6):
    toy, repo = t._toy_repo(tmp_path, monkeypatch)
    declared = json.loads(VENDORED.read_text())
    declared["dimensions"] = {**declared["dimensions"], "clear_width": clear_width}
    digest = t._lock_world(repo, declared)
    t._edit_lock(repo, world={**t.LEGACY_WORLD, "sha256": digest})
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return toy, repo, digest


def _spy_single_world_read(monkeypatch, world_path: Path):
    """Spy on ``_read_locked`` and ``Path.read_bytes``, mutating the world after the first read."""
    locked_reads: list[str] = []
    byte_reads: list[str] = []
    target = world_path.resolve()

    real_locked = sources._read_locked
    real_read_bytes = Path.read_bytes

    def counting_locked(root, entry, kind):
        data = real_locked(root, entry, kind)
        if kind == "world":
            locked_reads.append(hashlib.sha256(data).hexdigest())
            doc = json.loads(data)
            doc["dimensions"]["clear_width"] = 9.9
            world_path.write_text(json.dumps(doc))
        return data

    def counting_bytes(self):
        data = real_read_bytes(self)
        if self.resolve() == target:
            byte_reads.append(hashlib.sha256(data).hexdigest())
        return data

    monkeypatch.setattr(sources, "_read_locked", counting_locked)
    monkeypatch.setattr(Path, "read_bytes", counting_bytes)
    return locked_reads, byte_reads


def test_evidence_reads_sources_world_once_across_trials(tmp_path, monkeypatch):
    _toy, repo, digest = _declared_world_repo(tmp_path, monkeypatch)
    world_path = repo / "locked/world.json"
    locked_reads, byte_reads = _spy_single_world_read(monkeypatch, world_path)

    out = tmp_path / "ev"
    manifest = t._evidence(repo, out, trials=2)

    assert locked_reads == [digest]
    assert byte_reads == [digest]
    assert manifest["world"]["clear_width"] == 1.6
    assert len(manifest["trials"]["results"]) == 2
    run_manifest = json.loads((out / "run-manifest.json").read_text())
    assert run_manifest["world"]["sha256"] == digest


def test_media_reads_sources_world_once_through_prepare_run(tmp_path, monkeypatch):
    pytest.importorskip("mcap")
    pytest.importorskip("imageio_ffmpeg")

    _toy, repo, digest = _declared_world_repo(tmp_path, monkeypatch)
    ev_out = tmp_path / "ev"
    t._evidence(repo, ev_out, trials=2)
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps({"trials": []}))

    world_path = repo / "locked/world.json"
    locked_reads, byte_reads = _spy_single_world_read(monkeypatch, world_path)

    world_source_calls: list[tuple] = []
    real_world_source = sources.world_source

    def counting_world_source(root, lock):
        world_source_calls.append((root, lock))
        return real_world_source(root, lock)

    monkeypatch.setattr(sources, "world_source", counting_world_source)

    seen_widths: list[float] = []
    real_place = compose.place

    def capturing_place(robot, name, world, spawn, start_offset=None):
        seen_widths.append(world["dimensions"]["clear_width"])
        return real_place(robot, name, world, spawn, start_offset)

    monkeypatch.setattr(compose, "place", capturing_place)

    doc = media.media("test_evidence:TOY", "stock", ev_out, verification, tmp_path / "media")

    assert len(world_source_calls) == 1
    assert locked_reads == [digest]
    assert byte_reads == [digest]
    assert seen_widths == [1.6, 1.6]
    assert len(doc["trials"]) == 2


def test_media_checks_world_identity_from_single_read(tmp_path, monkeypatch):
    pytest.importorskip("mcap")
    pytest.importorskip("imageio_ffmpeg")

    _toy, repo, digest = _declared_world_repo(tmp_path, monkeypatch)
    ev_out = tmp_path / "ev"
    t._evidence(repo, ev_out, trials=1)
    manifest_path = ev_out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lock"]["content"]["world"]["sha256"] = "ab" * 32
    manifest_path.write_text(json.dumps(manifest))

    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps({"trials": []}))

    locked_reads, byte_reads = _spy_single_world_read(monkeypatch, repo / "locked/world.json")
    with pytest.raises(SystemExit, match="world.sha256"):
        media.media("test_evidence:TOY", "stock", ev_out, verification, tmp_path / "media")
    assert locked_reads == [digest]
    assert byte_reads == [digest]


def test_walk_stand_reads_sources_world_once_through_prepare_run(tmp_path, monkeypatch):
    toy, repo, digest = _declared_world_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(
        toy.Toy, "runner_settings", lambda self: {"sample_hz": 50.0, "stand_seconds": 0.2}
    )

    world_path = repo / "locked/world.json"
    locked_reads, byte_reads = _spy_single_world_read(monkeypatch, world_path)

    world_source_calls: list[tuple] = []
    real_world_source = sources.world_source

    def counting_world_source(root, lock):
        world_source_calls.append((root, lock))
        return real_world_source(root, lock)

    monkeypatch.setattr(sources, "world_source", counting_world_source)

    seen_widths: list[float] = []
    real_place = compose.place

    def capturing_place(robot, name, world, spawn, start_offset=None):
        seen_widths.append(world["dimensions"]["clear_width"])
        return real_place(robot, name, world, spawn, start_offset)

    monkeypatch.setattr(compose, "place", capturing_place)

    result = walk.stand(toy.TOY, "stock", out=None, lock=repo / "caller.lock.json")

    assert len(world_source_calls) == 1
    assert locked_reads == [digest]
    assert byte_reads == [digest]
    assert seen_widths == [1.6]
    assert result["check"] == "zero-command-stand"
    assert result["standing"] is True


def test_walk_stand_checks_world_identity_from_single_read(tmp_path, monkeypatch):
    toy, repo, digest = _declared_world_repo(tmp_path, monkeypatch)
    t._edit_lock(repo, world={**t.LEGACY_WORLD, "sha256": "ab" * 32})
    monkeypatch.setattr(
        toy.Toy, "runner_settings", lambda self: {"sample_hz": 50.0, "stand_seconds": 0.2}
    )

    locked_reads, byte_reads = _spy_single_world_read(monkeypatch, repo / "locked/world.json")
    with pytest.raises(SystemExit, match="world.sha256"):
        walk.stand(toy.TOY, "stock", out=None, lock=repo / "caller.lock.json")
    assert locked_reads == [digest]
    assert byte_reads == [digest]


def test_vendored_world_is_read_once_in_load_fixture_evidence_media_and_walk(tmp_path, monkeypatch):
    pytest.importorskip("mcap")
    pytest.importorskip("imageio_ffmpeg")

    vendored_target = VENDORED.resolve()
    reads: list[str] = []
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def counting_bytes(self):
        if self.resolve() == vendored_target:
            reads.append("bytes")
        return real_read_bytes(self)

    def counting_text(self, *args, **kwargs):
        if self.resolve() == vendored_target:
            reads.append("text")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting_bytes)
    monkeypatch.setattr(Path, "read_text", counting_text)

    loaded = hallway.load_fixture("hallway-open", "0.1.0")
    assert (
        len(reads) == 1
        and loaded["world_sha256"] == hashlib.sha256(VENDORED.read_bytes()).hexdigest()
    )
    # Subtract the test's own VENDORED.read_bytes() above.
    reads.clear()

    compose.vendored_world({"world": t.LEGACY_WORLD})
    assert len(reads) == 1
    reads.clear()

    toy, repo = t._toy_repo(tmp_path, monkeypatch)
    reads.clear()
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    monkeypatch.setattr(
        toy.Toy, "runner_settings", lambda self: {"sample_hz": 50.0, "stand_seconds": 0.2}
    )

    ev_out = tmp_path / "ev"
    t._evidence(repo, ev_out, trials=2)
    assert len(reads) == 1
    reads.clear()

    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps({"trials": []}))
    media.media("test_evidence:TOY", "stock", ev_out, verification, tmp_path / "media")
    assert len(reads) == 1
    reads.clear()

    walk.stand(toy.TOY, "stock", out=None, lock=repo / "caller.lock.json")
    assert len(reads) == 1


def test_prepare_run_and_recording_reuse_passed_world_source(tmp_path, monkeypatch):
    toy, repo, digest = _declared_world_repo(tmp_path, monkeypatch)
    lock = json.loads((repo / "caller.lock.json").read_text())
    selected = sources.world_source(repo, lock)
    params = sources.parameters_source(repo, lock)

    # Mutate both files on disk; reusing the loaded sources must not touch disk again.
    (repo / "locked/world.json").write_text("{}\n")
    (repo / lock["sources"]["parameters"]["path"]).write_text("{}\n")

    monkeypatch.setattr(
        sources,
        "world_source",
        lambda *a: (_ for _ in ()).throw(AssertionError("world_source re-read")),
    )
    monkeypatch.setattr(
        sources,
        "parameters_source",
        lambda *a: (_ for _ in ()).throw(AssertionError("parameters_source re-read")),
    )

    _plan, prepared = evidence.recording(toy.TOY, lock, "stock", params, selected)
    assert prepared.world_source is selected
    assert prepared.world_source.sha256 == digest
    assert prepared.world["dimensions"]["clear_width"] == 1.6
