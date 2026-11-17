"""Reviewer tests for #88: a world declared in lock sources.world (#119).

The declared bytes replace the vendored copy, so they must be checked before use:
an incompatible document is refused with a clear message and nothing is written.
"""

import json

import pytest
import test_scenario_sources as t

VENDORED = t.ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json"


def _declare(tmp_path, monkeypatch, world, lock_world):
    _toy, repo = t._toy_repo(tmp_path, monkeypatch)
    digest = t._lock_world(repo, world)
    t._edit_lock(repo, world={**lock_world, "sha256": digest})
    return repo


def test_declared_world_with_unvendored_version_is_used(tmp_path, monkeypatch):
    world = json.loads(VENDORED.read_text())
    world["content_version"] = "0.9.0"
    repo = _declare(tmp_path, monkeypatch, world, {"id": "hallway-open", "version": "0.9.0"})
    manifest = t._evidence(repo, tmp_path / "out", trials=1)
    assert manifest["trials"]["results"][0]["outcome"] == "crossed_exit"


def test_declared_world_without_geometry_is_refused_clearly(tmp_path, monkeypatch):
    world = json.loads(VENDORED.read_text())
    del world["geometry"]
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as refused:
        t._evidence(repo, out, trials=1)
    assert str(refused.value).startswith(("ERROR", "INCOMPATIBLE", "UNSUPPORTED"))
    assert "geometry" in str(refused.value)
    assert not out.exists()


def test_declared_world_that_is_not_an_object_is_refused(tmp_path, monkeypatch):
    repo = _declare(tmp_path, monkeypatch, [], t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as refused:
        t._evidence(repo, out, trials=1)
    assert str(refused.value).startswith(("ERROR", "INCOMPATIBLE", "UNSUPPORTED"))
    assert not out.exists()
