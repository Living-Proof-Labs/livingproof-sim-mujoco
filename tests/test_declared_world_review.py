"""Reviewer tests for #88: a world declared in lock sources.world (#119).

The declared bytes replace the vendored copy, so they must be checked before use:
an incompatible document is refused with a clear message and nothing is written.
"""

import json
import math
import shutil

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


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: w.update(geometry="x"), id="geometry-not-a-list"),
        pytest.param(lambda w: w["geometry"][1].pop("size"), id="geometry-item-without-size"),
        pytest.param(lambda w: w["places"].__setitem__(0, "x"), id="place-not-an-object"),
    ],
)
def test_declared_world_with_malformed_shapes_is_refused_clearly(tmp_path, monkeypatch, edit):
    world = json.loads(VENDORED.read_text())
    edit(world)
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as refused:
        t._evidence(repo, out, trials=1)
    assert str(refused.value).startswith(("ERROR", "INCOMPATIBLE", "UNSUPPORTED"))
    assert not out.exists()


def test_declared_world_with_an_unmappable_feature_is_unsupported(tmp_path, monkeypatch):
    world = json.loads(VENDORED.read_text())
    world["geometry"][1]["type"] = "mesh"
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="UNSUPPORTED: .*type 'mesh'"):
        t._evidence(repo, out, trials=1)
    assert not out.exists()


def _wall(world: dict) -> dict:
    return next(g for g in world["geometry"] if g["role"] == "wall")


def _pose(world: dict) -> dict:
    return next(p for p in world["places"] if p["type"] == "pose")


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: w["geometry"].__setitem__(0, "x"), id="geometry-item-not-an-object"),
        pytest.param(lambda w: w.update(geometry=[]), id="geometry-empty"),
        pytest.param(lambda w: w.update(dimensions=[]), id="dimensions-not-an-object"),
        pytest.param(lambda w: _wall(w).update(size="big"), id="wall-size-a-string"),
        pytest.param(lambda w: _wall(w).update(size=[1, 2]), id="wall-size-two-numbers"),
        pytest.param(lambda w: _wall(w).update(size=["a", "b", "c"]), id="wall-size-strings"),
        pytest.param(lambda w: _wall(w).update(centre="here"), id="wall-centre-a-string"),
        pytest.param(lambda w: _wall(w).update(centre=None), id="wall-centre-null"),
        pytest.param(lambda w: _wall(w).update(name=7), id="geometry-name-not-a-string"),
        pytest.param(lambda w: _pose(w).update(position="x"), id="pose-position-a-string"),
        pytest.param(lambda w: _pose(w).update(yaw_deg="north"), id="pose-yaw-a-string"),
    ],
)
def test_declared_world_with_wrong_typed_values_is_refused_clearly(tmp_path, monkeypatch, edit):
    world = json.loads(VENDORED.read_text())
    edit(world)
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as refused:
        t._evidence(repo, out, trials=1)
    assert str(refused.value).startswith(("ERROR", "INCOMPATIBLE", "UNSUPPORTED"))
    assert not out.exists()


def _place(world: dict, name: str) -> dict:
    return next(p for p in world["places"] if p["name"] == name)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: _wall(w).update(size=[True, 1, 1]), id="size-bool"),
        pytest.param(lambda w: _wall(w).update(size=[math.nan, 1, 1]), id="size-nan"),
        pytest.param(lambda w: _wall(w).update(size=[math.inf, 1, 1]), id="size-inf"),
        pytest.param(lambda w: _pose(w).update(yaw_deg=True), id="yaw-bool"),
        pytest.param(lambda w: _pose(w).update(position=[math.inf, 0, 0]), id="position-inf"),
        pytest.param(
            lambda w: _place(w, "exit").update(points=[[[0, 0, 0]], [1, 1, 1]]), id="points-nested"
        ),
        pytest.param(lambda w: _wall(w).update(size=[-1, 1, 1]), id="size-negative"),
        pytest.param(lambda w: _wall(w).update(size=[0, 0, 0]), id="size-zero"),
        pytest.param(lambda w: _place(w, "exit").update(points=[]), id="segment-no-points"),
        pytest.param(
            lambda w: _place(w, "exit").update(points=[[0, 0, 0]]), id="segment-one-point"
        ),
        pytest.param(
            lambda w: _place(w, "passage-centreline").update(points=[[0, 0, 0]]),
            id="polyline-one-point",
        ),
    ],
)
def test_declared_world_with_invalid_numbers_or_point_counts_is_incompatible(
    tmp_path, monkeypatch, edit
):
    world = json.loads(VENDORED.read_text())
    edit(world)
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="^INCOMPATIBLE"):
        t._evidence(repo, out, trials=1)
    assert not out.exists()


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: w.pop("license"), id="missing-license"),
        pytest.param(lambda w: w.update(content_version="latest"), id="content-version-not-semver"),
        pytest.param(lambda w: w["dimensions"].update(bogus=1), id="dimensions-extra-key"),
    ],
)
def test_declared_world_breaking_published_schema_rules_is_incompatible(
    tmp_path, monkeypatch, edit
):
    """Rules only the published schema states, so the schema check itself is exercised."""
    world = json.loads(VENDORED.read_text())
    edit(world)
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="^INCOMPATIBLE"):
        t._evidence(repo, out, trials=1)
    assert not out.exists()


def test_declared_world_with_a_tampered_vendored_schema_is_refused(tmp_path, monkeypatch):
    from livingproof_sim_mujoco import world_shapes

    copy = tmp_path / "schema"
    shutil.copytree(world_shapes.SCHEMA_DIR, copy)
    target = copy / "world.schema.json"
    target.write_text(target.read_text().replace('"minItems": 2', '"minItems": 1', 1))
    monkeypatch.setattr(world_shapes, "SCHEMA_DIR", copy)
    world_shapes._load_schema.cache_clear()
    world_shapes._schema_pin.cache_clear()
    try:
        repo = _declare(tmp_path, monkeypatch, json.loads(VENDORED.read_text()), t.LEGACY_WORLD)
        out = tmp_path / "out"
        with pytest.raises(SystemExit, match="checksum"):
            t._evidence(repo, out, trials=1)
        assert not out.exists()
    finally:
        world_shapes._load_schema.cache_clear()
        world_shapes._schema_pin.cache_clear()


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda w: w["geometry"].append(dict(w["geometry"][1])),
            id="duplicate-geometry-name",
        ),
        pytest.param(
            lambda w: w["places"].append(dict(_place(w, "spawn-entrance"))),
            id="duplicate-place-name",
        ),
    ],
)
def test_declared_world_with_duplicate_names_is_refused_clearly(tmp_path, monkeypatch, edit):
    world = json.loads(VENDORED.read_text())
    edit(world)
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="^(INCOMPATIBLE|UNSUPPORTED)"):
        t._evidence(repo, out, trials=1)
    assert not out.exists()


@pytest.mark.parametrize(
    ("field", "edit"),
    [
        pytest.param(
            "geometry",
            lambda w: w["geometry"].append(dict(w["geometry"][1])),
            id="duplicate-geometry-name",
        ),
        pytest.param(
            "places",
            lambda w: w["places"].append(dict(_place(w, "spawn-entrance"))),
            id="duplicate-place-name",
        ),
    ],
)
def test_declared_world_duplicate_names_are_incompatible(tmp_path, monkeypatch, field, edit):
    """A declared document with repeated names is invalid input, so INCOMPATIBLE."""
    world = json.loads(VENDORED.read_text())
    edit(world)
    repo = _declare(tmp_path, monkeypatch, world, t.LEGACY_WORLD)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=f"^INCOMPATIBLE: .*duplicate {field} name"):
        t._evidence(repo, out, trials=1)
    assert not out.exists()
