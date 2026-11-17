# SPDX-License-Identifier: Apache-2.0
"""Reviewer tests for #89 (livingproof-verification#119 slice 1).

livingproof-world/0.3 requires ``traversals``: a non-empty list of objects with
exactly ``name``, ``from``, ``start``, ``route`` and ``finish``, each naming a
place. A world that breaks that, or lacks geometry or places, is refused with
``Unsupported`` naming the problem, not accepted or left to a later KeyError.
"""

from __future__ import annotations

import json
import shutil
import re
import hashlib
import math

import pytest
import test_world_format_0_3 as t

from livingproof_sim_mujoco import hallway

VENDORED = t.ROOT / "src" / "livingproof_sim_mujoco" / "worlds"


def _first(world: dict, **changes) -> None:
    world["traversals"][0].update(changes)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: _first(w, start="nope"), id="start-not-a-place"),
        pytest.param(lambda w: _first(w, route="nope"), id="route-not-a-place"),
        pytest.param(lambda w: _first(w, finish="nope"), id="finish-not-a-place"),
        pytest.param(lambda w: _first(w, **{"from": "nope"}), id="from-not-a-place"),
        pytest.param(lambda w: w["traversals"][0].pop("finish"), id="missing-finish"),
        pytest.param(lambda w: _first(w, speed=1), id="extra-field"),
        pytest.param(lambda w: w.update(traversals="x"), id="not-a-list"),
        pytest.param(lambda w: w.update(traversals=["x"]), id="item-not-an-object"),
        pytest.param(lambda w: w.update(traversals=[]), id="empty"),
        pytest.param(lambda w: w.pop("traversals"), id="absent"),
    ],
)
def test_invalid_traversals_are_refused(edit):
    world = t._world()
    edit(world)
    with pytest.raises(hallway.Unsupported, match="traversal"):
        hallway.supported(world)


@pytest.mark.parametrize("field", ["geometry", "places"])
def test_incomplete_world_is_refused_not_a_key_error(field):
    world = t._world()
    del world[field]
    with pytest.raises(hallway.Unsupported, match=field):
        hallway.supported(world)


@pytest.mark.parametrize(
    "fixture,expected_format",
    [
        ("hallway-open/0.1.0", "livingproof-world/0.1"),
        ("hallway-corner/0.1.0", "livingproof-world/0.2"),
    ],
)
def test_older_vendored_formats_are_still_accepted(fixture, expected_format):
    world = json.loads((VENDORED / fixture / "world.json").read_text())
    assert world["format"] == expected_format
    hallway.supported(world)


def _wall(world: dict) -> dict:
    return next(g for g in world["geometry"] if g["role"] == "wall")


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: w["geometry"].__setitem__(0, "x"), id="geometry-item-not-an-object"),
        pytest.param(lambda w: w["places"].__setitem__(0, "x"), id="place-item-not-an-object"),
        pytest.param(lambda w: _wall(w).pop("size"), id="wall-without-size"),
        pytest.param(lambda w: _wall(w).pop("centre"), id="wall-without-centre"),
        pytest.param(lambda w: _wall(w).update(size="big"), id="wall-size-not-a-list"),
        pytest.param(lambda w: _first(w, start=["spawn-west"]), id="traversal-start-a-list"),
        pytest.param(lambda w: _first(w, finish={"a": 1}), id="traversal-finish-an-object"),
    ],
)
def test_wrong_shaped_fields_are_refused_not_a_crash(edit):
    world = t._world()
    edit(world)
    with pytest.raises(hallway.Unsupported):
        hallway.supported(world)


def _named(world: dict, name: str) -> dict:
    return next(p for p in world["places"] if p["name"] == name)


def _pose(world: dict) -> dict:
    return next(p for p in world["places"] if p["type"] == "pose")


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: _wall(w).update(size=[True, 1, 1]), id="size-bool"),
        pytest.param(lambda w: _wall(w).update(size=[math.nan, 1, 1]), id="size-nan"),
        pytest.param(lambda w: _wall(w).update(size=[math.inf, 1, 1]), id="size-inf"),
        pytest.param(lambda w: _pose(w).update(yaw_deg=True), id="yaw-bool"),
        pytest.param(lambda w: _pose(w).update(position=[math.inf, 0, 0]), id="position-inf"),
        pytest.param(
            lambda w: _named(w, "end-north").update(points=[[[0, 0, 0]], [1, 1, 1]]),
            id="points-nested",
        ),
        pytest.param(lambda w: _wall(w).update(size=[-1, 1, 1]), id="size-negative"),
        pytest.param(lambda w: _wall(w).update(size=[0, 0, 0]), id="size-zero"),
        pytest.param(lambda w: _named(w, "end-north").update(points=[]), id="segment-no-points"),
        pytest.param(
            lambda w: _named(w, "end-north").update(points=[[0, 0, 0]]), id="segment-one-point"
        ),
        pytest.param(
            lambda w: _named(w, "route-forward").update(points=[[0, 0, 0]]),
            id="polyline-one-point",
        ),
    ],
)
def test_invalid_numbers_or_point_counts_are_unsupported(edit):
    world = t._world()
    edit(world)
    with pytest.raises(hallway.Unsupported):
        hallway.supported(world)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda w: w.pop("license"), id="missing-license"),
        pytest.param(lambda w: w.update(content_version="latest"), id="content-version-not-semver"),
        pytest.param(lambda w: w["dimensions"].update(bogus=1), id="dimensions-extra-key"),
        pytest.param(lambda w: _wall(w).update(name="Not Kebab"), id="geometry-name-pattern"),
    ],
)
def test_published_schema_rules_are_unsupported(edit):
    """Rules only the published schema states, so the schema check itself is exercised."""
    world = t._world()
    edit(world)
    with pytest.raises(hallway.Unsupported):
        hallway.supported(world)


def test_vendored_schemas_match_their_pins():
    from livingproof_sim_mujoco import world_shapes

    pin = json.loads((world_shapes.SCHEMA_DIR / "pins.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", pin["commit"])
    assert set(world_shapes.SCHEMA_BY_FORMAT.values()) <= set(pin["files"])
    for name, digest in pin["files"].items():
        data = (world_shapes.SCHEMA_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_a_tampered_vendored_schema_is_refused(tmp_path, monkeypatch):
    from livingproof_sim_mujoco import world_shapes

    copy = tmp_path / "schema"
    shutil.copytree(world_shapes.SCHEMA_DIR, copy)
    target = copy / "world-0.3.schema.json"
    target.write_text(target.read_text().replace('"minItems": 2', '"minItems": 1', 1))
    monkeypatch.setattr(world_shapes, "SCHEMA_DIR", copy)
    world_shapes._load_schema.cache_clear()
    world_shapes._schema_pin.cache_clear()
    try:
        with pytest.raises(hallway.Unsupported, match="checksum"):
            hallway.supported(t._world())
    finally:
        world_shapes._load_schema.cache_clear()
        world_shapes._schema_pin.cache_clear()


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda w: w["geometry"][3].update(name=w["geometry"][2]["name"]),
            id="duplicate-geometry-name",
        ),
        pytest.param(
            lambda w: w["places"][2].update(name=w["places"][0]["name"]),
            id="duplicate-place-name",
        ),
    ],
)
def test_duplicate_names_are_unsupported_not_a_mujoco_error(edit):
    world = t._world()
    edit(world)
    with pytest.raises(hallway.Unsupported, match="name"):
        hallway.supported(world)
