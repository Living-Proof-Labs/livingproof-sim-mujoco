# SPDX-License-Identifier: Apache-2.0
"""Reviewer tests for #89 (livingproof-verification#119 slice 1).

livingproof-world/0.3 requires ``traversals``: a non-empty list of objects with
exactly ``name``, ``from``, ``start``, ``route`` and ``finish``, each naming a
place. A world that breaks that, or lacks geometry or places, is refused with
``Unsupported`` naming the problem, not accepted or left to a later KeyError.
"""

from __future__ import annotations

import json

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
