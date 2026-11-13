# SPDX-License-Identifier: Apache-2.0
"""Frozen hallway scenario: names, the tagged parameter file, and recorded rounding.

This is the compatibility module for scenario ids. Generic modules do not name
the scenario or the world. The parameter file is the byte copy of
robot-capability-specs v2.2.0 ``specs/scenarios/walk-hallway/parameters.json``
(commit 23bec516bda29be7b26555286cbf2e8217ac5c81). It is the last resort when
the verification lock and the robot pins do not carry the file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARAMETERS_SHA256 = "9226aa6bc0e577863bdc210514e7e7c3a7edc32f49c4f7b46aa7249837f08b3d"
PARAMETERS_TAG = "v2.2.0"
PARAMETERS_COMMIT = "23bec516bda29be7b26555286cbf2e8217ac5c81"
SCENARIO = "walk-hallway@0.1.0"
EXIT, SPAWN = "exit", "spawn-entrance"
# Pins path used when a robot lock does not name a world id and version.
FROZEN_WORLD_PATH = "worlds/hallway-open/0.1.0/world.json"
WORLD_FORMAT_0_1 = "livingproof-world/0.1"
WORLD_FORMAT_0_2 = "livingproof-world/0.2"
_PATH = Path(__file__).with_name("parameters.json")


def parameters_document() -> dict:
    """The tagged parameter file. A sha256 mismatch is a test-system error."""
    raw = _PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PARAMETERS_SHA256:
        raise RuntimeError(
            f"ERROR: vendored parameter file sha256 {digest} != {PARAMETERS_SHA256} "
            f"({PARAMETERS_TAG} {PARAMETERS_COMMIT}). "
            "Restore compat/parameters.json from that tag."
        )
    return json.loads(raw)


_DOCUMENT = parameters_document()
COMMAND_SPEED = _DOCUMENT["parameters"]["command_speed"]["value"]
WALL_CLEARANCE = _DOCUMENT["parameters"]["wall_clearance"]["values"]


def evidence_run_id(scenario: str, robot_id: str, configuration: str, created) -> str:
    """Run id. The frozen scenario keeps its historical prefix."""
    if isinstance(scenario, str) and scenario.startswith("walk-hallway@"):
        head = "walk-hallway"
    elif isinstance(scenario, str) and "@" in scenario:
        head = scenario.split("@", 1)[0]
    else:
        head = "scenario"
    return f"{head}-{robot_id}-{configuration}-{created:%Y%m%dT%H%M%SZ}"


def recorded_limit(document: dict, name: str, raw_value: float, spec_value: float) -> float:
    """Value stored on a frozen hallway record.

    ``time_limit`` stays unrounded and ``lateral_limit`` stays at 6 decimal
    places, which is what hallway evidence already records. Any other scenario
    stores the spec-rounded value from ``resolve``.
    """
    scenario = document.get("scenario") if isinstance(document, dict) else None
    frozen = (
        isinstance(scenario, dict)
        and scenario.get("id") == "walk-hallway"
        and scenario.get("revision") == "0.1.0"
    )
    if frozen and name == "time_limit":
        return raw_value
    if frozen and name == "lateral_limit":
        return round(float(raw_value), 6)
    return spec_value


def unsupported_world(world: dict) -> str | None:
    """ERROR text when this release cannot record the world, else None.

    Format 0.2 needs a neutral observation writer. This release does not have
    one, and the frozen hallway writer must not emit its format for that world.
    """
    if not isinstance(world, dict):
        return None
    if world.get("format") != WORLD_FORMAT_0_2:
        return None
    ident = f"{world.get('id')!r} {world.get('content_version')!r}"
    return (
        "ERROR: world format livingproof-world/0.2 has no observation writer yet "
        f"({ident}). The frozen hallway observation format stays on format 0.1 worlds. "
        "A neutral format for 0.2 worlds is not in this release, so nothing was written. "
        "Example: record a livingproof-world/0.1 world, such as the pinned open hallway."
    )
