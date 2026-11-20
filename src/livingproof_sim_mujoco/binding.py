# SPDX-License-Identifier: Apache-2.0
"""The producer and per-trial halves of livingproof-observation-binding/1.0.0 (#28).

The format and the hallway terms are livingproof-evaluator's (v1.6.0). This module says which
MCAP channel and fields of this simulator's recording carry each term, and merges that with the
robot's own entries into the per-trial binding written next to the recording. The producer binding
is the same for every robot (G1, Go1, T1); the robot supplies only what only it knows: the base
body and the nominal base height.

An observable is bound only when the producer carries it and the robot supplies what it needs.
Anything else is unbound, with the evaluator's reason, and is never guessed.
"""

from __future__ import annotations

FORMAT = "livingproof-observation-binding/1.0.0"
PRODUCER_ID = "livingproof-sim-mujoco"
# The vocabulary's terms, in its order (livingproof-evaluator examples/vocabulary/hallway).
TERMS = ["base", "base_height", "body_contact"]
POSE_TOPIC = "/base_pose"
POSE_SCHEMA_NAME = "foxglove.PoseInFrame"
CONTACT_TOPIC = "/wall_contacts"
CONTACT_SCHEMA_NAME = "livingproof.WallContact"
PRODUCER_MISSING = "producer-does-not-carry"
ROBOT_MISSING = "robot-entry-missing"
# What a robot must supply for an observable, besides the producer's channel.
_ROBOT_NEEDS = {"base": ("body",), "base_height": ("nominal",), "body_contact": ()}


def _channel(topic: str, schema: str, fields: dict) -> dict:
    return {
        "topic": topic,
        "encoding": "json",
        "schema": schema,
        "fields": fields,
        "frame": "world",
    }


def producer_binding() -> dict:
    """Which channels carry which terms. Shared across robots."""
    return {
        "format": FORMAT,
        "kind": "producer",
        "producer": {"id": PRODUCER_ID},
        "observables": {
            "base": _channel(
                POSE_TOPIC,
                POSE_SCHEMA_NAME,
                {"x": "pose.position.x", "y": "pose.position.y"},
            ),
            "base_height": _channel(POSE_TOPIC, POSE_SCHEMA_NAME, {"z": "pose.position.z"}),
            # One message per contact event: {wall, start, end, depth}.
            "body_contact": _channel(
                CONTACT_TOPIC,
                CONTACT_SCHEMA_NAME,
                {"wall": "wall", "start": "start", "end": "end", "depth": "depth"},
            ),
        },
    }


def nominal_from(configuration: dict, ref: str) -> dict | None:
    """The configuration's nominal base height as a binding nominal, or None when it has none.

    The binding states meters; a configuration in another unit, or with a value that is not a
    positive number, offers no nominal rather than a converted or guessed one.
    """
    locomotion = configuration.get("locomotion") if isinstance(configuration, dict) else None
    nominal = locomotion.get("nominal_base_height") if isinstance(locomotion, dict) else None
    if not isinstance(nominal, dict) or nominal.get("unit") != "m":
        return None
    value = nominal.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return {"value": value, "unit": "meters", "source": "measured", "ref": ref}


def robot_binding(robot_id: str, configuration: str, base_body: str, nominal: dict | None) -> dict:
    """What only the robot knows: its base body and, when it has one, its nominal."""
    observables: dict = {"base": {"body": base_body}}
    if nominal is not None:
        observables["base_height"] = {"nominal": nominal}
    return {
        "format": FORMAT,
        "kind": "robot",
        "robot": {"id": robot_id},
        "configuration": configuration,
        "observables": observables,
    }


def merge(robot: dict, producer: dict, trial_id: str, recording: dict) -> dict:
    """The per-trial binding, in the vocabulary's order (the evaluator's reference rules)."""
    observables: dict = {}
    unbound: list = []
    for term in TERMS:
        carried = producer["observables"].get(term)
        supplied = robot["observables"].get(term, {})
        if carried is None:
            unbound.append({"id": term, "reason": PRODUCER_MISSING})
        elif any(need not in supplied for need in _ROBOT_NEEDS[term]):
            unbound.append({"id": term, "reason": ROBOT_MISSING})
        else:
            observables[term] = {**carried, **supplied}
    return {
        "format": FORMAT,
        "kind": "merged",
        "trial_id": trial_id,
        "robot": {"id": robot["robot"]["id"], "configuration": robot["configuration"]},
        "producer": {"id": producer["producer"]["id"]},
        "recording": recording,
        "observables": observables,
        "bound_observables": list(observables),
        "unbound": unbound,
    }
