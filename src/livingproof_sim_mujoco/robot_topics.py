# SPDX-License-Identifier: Apache-2.0
"""The adapter hook for publishing robot topics into the trial MCAP (#107).

A robot adapter may define ``publish(name, model, step) -> iterable of Message``. It is no-op by
default: an adapter without it records exactly the simulator's own topics. The recorder calls it
at its sample steps (``media.MCAP_HZ``) on every trial path, after the physics step, with a
read-only :class:`StepView` snapshot, never ``data``, and a copy of the model made once per
trial. The adapter does not write the MCAP and cannot change the trial: what it does to its copy
of the model is not seen by the simulation, and the media re-run still hashes to the evidence.

Each :class:`Message` names its topic, the schema (name and a JSON Schema document) and a JSON
payload. The recorder writes it at the step's simulation time, adds ``timestamp`` to the payload when
it has none, and keeps the file in log_time order.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np

from livingproof_sim_mujoco import binding

# The simulator's own channels; an adapter cannot write to them.
RESERVED = frozenset({binding.POSE_TOPIC, "/joint_states", binding.CONTACT_TOPIC})


@dataclass(frozen=True)
class StepView:
    """A read-only copy of what an adapter may look at after a step."""

    time: float
    xpos: np.ndarray
    xquat: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass(frozen=True)
class Message:
    topic: str
    schema_name: str
    schema: dict
    payload: dict


def snapshot(data) -> StepView:
    """Copies of the step's state, marked read-only."""
    arrays = []
    for source in (data.xpos, data.xquat, data.qpos, data.qvel):
        copy = np.array(source, copy=True)
        copy.flags.writeable = False
        arrays.append(copy)
    return StepView(float(data.time), *arrays)


def publisher(robot, name: str, model):
    """The adapter's hook bound to a configuration, or None when it defines none.

    Called once per trial: the hook gets one copy of the model for the whole trial, not the model
    the simulation runs on, so a change the adapter makes to it cannot reach the trial."""
    hook = getattr(robot, "publish", None)
    if not callable(hook):
        return None
    own = copy.copy(model)
    return lambda step: hook(name, own, step)


def items(returned) -> list:
    """What a hook returned as a list of items: nothing, or an iterable of Message.

    A single Message (or any other value that is not an iterable) is a ValueError, not a
    TypeError from iterating it."""
    if returned is None:
        return []
    if isinstance(returned, (Message, str, bytes, dict)):
        raise ValueError(
            "publish must return a list (or other iterable) of robot_topics.Message, "
            f"not a single {type(returned).__name__}; wrap it in a list"
        )
    try:
        return list(returned)
    except TypeError:
        raise ValueError(
            f"publish must return a list (or other iterable) of robot_topics.Message, got {returned!r}"
        ) from None


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def check(message: Message) -> Message:
    """The message, or ValueError saying why it cannot be recorded."""
    if not isinstance(message, Message):
        raise ValueError(f"a published item must be a robot_topics.Message, got {message!r}")
    topic = message.topic
    if not isinstance(topic, str) or not topic.startswith("/") or len(topic) < 2:
        raise ValueError(f"topic {topic!r} must be a string that starts with '/'")
    if topic in RESERVED:
        raise ValueError(
            f"topic {topic} is reserved for the simulator's own recording ({sorted(RESERVED)})"
        )
    if not isinstance(message.schema_name, str) or not message.schema_name:
        raise ValueError(f"topic {topic}: schema_name must be a non-empty string")
    if not isinstance(message.schema, dict) or not isinstance(message.payload, dict):
        raise ValueError(f"topic {topic}: schema and payload must be JSON objects")
    if not _finite(message.payload):
        raise ValueError(f"topic {topic}: payload must be finite JSON (no NaN or Infinity)")
    try:
        json.dumps(message.schema)
        json.dumps(message.payload)
    except TypeError as exc:
        raise ValueError(f"topic {topic}: schema and payload must be JSON ({exc})") from None
    return message
