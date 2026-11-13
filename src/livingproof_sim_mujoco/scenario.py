"""Scenario parameters and preconditions from a robot profile, configuration and world.

Limits, the command and ``fits_width`` come from ``livingproof_scenario_parameters``.
The formulas stay in the scenario parameter file. Evidence only: grading belongs
to the capability verifier.

``COMMAND_SPEED``, ``SCENARIO``, ``WALL_CLEARANCE``, ``EXIT`` and ``SPAWN`` stay
exported so robot packages that already import them keep importing.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

from livingproof_scenario_parameters import (
    ScenarioParameterError,
    evaluate,
    quantize_half_up_decimal_text,
    read_inputs,
    resolve,
)

from livingproof_sim_mujoco.compat.hallway import (
    COMMAND_SPEED,
    EXIT,
    SCENARIO,
    SPAWN,
    WALL_CLEARANCE,
    parameters_document,
    recorded_limit,
)

__all__ = [
    "COMMAND_SPEED",
    "EXIT",
    "SCENARIO",
    "SPAWN",
    "WALL_CLEARANCE",
    "ScenarioParameterError",
    "command_vector",
    "fits_width",
    "git_head",
    "millimetres",
    "parameters",
    "place",
    "preconditions",
    "prepare",
    "resolve_limits",
]


class Prepared:
    """Resolved scenario: the recorded parameter dict and the velocity command."""

    def __init__(self, recorded: dict, command: list[float], resolved: dict, document: dict):
        self.recorded = recorded
        self.command = command
        self.resolved = resolved
        self.document = document


def millimetres(metres: float) -> int:
    """Metres to integer millimetres, half up from the decimal text."""
    return quantize_half_up_decimal_text(metres, 3)


def _document(document: dict | None) -> dict:
    return parameters_document() if document is None else document


def fits_width(
    width: float, clearance: float, clear_width: float, document: dict | None = None
) -> bool:
    """The parameter file's ``fits_width`` precondition, via the package.

    ``clearance`` replaces the file's class table for this call so a test can
    pass the clearance directly. The comparison expression is still the file's.
    """
    doc = copy.deepcopy(_document(document))
    doc["parameters"]["wall_clearance"] = {"value": clearance, "unit": "m"}
    raw = {
        "route_length": 1.0,
        "clear_width": clear_width,
        "robot_width": {"value": width, "unit": "m"},
        "robot_class": "humanoid",
        "fall_base_height": {"value": 1.0, "unit": "m"},
    }
    return bool(resolve(doc, raw)["preconditions"]["fits_width"])


def resolve_limits(document: dict, raw: dict) -> dict:
    """Spec-rounded limits and preconditions. Same resolver ``prepare`` uses."""
    return resolve(document, raw)


def command_vector(document: dict, inputs: dict) -> list[float]:
    """``[forward, sideways, turn_rate]`` from the parameter file's command.

    Only ``mode`` ``constant`` is accepted. Any other mode is an ERROR.
    """
    command = document.get("command") if isinstance(document, dict) else None
    mode = command.get("mode") if isinstance(command, dict) else None
    if not isinstance(command, dict) or mode != "constant":
        raise ScenarioParameterError(
            f"ERROR: command mode {mode!r} is not supported. Expected mode 'constant'. "
            'Example: "command": {"mode": "constant", '
            '"forward": {"parameter": "command_speed"}, "sideways": 0, "turn_rate": 0}'
        )
    return [
        float(evaluate(command["forward"], document, inputs)),
        float(evaluate(command["sideways"], document, inputs)),
        float(evaluate(command["turn_rate"], document, inputs)),
    ]


def place(world: dict, name: str) -> dict:
    return next(p for p in world["places"] if p["name"] == name)


def route_geometry(world: dict, document: dict) -> tuple[list, dict] | None:
    """Route polyline and exit segment from the world, named by the parameter file."""
    names = {p["name"]: p for p in world.get("places", [])}
    route = names.get(document["places"]["route"])
    exit_place = names.get(document["places"]["exit"])
    if not route or not route.get("points") or not exit_place or not exit_place.get("points"):
        return None
    return route["points"], {
        "points": exit_place["points"],
        "reference": exit_place.get("reference"),
    }


def prepare(robot: dict, conf: dict, world: dict, document: dict | None = None) -> Prepared:
    """Resolve the parameter file against the robot, configuration and world."""
    document = _document(document)
    raw = read_inputs(document, {"world": world, "robot": robot, "configuration": conf})
    resolved = resolve(document, raw)
    inputs = resolved["inputs"]
    raw_time = evaluate(document["limits"]["time_limit"]["expression"], document, inputs)
    raw_lateral = evaluate(document["limits"]["lateral_limit"]["expression"], document, inputs)
    exit_place = place(world, document["places"]["exit"])
    reference = exit_place.get("reference") or exit_place["points"][0]
    recorded = {
        "command_speed": float(evaluate({"parameter": "command_speed"}, document, inputs)),
        "route_length": inputs["route_length"],
        "time_limit": recorded_limit(
            document, "time_limit", raw_time, resolved["limits"]["time_limit"]
        ),
        "fall_height": resolved["limits"]["fall_height"],
        "wall_clearance": float(evaluate({"parameter": "wall_clearance"}, document, inputs)),
        "lateral_limit": recorded_limit(
            document, "lateral_limit", raw_lateral, resolved["limits"]["lateral_limit"]
        ),
        "exit_x": float(reference[0]),
        "spawn": document["places"]["start"],
    }
    return Prepared(recorded, command_vector(document, inputs), resolved, document)


def parameters(robot: dict, conf: dict, world: dict, document: dict | None = None) -> dict:
    """Recorded parameters. The frozen scenario keeps its historical rounding."""
    return prepare(robot, conf, world, document).recorded


def preconditions(
    robot: dict,
    world: dict,
    name: str,
    configuration,
    document: dict | None = None,
    *,
    width_decimals: int = 2,
) -> list:
    """Checked before any trial and recorded in the evidence manifest.

    ``fits_width`` is the parameter file's precondition. The detail text is the
    frozen evidence sentence. ``configuration(name)`` returns the configuration
    profile or raises an exception carrying ``precondition`` and ``detail``.
    """
    document = _document(document)
    width = robot["dimensions"]["width"]["value"]
    clear = world["dimensions"]["clear_width"]
    clearance = float(
        evaluate(
            {"parameter": "wall_clearance"},
            document,
            {"robot_class": robot["class"]},
        )
    )
    need = width + 2 * clearance
    # The boolean is the parameter file's quantised comparison. The sentence is
    # the frozen evidence detail, which prints the unquantised sum.
    out = [
        (
            "fits_width",
            fits_width(width, clearance, clear, document),
            f"{width} + 2 x clearance = {need:.{width_decimals}f} m vs {clear} m",
        )
    ]
    try:
        conf = configuration(name)
        out.append(("configuration_offered", True, name))
        iface = conf["controller"]["interface"]
        out.append(("velocity_interface", "velocity_command" in iface, f"interface {iface}"))
    except Exception as e:  # the robot package's Unsupported
        if not hasattr(e, "precondition"):
            raise
        out.append((e.precondition, False, e.detail))
    out.append(
        ("simulator_declared", "mujoco" in robot["simulators"], f"simulators {robot['simulators']}")
    )
    return [{"name": n, "met": bool(ok), "detail": d} for n, ok, d in out]


def git_head(path: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--", ".", ":!artifacts"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return head, dirty
