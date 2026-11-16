"""Scenario parameters and preconditions from a robot profile, configuration and world.

Limits, the command and ``fits_width`` come from ``livingproof_scenario_parameters``.
The formulas stay in the scenario parameter file. Evidence only: grading belongs
to the capability verifier.

``COMMAND_SPEED``, ``SCENARIO``, ``WALL_CLEARANCE``, ``EXIT`` and ``SPAWN`` stay
exported so robot packages that already import them keep importing.
"""

from __future__ import annotations

import copy
import json
import re
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
    "requested_scenario_id",
    "resolve_limits",
    "scenario_support_required",
    "scenario_supported_precondition",
]


class Prepared:
    """Resolved scenario: the recorded parameter dict and the velocity command."""

    def __init__(self, recorded: dict, command: list[float], resolved: dict, document: dict):
        self.recorded = recorded
        self.command = command
        self.resolved = resolved
        self.document = document
        # Set by evidence.prepare_run: the world document that was loaded and
        # checked, and its sources.WorldSource. The manifests name that world.
        self.world: dict | None = None
        self.world_source = None


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


# The only command the frozen hallway scenario accepts (#40 step 3).
HALLWAY_COMMAND = (0.5, 0.0, 0.0)


def command_vector(document: dict, inputs: dict) -> list[float] | None:
    """``[forward, sideways, turn_rate]`` from the parameter file's command.

    Only ``mode`` ``constant`` with the vector ``[0.5, 0, 0]`` is accepted.
    Any other mode or vector is an ERROR. A parameter file with no ``command``
    returns None: the scenario states the outcome, and this runner does not
    invent a velocity command.
    """
    if not isinstance(document, dict) or "command" not in document:
        return None
    command = document.get("command")
    mode = command.get("mode") if isinstance(command, dict) else None
    if not isinstance(command, dict) or mode != "constant":
        raise ScenarioParameterError(
            f"ERROR: command mode {mode!r} is not supported. Expected mode 'constant'. "
            'Example: "command": {"mode": "constant", '
            '"forward": {"parameter": "command_speed"}, "sideways": 0, "turn_rate": 0}'
        )
    vector = [
        float(evaluate(command["forward"], document, inputs)),
        float(evaluate(command["sideways"], document, inputs)),
        float(evaluate(command["turn_rate"], document, inputs)),
    ]
    if vector != list(HALLWAY_COMMAND):
        raise ScenarioParameterError(
            f"ERROR: command {vector} is not supported. The frozen hallway runs only the "
            f"constant command {list(HALLWAY_COMMAND)} (forward m/s, sideways m/s, turn rate rad/s), "
            "so evidence from any other command could not be told apart from hallway evidence. "
            'Example: "command_speed": {"value": 0.5}, "sideways": 0, "turn_rate": 0'
        )
    return vector


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


# livingproof-robot-manifest/0.1. Robot packages publish this at the repository root.
ROBOT_MANIFEST = "manifest.json"
MANIFEST_FORMAT = "livingproof-robot-manifest/0.1"
_SCENARIO_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NO_LIST = "the robot manifest has no scenarios list"


def scenario_support_required(document: dict) -> bool:
    """Whether this scenario lists the precondition ``scenario_supported``.

    robot-capability-specs v2.3.0, Scenario support: the check applies only to
    a scenario that lists that precondition. A parameter file with no
    ``command`` is outcome-only, and only an adapter whose robot manifest
    declares the scenario is run. A ``preconditions`` entry of the same name
    says so too. A scenario that still has a command and does not name the
    precondition is not checked, so a manifest with no list keeps running it.
    """
    if not isinstance(document, dict):
        return False
    named = document.get("preconditions")
    if isinstance(named, dict) and "scenario_supported" in named:
        return True
    return "command" not in document


def requested_scenario_id(lock: dict, document: dict) -> str:
    """Scenario id of this run, without the ``@revision`` suffix."""
    raw = lock.get("scenario") if isinstance(lock, dict) else None
    if isinstance(raw, dict):
        raw = raw.get("id")
    if isinstance(raw, str):
        ident = raw.split("@", 1)[0]
        if _SCENARIO_ID.fullmatch(ident):
            return ident
    scenario = document.get("scenario") if isinstance(document, dict) else None
    ident = scenario.get("id") if isinstance(scenario, dict) else None
    if isinstance(ident, str) and _SCENARIO_ID.fullmatch(ident):
        return ident
    raise SystemExit(
        'ERROR: the requested scenario id is missing. Example: "scenario": "<scenario>@0.1.0"'
    )


def _read_robot_manifest(root: Path) -> dict | None:
    """The robot manifest, or None when the repository has no manifest file.

    A missing file names no scenarios. Invalid manifest data is a test-system
    ERROR: it is never read as supported or unsupported.
    """
    path = Path(root) / ROBOT_MANIFEST
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ERROR: {ROBOT_MANIFEST} is not JSON ({exc}). "
            "Invalid manifest data is a test-system ERROR, never UNSUPPORTED. "
            f'Example: {{"format": "{MANIFEST_FORMAT}", "scenarios": ["<scenario>"]}}'
        ) from exc
    if not isinstance(document, dict) or document.get("format") != MANIFEST_FORMAT:
        raise SystemExit(
            f"ERROR: {ROBOT_MANIFEST} must be a {MANIFEST_FORMAT} object. "
            "Invalid manifest data is a test-system ERROR, never UNSUPPORTED. "
            f'Example: {{"format": "{MANIFEST_FORMAT}", "scenarios": ["<scenario>"]}}'
        )
    if "scenarios" not in document:
        return document
    scenarios = document["scenarios"]
    valid = (
        isinstance(scenarios, list)
        and len(scenarios) == len(set(scenarios))
        and all(isinstance(item, str) and _SCENARIO_ID.fullmatch(item) for item in scenarios)
    )
    if not valid:
        raise SystemExit(
            f"ERROR: {ROBOT_MANIFEST} scenarios must be unique scenario ids, "
            f"got {scenarios!r}. Invalid manifest data is a test-system ERROR, "
            'never UNSUPPORTED. Example: "scenarios": ["<scenario>"]'
        )
    return document


def scenario_supported_precondition(root: Path, scenario_id: str) -> dict:
    """``scenario_supported`` against the robot manifest's ``scenarios`` list.

    A list that does not name ``scenario_id``, an empty list, or a manifest
    with no ``scenarios`` list is unmet. The spec counts a missing list as
    naming nothing. A missing manifest file is that same case: the adapter
    has not declared the scenario.
    """
    manifest = _read_robot_manifest(root)
    if manifest is None or "scenarios" not in manifest:
        return {"name": "scenario_supported", "met": False, "detail": _NO_LIST}
    scenarios = manifest["scenarios"]
    if scenario_id in scenarios:
        return {"name": "scenario_supported", "met": True, "detail": scenario_id}
    return {
        "name": "scenario_supported",
        "met": False,
        "detail": f"{scenario_id} is not named by the robot manifest scenarios",
    }


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
