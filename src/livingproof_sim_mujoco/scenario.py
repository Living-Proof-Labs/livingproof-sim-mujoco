"""walk-hallway@0.1.0 parameters and preconditions from a robot profile, configuration and
world. Evidence only: grading belongs to the capability verifier."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCENARIO = "walk-hallway@0.1.0"
COMMAND_SPEED = 0.5  # m/s, shared
WALL_CLEARANCE = {"humanoid": 0.05, "quadruped": 0.10}  # m, walk-hallway README
SPAWN, EXIT = "spawn-entrance", "exit"


def place(world: dict, name: str) -> dict:
    return next(p for p in world["places"] if p["name"] == name)


def parameters(robot: dict, conf: dict, world: dict) -> dict:
    """Acceptance parameters resolved from the scenario, robot, configuration and world."""
    route = place(world, EXIT)["reference"][0] - place(world, SPAWN)["position"][0]
    clear = world["dimensions"]["clear_width"]
    width = robot["dimensions"]["width"]["value"]
    return {
        "command_speed": COMMAND_SPEED,
        "route_length": route,
        "time_limit": 2 * route / COMMAND_SPEED + 5,
        "fall_height": conf["locomotion"]["fall_base_height"]["value"],
        "wall_clearance": WALL_CLEARANCE[robot["class"]],
        "lateral_limit": round(clear / 2 - width / 2, 6),
        "exit_x": place(world, EXIT)["reference"][0],
        "spawn": SPAWN,
    }


def preconditions(
    robot: dict, world: dict, name: str, configuration, *, width_decimals: int = 2
) -> list:
    """Checked before any trial and recorded in the evidence manifest.

    ``configuration(name)`` returns the configuration profile or raises an exception carrying
    ``precondition`` and ``detail`` (the robot package's ``Unsupported``).
    """
    width = robot["dimensions"]["width"]["value"]
    clear = world["dimensions"]["clear_width"]
    need = width + 2 * WALL_CLEARANCE[robot["class"]]
    out = [
        (
            "fits_width",
            need <= clear,
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
