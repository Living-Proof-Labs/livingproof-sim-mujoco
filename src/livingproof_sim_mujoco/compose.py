# SPDX-License-Identifier: Apache-2.0
"""Attach a vendored world to a robot model.

The caller names the world (id and version on the verification lock). This
module loads that release from the files this package vendors, checks
``release.json``, and adds the world's wall boxes and named places to the
robot model. The robot package does not supply them.

Wall boxes are the world's own geometry (the same centres and half-sizes
``hallway.to_mjcf`` writes). They stay out of the contact solver, which is how
clearance has been measured: ``mj_geomDistance`` against those geoms. Place
sites use the ``place:`` prefix ``initial_state`` reads.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco

from livingproof_sim_mujoco import hallway
from livingproof_sim_mujoco.pins import sha256
from livingproof_sim_mujoco.scene import add_world

# A base whose x and y are this close to 0 is at the world origin.
_AT_ORIGIN = 1e-6


def vendored_world(lock: dict) -> tuple[dict, str, str]:
    """``(world, sha256, relative path)`` for the lock's world id and version.

    The bytes are the vendored ``world.json``, after ``release.json`` and the
    world's MJCF mapping are checked. A lock ``world.sha256`` that disagrees
    is an error. An id or version this package does not vendor is
    UNSUPPORTED, not a substitute.
    """
    declared = lock.get("world") if isinstance(lock, dict) else None
    if not isinstance(declared, dict):
        raise SystemExit(
            "INCOMPATIBLE: the verification lock needs a world object with id and version. "
            'Example: "world": {"id": "<world>", "version": "0.1.0"}'
        )
    fixture, version = declared.get("id"), declared.get("version")
    if not isinstance(fixture, str) or not fixture or not isinstance(version, str) or not version:
        raise SystemExit(
            "INCOMPATIBLE: the verification lock world needs id and version. "
            'Example: "world": {"id": "<world>", "version": "0.1.0"}'
        )
    try:
        loaded = hallway.load_fixture(fixture, version)
    except hallway.FixtureError as exc:
        raise SystemExit(f"UNSUPPORTED: {exc}") from exc
    world, extension = loaded["world"], loaded["extension"]
    try:
        # Reject a world this adapter cannot map before any robot is moved.
        hallway.to_mjcf(world, extension)
    except hallway.Unsupported as exc:
        raise SystemExit(str(exc)) from exc
    path = hallway.WORLDS / fixture / version / "world.json"
    digest = sha256(path)
    named = declared.get("sha256")
    if named is not None and named != digest:
        raise SystemExit(
            f"ERROR: lock world.sha256 is {named} but the vendored {fixture}/{version} "
            f"world.json is {digest}. The run would name a different world from the one "
            "loaded. Remove world.sha256 or set it to the vendored file."
        )
    return world, digest, f"worlds/{fixture}/{version}/world.json"


def _scene_and_spec(robot, name: str):
    """The adapter scene and the MjSpec it compiled, if it used one.

    ``MjSpec.compile`` does not register an XML document, so ``mj_saveLastXML``
    cannot export those models. The spec the adapter just compiled is the robot
    model, and the world is added to that spec.
    """
    captured: list = []
    real = mujoco.MjSpec.compile

    def spy(self, *args, **kwargs):
        captured.append(self)
        return real(self, *args, **kwargs)

    mujoco.MjSpec.compile = spy
    try:
        scene = robot.scene(name)
    finally:
        mujoco.MjSpec.compile = real
    return scene, captured[-1] if captured else None


def _spec_from_xml_model(model: mujoco.MjModel):
    """``(spec, directory)`` for a model loaded from XML.

    The directory holds meshes written beside the XML. The caller compiles
    before cleaning it up.
    """
    directory = tempfile.TemporaryDirectory()
    xml = Path(directory.name) / "robot.xml"
    try:
        saved = mujoco.mj_saveLastXML(str(xml), model)
    except mujoco.FatalError as exc:
        directory.cleanup()
        raise SystemExit(
            "INCOMPATIBLE: the robot model could not be saved for composition "
            f"({exc}). The world was not attached."
        ) from exc
    if saved is not None and not saved:
        directory.cleanup()
        raise SystemExit(
            "INCOMPATIBLE: the robot model could not be saved for composition "
            "(mj_saveLastXML failed). The world was not attached."
        )
    try:
        spec = mujoco.MjSpec.from_file(str(xml))
    except Exception:
        directory.cleanup()
        raise
    return spec, directory


def attach_world(model: mujoco.MjModel, world: dict, spec: mujoco.MjSpec | None = None):
    """Return ``(model, wall names)`` with the world's walls and places added.

    ``spec`` is the spec the adapter compiled. Without one, the model must be a
    loaded XML document.
    """
    directory = None
    if spec is None:
        spec, directory = _spec_from_xml_model(model)
    try:
        walls = add_world(spec, world)
        compiled = spec.compile()
    finally:
        if directory is not None:
            directory.cleanup()
    return compiled, walls


def wall_names(world: dict) -> list[str]:
    return [g["name"] for g in world.get("geometry", []) if g.get("role") == "wall"]


def require_composed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    world: dict,
    spawn: str,
    base_body: str,
) -> None:
    """Refuse a model with no declared walls, or a start at the origin.

    A scenario whose world has walls and a composed model that is missing
    any of them is INCOMPATIBLE. So is a robot base at the origin when the
    named start place is not the origin. Neither case is recorded as a pass.
    """
    names = wall_names(world)
    if names:
        missing = [
            name for name in names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
        ]
        if missing:
            raise SystemExit(
                "INCOMPATIBLE: the world declares walls "
                f"{', '.join(names)} but the composed model is missing {', '.join(missing)}. "
                "No trial was recorded. The simulator adds those geoms from the vendored "
                "world. Example: a lock world whose id and version are a release this "
                "package vendors."
            )
    place = next((p for p in world.get("places", []) if p.get("name") == spawn), None)
    if not isinstance(place, dict) or place.get("type") != "pose" or "position" not in place:
        raise SystemExit(
            f"INCOMPATIBLE: start place {spawn!r} is not a pose in the vendored world. "
            "No trial was recorded. The scenario start must name a pose in that world."
        )
    pos = place["position"]
    place_at_origin = abs(float(pos[0])) <= _AT_ORIGIN and abs(float(pos[1])) <= _AT_ORIGIN
    base = data.xpos[model.body(base_body).id]
    at_origin = abs(float(base[0])) <= _AT_ORIGIN and abs(float(base[1])) <= _AT_ORIGIN
    if not place_at_origin and at_origin:
        raise SystemExit(
            "INCOMPATIBLE: start place "
            f"{spawn} is at ({float(pos[0])}, {float(pos[1])}) but the robot base is at "
            "the origin. No trial was recorded. initial_state must use the place compiled "
            "into the model, not the nominal pose at the origin. "
            "Example: initial_state(model, data, ctrl, 'spawn-entrance')"
        )


def place(robot, name: str, world: dict, spawn: str, start_offset=None):
    """Compose the world, place the robot, and refuse a missing wall or an origin start.

    Returns ``(model, data, ctrl, wall names)``. The policy is the caller's:
    this function does not step.
    """
    from livingproof_sim_mujoco import evidence as ev

    scene, spec = _scene_and_spec(robot, name)
    model, walls = attach_world(scene.model, world, spec)
    scene.model = model
    ctrl = robot.controller(name, scene)
    data = mujoco.MjData(model)
    ev.nominal_start(robot, model, data, ctrl, spawn, start_offset)
    require_composed(model, data, world, spawn, robot.base_body(name))
    return model, data, ctrl, walls
