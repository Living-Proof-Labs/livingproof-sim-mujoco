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

import re
import tempfile
import threading
from pathlib import Path

import mujoco
import numpy as np

from livingproof_sim_mujoco import hallway
from livingproof_sim_mujoco.pins import sha256
from livingproof_sim_mujoco.scene import add_world

# A base whose x and y are this close to 0 is at the world origin.
_AT_ORIGIN = 1e-6
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# Serializes the brief ``MjSpec.compile`` stand-in used to see which spec an
# adapter compiled. The stand-in calls the original function and records a spec
# only on the composing thread. It is removed before the lock is released.
_COMPILE_LOCK = threading.Lock()
_tls = threading.local()


def vendored_world(lock: dict) -> tuple[dict, str, str]:
    """``(world, sha256, relative path)`` for the lock's world id and version.

    The bytes are the vendored ``world.json``, after ``release.json`` and the
    world's MJCF mapping are checked. A lock ``world.sha256``, ``world.tag`` or
    ``world.commit`` that is not this file is an error. An id or version this
    package does not vendor is UNSUPPORTED, not a substitute.
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
    _require_same_release(declared, loaded["pin"], digest, fixture, version)
    return world, digest, f"worlds/{fixture}/{version}/world.json"


def _release_pairs(pin: dict, fixture: str, version: str) -> tuple[tuple[str, str], ...]:
    pairs = list(hallway.same_byte_releases(fixture, version))
    pin_pair = (pin.get("tag"), pin.get("commit"))
    if pin_pair not in pairs and all(isinstance(part, str) and part for part in pin_pair):
        pairs.append(pin_pair)
    return tuple(pairs)


def _release_error(field: str, value, pin: dict, digest: str, fixture: str, version: str) -> str:
    return (
        f"ERROR: lock {field} is {value!r} but the vendored {fixture}/{version} "
        f"world.json is {pin['tag']} {pin['commit']} ({digest}). The run would name a "
        f"different world from the one loaded. Remove {field} or set it to a release of "
        "that file. "
        f'Example: "world": {{"id": "{fixture}", "version": "{version}", '
        f'"tag": "{pin["tag"]}", "commit": "{pin["commit"]}"}}'
    )


def _require_same_release(
    declared: dict, pin: dict, digest: str, fixture: str, version: str
) -> None:
    """Refuse a tag or commit that is not a release of the vendored world file."""
    tag, commit = declared.get("tag"), declared.get("commit")
    if tag is None and commit is None:
        return
    pairs = _release_pairs(pin, fixture, version)
    if tag is not None and (not isinstance(tag, str) or not tag):
        raise SystemExit(_release_error("world.tag", tag, pin, digest, fixture, version))
    if commit is not None and (not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None):
        raise SystemExit(_release_error("world.commit", commit, pin, digest, fixture, version))
    if tag is not None and commit is not None:
        if (tag, commit) in pairs:
            return
        known_tags = {item[0] for item in pairs}
        known_commits = {item[1] for item in pairs}
        if tag not in known_tags:
            field = "world.tag" if commit in known_commits else "world.tag and world.commit"
        elif commit not in known_commits:
            field = "world.commit"
        else:
            field = "world.tag and world.commit"
        shown = (tag, commit) if "and" in field else (commit if field == "world.commit" else tag)
        raise SystemExit(_release_error(field, shown, pin, digest, fixture, version))
    if tag is not None and tag not in {item[0] for item in pairs}:
        raise SystemExit(_release_error("world.tag", tag, pin, digest, fixture, version))
    if commit is not None and commit not in {item[1] for item in pairs}:
        raise SystemExit(_release_error("world.commit", commit, pin, digest, fixture, version))


def _scene_and_spec(robot, name: str):
    """``(scene, spec, directory)`` for the adapter scene.

    ``MjSpec.compile`` does not register an XML document, so ``mj_saveLastXML``
    cannot export those models. The spec is recorded only for this thread, and
    only while ``_COMPILE_LOCK`` is held. The stand-in calls the original
    ``compile``, so another thread still receives the model it compiled. The
    original function is put back before the lock is released. A model loaded
    from XML is saved before the lock is released, because the last XML is
    process-global. ``directory`` holds that XML's meshes.
    """
    captured: list = []
    _tls.specs = captured
    spec = None
    directory = None
    try:
        with _COMPILE_LOCK:
            real = mujoco.MjSpec.compile

            def spy(self, *args, **kwargs):
                model = real(self, *args, **kwargs)
                bucket = getattr(_tls, "specs", None)
                if bucket is not None:
                    bucket.append(self)
                return model

            mujoco.MjSpec.compile = spy
            try:
                scene = robot.scene(name)
            finally:
                mujoco.MjSpec.compile = real
            # Last XML is process-global. Read it before releasing the lock.
            if captured:
                spec = captured[-1]
            else:
                spec, directory = _spec_from_xml_model(scene.model)
    finally:
        _tls.specs = None
    return scene, spec, directory


def _spec_from_xml_model(model: mujoco.MjModel):
    """``(spec, directory)`` for a model loaded from XML.

    The directory holds meshes written beside the XML. The caller compiles
    before cleaning it up.
    """
    directory = tempfile.TemporaryDirectory()
    xml = Path(directory.name) / "robot.xml"
    try:
        mujoco.mj_saveLastXML(str(xml), model)
    except mujoco.FatalError as exc:
        directory.cleanup()
        raise SystemExit(
            "INCOMPATIBLE: the robot model could not be saved for composition "
            f"({exc}). The world was not attached."
        ) from exc
    try:
        spec = mujoco.MjSpec.from_file(str(xml))
    except Exception:
        directory.cleanup()
        raise
    return spec, directory


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.allclose(left, right, rtol=0, atol=0, equal_nan=True))
    return bool(np.array_equal(left, right))


def _value_change(label: str, left, right) -> str | None:
    """One compiled value ``label`` that a recompile would replace, or None."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not _arrays_equal(left, right):
            return label
        return None
    if callable(left) or callable(right):
        return None
    if isinstance(left, (bool, np.bool_)):
        if bool(left) != bool(right):
            return f"{label} is {bool(left)} and a recompile is {bool(right)}"
        return None
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        if left != right:
            return f"{label} is {left} and a recompile is {right}"
    return None


def _model_change(original: mujoco.MjModel, recompiled: mujoco.MjModel) -> str | None:
    """A compiled value the recompile would drop, or None when the models match.

    Recompiling the spec loses edits made on the MjModel after ``compile``.
    Those edits are a different simulation from the one the adapter built.
    """
    for name in dir(original.opt):
        if name.startswith("_"):
            continue
        change = _value_change(
            f"opt.{name}", getattr(original.opt, name), getattr(recompiled.opt, name)
        )
        if change:
            return change
    for name in dir(original):
        if name.startswith("_"):
            continue
        change = _value_change(name, getattr(original, name), getattr(recompiled, name))
        if change:
            return change
    return None


def attach_world(
    model: mujoco.MjModel,
    world: dict,
    spec: mujoco.MjSpec | None = None,
    directory: tempfile.TemporaryDirectory | None = None,
):
    """Return ``(model, wall names)`` with the world's walls and places added.

    ``spec`` is the spec the adapter compiled, or one rebuilt from its XML.
    ``directory`` holds meshes for that XML and is removed after compile.
    A recompile that would drop a change made on the compiled model is
    INCOMPATIBLE.
    """
    if spec is None:
        spec, directory = _spec_from_xml_model(model)
    try:
        change = _model_change(model, spec.compile())
        if change:
            raise SystemExit(
                "INCOMPATIBLE: the compiled robot model differs from a recompile of its "
                f"spec ({change}). Composing the world would drop that change. No trial "
                "was recorded. Set the value on the MjSpec before compile. "
                "Example: spec.option.timestep = 0.001, then spec.compile()"
            )
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

    scene, spec, directory = _scene_and_spec(robot, name)
    model, walls = attach_world(scene.model, world, spec, directory)
    scene.model = model
    ctrl = robot.controller(name, scene)
    data = mujoco.MjData(model)
    ev.nominal_start(robot, model, data, ctrl, spawn, start_offset)
    require_composed(model, data, world, spawn, robot.base_body(name))
    return model, data, ctrl, walls
