# SPDX-License-Identifier: Apache-2.0
"""Load a pinned livingproof-worlds fixture into MuJoCo and run the hallway conformance checks.

The world declaration (world.json) is the only source of geometry and named places.
MuJoCo-only settings live in a small per-world extension file and never override them.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from xml.sax.saxutils import quoteattr

WORLDS = Path(__file__).parent / "worlds"
RESULT_FORMAT = "livingproof-world-result/0.1"
SUPPORTED_FORMAT = "livingproof-world/0.1"
GEOMETRY_KEYS = {"name", "type", "role", "centre", "size"}
PLACE_KEYS = {
    "segment": {"name", "type", "points", "reference"},
    "region": {"name", "type", "x", "y"},
    "polyline": {"name", "type", "points"},
    "pose": {"name", "type", "position", "yaw_deg"},
}
WORLD_KEYS = {
    "format",
    "id",
    "content_version",
    "title",
    "description",
    "specification",
    "units",
    "frame",
    "dimensions",
    "geometry",
    "places",
    "tolerances",
    "provenance",
    "license",
}


class FixtureError(Exception):
    """The fixture is unknown, not pinned, or does not match its released checksums."""


class Unsupported(Exception):
    """A declared feature has no MuJoCo mapping; never ignore it."""

    def __init__(self, fixture: str, version_: str, feature: str):
        self.fixture, self.version, self.feature = fixture, version_, feature
        super().__init__(f"UNSUPPORTED: {fixture} {version_}: {feature}")

    def record(self) -> dict:
        return {
            "status": "UNSUPPORTED",
            "fixture": self.fixture,
            "content_version": self.version,
            "feature": self.feature,
        }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins() -> dict:
    return json.loads((WORLDS / "pins.json").read_text())


def load_fixture(fixture: str, content_version: str, root: Path = WORLDS) -> dict:
    """Return the pinned world after verifying release.json and every file it lists."""
    key = f"{fixture}/{content_version}"
    pin = pins().get(key)
    if pin is None:
        raise FixtureError(
            f"unknown fixture {key}: pinned fixtures are {', '.join(sorted(pins()))}; "
            "an unknown identifier or version is reported, not substituted"
        )
    directory = root / fixture / content_version
    release = directory / "release.json"
    if not release.is_file():
        raise FixtureError(f"{key}: missing {release}")
    actual = sha256(release)
    if actual != pin["release_sha256"]:
        raise FixtureError(
            f"{key}: release.json checksum mismatch: expected {pin['release_sha256']} "
            f"(livingproof-worlds {pin['tag']} {pin['commit']}), got {actual}"
        )
    manifest = json.loads(release.read_text())
    if (manifest.get("id"), manifest.get("content_version")) != (fixture, content_version):
        raise FixtureError(
            f"{key}: release.json names {manifest.get('id')} {manifest.get('content_version')}"
        )
    for name, expected in sorted(manifest["files"].items()):
        path = directory / name
        if not path.is_file():
            raise FixtureError(f"{key}: missing released file {name}")
        got = sha256(path)
        if got != expected:
            raise FixtureError(
                f"{key}: checksum mismatch for {name}: expected {expected}, got {got}"
            )
    extras = sorted(p.name for p in directory.iterdir() if p.name not in manifest["files"])
    if extras != ["release.json"]:
        raise FixtureError(
            f"{key}: files not in release.json: {', '.join(e for e in extras if e != 'release.json')}"
        )
    world = json.loads((directory / "world.json").read_text())
    extension = json.loads((root / pin["extension"]).read_text())
    return {"world": world, "extension": extension, "pin": pin}


def supported(world: dict) -> None:
    """Raise Unsupported for the first declared feature this adapter cannot map."""
    fid, ver = world.get("id", "?"), world.get("content_version", "?")
    if world.get("format") != SUPPORTED_FORMAT:
        raise Unsupported(fid, ver, f"world format {world.get('format')!r}")
    for key in sorted(set(world) - WORLD_KEYS):
        raise Unsupported(fid, ver, f"top-level field {key!r}")
    if world.get("units") != "m":
        raise Unsupported(fid, ver, f"units {world.get('units')!r}")
    for g in world["geometry"]:
        if g.get("type") != "box":
            raise Unsupported(fid, ver, f"geometry {g.get('name')!r} type {g.get('type')!r}")
        if g.get("role") not in ("floor", "wall"):
            raise Unsupported(fid, ver, f"geometry {g.get('name')!r} role {g.get('role')!r}")
        for key in sorted(set(g) - GEOMETRY_KEYS):
            raise Unsupported(fid, ver, f"geometry {g.get('name')!r} field {key!r}")
    for p in world["places"]:
        allowed = PLACE_KEYS.get(p.get("type"))
        if allowed is None:
            raise Unsupported(fid, ver, f"place {p.get('name')!r} type {p.get('type')!r}")
        for key in sorted(set(p) - allowed):
            raise Unsupported(fid, ver, f"place {p.get('name')!r} field {key!r}")


def _v(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def yaw_quat(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def to_mjcf(world: dict, extension: dict, probe: bool = False) -> str:
    """Translate world.json to MJCF. Geometry is copied verbatim; places become sites."""
    supported(world)
    opt, con = extension["option"], extension["contact"]
    out = [
        f"<mujoco model={quoteattr(world['id'] + '-' + world['content_version'])}>",
        '  <compiler angle="radian" autolimits="true"/>',
        f'  <option timestep="{opt["timestep"]}" integrator="{opt["integrator"]}" '
        f'gravity="{_v(opt["gravity"])}" cone="{opt["cone"]}"/>',
        "  <default>",
        f'    <geom solref="{_v(con["solref"])}" solimp="{_v(con["solimp"])}" '
        f'friction="{_v(con["friction"])}" condim="{con["condim"]}" margin="{con["margin"]}"/>',
        '    <site group="4" rgba="1 0.5 0 0.6"/>',
        "  </default>",
        '  <visual><global offwidth="1280" offheight="960"/></visual>',
        "  <worldbody>",
        '    <light pos="0 0 6" dir="0 0 -1" directional="true"/>',
    ]
    colours = {"floor": "0.80 0.82 0.85 1", "wall": "0.45 0.50 0.58 1"}
    for g in world["geometry"]:
        half = [s / 2 for s in g["size"]]
        out.append(
            f'    <geom name={quoteattr(g["name"])} type="box" pos="{_v(g["centre"])}" '
            f'size="{_v(half)}" rgba="{colours[g["role"]]}"/>'
        )
    for p in world["places"]:
        n, t = p["name"], p["type"]
        if t == "segment":
            out.append(f'    <site name={quoteattr(n)} pos="{_v(p["reference"])}" size="0.03"/>')
            for i, pt in enumerate(p["points"]):
                out.append(f'    <site name={quoteattr(f"{n}/{i}")} pos="{_v(pt)}" size="0.02"/>')
        elif t == "region":
            centre = [(p["x"][0] + p["x"][1]) / 2, (p["y"][0] + p["y"][1]) / 2, 0.0]
            half = [(p["x"][1] - p["x"][0]) / 2, (p["y"][1] - p["y"][0]) / 2, 0.001]
            out.append(
                f'    <site name={quoteattr(n)} type="box" pos="{_v(centre)}" size="{_v(half)}" '
                'rgba="0.1 0.6 0.5 0.25"/>'
            )
        elif t == "polyline":
            for i, pt in enumerate(p["points"]):
                out.append(f'    <site name={quoteattr(f"{n}/{i}")} pos="{_v(pt)}" size="0.02"/>')
        elif t == "pose":
            out.append(
                f'    <site name={quoteattr(n)} type="box" pos="{_v(p["position"])}" '
                f'quat="{_v(yaw_quat(p["yaw_deg"]))}" size="0.12 0.03 0.005" rgba="0.1 0.3 0.9 0.8"/>'
            )
    if probe:
        pr = extension["probe"]
        out += [
            '    <body name="probe" pos="0 0 0">',
            '      <joint name="probe-x" type="slide" axis="1 0 0"/>',
            '      <joint name="probe-y" type="slide" axis="0 1 0"/>',
            '      <joint name="probe-z" type="slide" axis="0 0 1"/>',
            f'      <geom name="probe" type="sphere" size="{pr["radius"]}" density="{pr["density"]}" '
            f'friction="{_v(pr["friction"])}" priority="1" rgba="0.95 0.45 0.1 1"/>',
            "    </body>",
        ]
    out.append("  </worldbody>")
    if probe:
        pr = extension["probe"]
        out += [
            "  <actuator>",
            *(
                f'    <velocity name="drive-{a}" joint="probe-{a}" kv="{pr["kv"]}" '
                f'ctrlrange="{-pr["max_speed"]} {pr["max_speed"]}" '
                f'forcerange="{-pr["force_limit"]} {pr["force_limit"]}"/>'
                for a in "xy"
            ),
            "  </actuator>",
        ]
    out.append("</mujoco>")
    return "\n".join(out) + "\n"


def compile_model(xml: str):
    import mujoco

    return mujoco.MjModel.from_xml_string(xml)


def quantity(name, kind, canonical, measured, tolerance, unit, method):
    deviation = None if measured is None else measured - canonical
    return {
        "name": name,
        "kind": kind,
        "unit": unit,
        "canonical": canonical,
        "measured": measured,
        "deviation": deviation,
        "tolerance": tolerance,
        "pass": deviation is not None and abs(deviation) <= tolerance + 1e-12,
        "method": method,
    }


def measure_geometry(model, world: dict) -> list[dict]:
    """Read dimensions and named places back from the compiled model, not from world.json."""
    import mujoco

    tol_d, tol_a = world["tolerances"]["distance_m"], world["tolerances"]["angle_deg"]
    d = world["dimensions"]

    def geom(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise ValueError(f"compiled model has no geom {name!r}")
        return model.geom_pos[i], model.geom_size[i], model.geom_quat[i]

    def site(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError(f"compiled model has no site {name!r}")
        return model.site_pos[i], model.site_size[i], model.site_quat[i]

    floor = next(g["name"] for g in world["geometry"] if g["role"] == "floor")
    walls = [g["name"] for g in world["geometry"] if g["role"] == "wall"]
    fp, fs, _ = geom(floor)
    left = max(walls, key=lambda n: geom(n)[0][1])
    right = min(walls, key=lambda n: geom(n)[0][1])
    lp, ls, _ = geom(left)
    rp, rs, _ = geom(right)
    floor_top = fp[2] + fs[2]
    m = "compiled MjModel geom_pos/geom_size"
    out = [
        quantity(
            "clear_width",
            "dimension",
            d["clear_width"],
            float((lp[1] - ls[1]) - (rp[1] + rs[1])),
            tol_d,
            "m",
            m,
        ),
        quantity("length", "dimension", d["length"], float(2 * fs[0]), tol_d, "m", m),
        *[
            quantity(
                f"length/{n}", "dimension", d["length"], float(2 * geom(n)[1][0]), tol_d, "m", m
            )
            for n in walls
        ],
        *[
            quantity(
                f"wall_height/{n}",
                "dimension",
                d["wall_height"],
                float(geom(n)[0][2] + geom(n)[1][2] - floor_top),
                tol_d,
                "m",
                m,
            )
            for n in walls
        ],
        *[
            quantity(
                f"wall_thickness/{n}",
                "dimension",
                d["wall_thickness"],
                float(2 * geom(n)[1][1]),
                tol_d,
                "m",
                m,
            )
            for n in walls
        ],
        quantity(
            "exterior_width",
            "dimension",
            d["exterior_width"],
            float((lp[1] + ls[1]) - (rp[1] - rs[1])),
            tol_d,
            "m",
            m,
        ),
        quantity(
            "floor_thickness", "dimension", d["floor_thickness"], float(2 * fs[2]), tol_d, "m", m
        ),
        *[
            quantity(
                f"origin/{a}",
                "frame",
                0.0,
                float(v),
                tol_d,
                "m",
                "floor top-surface centre from geom_pos/geom_size",
            )
            for a, v in zip("xyz", (fp[0], fp[1], floor_top))
        ],
        quantity(
            "inner_face/+y", "dimension", d["clear_width"] / 2, float(lp[1] - ls[1]), tol_d, "m", m
        ),
        quantity(
            "inner_face/-y", "dimension", -d["clear_width"] / 2, float(rp[1] + rs[1]), tol_d, "m", m
        ),
    ]
    # Frame axes: every geom must stay axis-aligned (identity rotation) in the world frame.
    for n in [floor, *walls]:
        q = geom(n)[2]
        angle = math.degrees(2 * math.acos(min(1.0, abs(float(q[0])))))
        out.append(
            quantity(
                f"axes/{n}", "frame", 0.0, angle, tol_a, "deg", "compiled geom_quat rotation angle"
            )
        )
    sm = "compiled MjModel site_pos/site_size/site_quat"
    for p in world["places"]:
        n, t = p["name"], p["type"]
        pts = []
        if t == "segment":
            pts = [(n, p["reference"])] + [(f"{n}/{i}", pt) for i, pt in enumerate(p["points"])]
        elif t == "polyline":
            pts = [(f"{n}/{i}", pt) for i, pt in enumerate(p["points"])]
        elif t == "pose":
            pts = [(n, p["position"])]
        for sname, pt in pts:
            pos = site(sname)[0]
            out += [
                quantity(f"place/{sname}/{a}", "place", float(c), float(v), tol_d, "m", sm)
                for a, c, v in zip("xyz", pt, pos)
            ]
        if t == "region":
            pos, size, _ = site(n)
            for a, i in (("x", 0), ("y", 1)):
                lo, hi = p[a]
                out.append(
                    quantity(
                        f"place/{n}/{a}_min",
                        "place",
                        float(lo),
                        float(pos[i] - size[i]),
                        tol_d,
                        "m",
                        sm,
                    )
                )
                out.append(
                    quantity(
                        f"place/{n}/{a}_max",
                        "place",
                        float(hi),
                        float(pos[i] + size[i]),
                        tol_d,
                        "m",
                        sm,
                    )
                )
        if t == "pose":
            q = site(n)[2]
            yaw = math.degrees(2 * math.atan2(float(q[3]), float(q[0])))
            canonical = float(p["yaw_deg"])
            yaw = canonical + ((yaw - canonical + 180) % 360 - 180)
            out.append(
                quantity(
                    f"place/{n}/yaw", "place", canonical, yaw, tol_a, "deg", sm + " (yaw about +z)"
                )
            )
    return out


def _probe_ids(model):
    import mujoco

    j = {
        a: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"probe-{a}")]
        for a in "xyz"
    }
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe")
    return j, g


def _contacts(model, data, probe_geom):
    import mujoco

    names = set()
    for c in data.contact[: data.ncon]:
        if probe_geom in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == probe_geom else c.geom1
            names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
    return names


def _drive(model, data, target, axis, seconds, speed):
    """Velocity-servo drive toward a target coordinate on one axis; returns per-step samples."""
    import mujoco

    j, g = _probe_ids(model)
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive-{axis}")
    other = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive-" + ("y" if axis == "x" else "x")
    )
    samples = []
    for _ in range(int(round(seconds / model.opt.timestep))):
        pos = data.qpos[j[axis]]
        command = speed if target is None else max(-speed, min(speed, 10.0 * (target - pos)))
        data.ctrl[act] = command
        data.ctrl[other] = 0.0
        mujoco.mj_step(model, data)
        if any(w.number for w in data.warning):
            raise RuntimeError("MuJoCo raised a simulation warning")
        samples.append(
            {
                "t": float(data.time),
                "x": float(data.qpos[j["x"]]),
                "y": float(data.qpos[j["y"]]),
                "z": float(data.qpos[j["z"]]),
                "vx": float(data.qvel[j["x"]]),
                "vy": float(data.qvel[j["y"]]),
                "contacts": sorted(_contacts(model, data, g)),
            }
        )
    return samples


def _reset(model, data, settle=1.5):
    import mujoco

    j, _ = _probe_ids(model)
    mujoco.mj_resetData(model, data)
    data.qpos[j["z"]] = 0.75  # release from rest at spawn-centre (spec: centre at z = 0.75 m)
    return _drive(model, data, 0.0, "x", settle, 0.0)


def collision_checks(world: dict, extension: dict) -> tuple[list[dict], dict]:
    import mujoco

    model = compile_model(to_mjcf(world, extension, probe=True))
    data = mujoco.MjData(model)
    tol = world["tolerances"]["distance_m"]
    r = extension["probe"]["radius"]
    speed = extension["probe"]["max_speed"]
    half = world["dimensions"]["clear_width"] / 2
    out, notes = [], {}

    drop = _reset(model, data, settle=2.0)
    penetration = max(0.0, max(-(s["z"] - r) for s in drop))
    out.append(
        quantity(
            "collision/floor/penetration",
            "collision",
            0.0,
            penetration,
            tol,
            "m",
            "max over all steps of (floor top - sphere bottom), floored at 0",
        )
    )
    final = drop[-1]
    out.append(
        quantity(
            "collision/floor/settled_bottom_z",
            "collision",
            0.0,
            final["z"] - r,
            tol,
            "m",
            "sphere bottom z after 2.0 s",
        )
    )
    notes["floor_contacts"] = sorted({c for s in drop for c in s["contacts"]})

    for sign, label in ((1, "+y"), (-1, "-y")):
        _reset(model, data)
        push = _drive(model, data, None, "y", 5.0, sign * speed)
        overshoot = max(sign * s["y"] + r - half for s in push)
        touched = sorted({c for s in push for c in s["contacts"]} - {"floor"})
        max_speed = max(abs(s["vy"]) for s in push)
        out.append(
            quantity(
                f"collision/wall{label}/surface_past_inner_face",
                "collision",
                0.0,
                max(0.0, overshoot),
                tol,
                "m",
                "max over steps of (sphere surface - inner face), floored at 0",
            )
        )
        out.append(
            {
                **quantity(
                    f"collision/wall{label}/max_speed",
                    "drive",
                    speed,
                    max_speed,
                    0.0,
                    "m/s",
                    "max |vy| over steps; must not exceed the drive limit",
                ),
                "deviation": max_speed - speed,
                "pass": max_speed <= speed + 1e-9,
            }
        )
        out.append(
            {
                **quantity(
                    f"collision/wall{label}/reached_wall",
                    "collision",
                    1.0,
                    1.0 if touched else 0.0,
                    0.0,
                    "bool",
                    "probe contacted a wall geom",
                ),
                "contacts": touched,
            }
        )

    for target in (2.8, -2.8):
        _reset(model, data)
        run = _drive(model, data, target, "x", 16.0, speed)
        touched = sorted({c for s in run for c in s["contacts"]})
        others = [c for c in touched if c != "floor"]
        out.append(
            quantity(
                f"collision/open_end/{target:+.2f}/final_x",
                "collision",
                target,
                run[-1]["x"],
                tol,
                "m",
                "probe centre x after drive along passage-centreline",
            )
        )
        out.append(
            {
                **quantity(
                    f"collision/open_end/{target:+.2f}/non_floor_contacts",
                    "collision",
                    0.0,
                    float(len(others)),
                    0.0,
                    "count",
                    "distinct non-floor geoms contacted",
                ),
                "contacts": touched,
            }
        )
        out.append(
            quantity(
                f"collision/open_end/{target:+.2f}/max_abs_y",
                "collision",
                0.0,
                max(abs(s["y"]) for s in run),
                tol,
                "m",
                "max |y| during the drive",
            )
        )
    notes.update(
        gravity=[float(v) for v in model.opt.gravity],
        contacts_enabled=not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT),
        timestep=float(model.opt.timestep),
        sampling_hz=1.0 / float(model.opt.timestep),
        drive_method=f"{extension['probe']['drive']}, |command| <= {speed} m/s, kv {extension['probe']['kv']}, "
        f"force limit {extension['probe']['force_limit']} N; no teleporting",
        measurement_method="joint positions and contact list sampled every physics step",
        probe={k: extension["probe"][k] for k in ("radius", "density", "friction")},
    )
    return out, notes


def adapter_identity() -> dict:
    from livingproof_sim_mujoco.cli import identity

    ident = identity()
    return {"name": "livingproof-sim-mujoco", **ident, "mujoco": version("mujoco")}


def run_checks(fixture: str, content_version: str, root: Path = WORLDS) -> dict:
    """Return one engine-neutral result record (see docs/hallway-result.schema.json)."""
    loaded = load_fixture(fixture, content_version, root)
    world, extension, pin = loaded["world"], loaded["extension"], loaded["pin"]
    spec = world["specification"]
    base = {
        "format": RESULT_FORMAT,
        "fixture": {"id": fixture, "content_version": content_version},
        "contract": {
            "world_contract_revision": spec["world_contract_revision"],
            "document": spec["path"],
            "document_revision": spec["document_revision"],
        },
        "catalog": {"repository": spec["repository"], "tag": spec["tag"], "commit": spec["commit"]},
        "worlds": {k: pin[k] for k in ("repository", "tag", "commit", "path", "release_sha256")},
        "engine": {"name": "mujoco", "version": version("mujoco")},
        "adapter": adapter_identity(),
    }
    try:
        model = compile_model(to_mjcf(world, extension))
    except Unsupported as exc:
        return {**base, **exc.record(), "quantities": []}
    quantities = measure_geometry(model, world)
    collisions, notes = collision_checks(world, extension)
    quantities += collisions
    return {
        **base,
        "status": "PASS" if all(q["pass"] for q in quantities) else "FAIL",
        "compile": {
            "status": "PASS",
            "method": "mujoco.MjModel.from_xml_string",
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
        },
        "settings": {"extension": pins()[f"{fixture}/{content_version}"]["extension"], **notes},
        "quantities": quantities,
    }
