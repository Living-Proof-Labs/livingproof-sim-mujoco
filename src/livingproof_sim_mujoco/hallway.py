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

from livingproof_sim_mujoco import contract

WORLDS = Path(__file__).parent / "worlds"
RESULT_FORMAT = contract.RESULT_FORMAT
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


# Only these released files are vendored and checksum-verified. The generated docs
# listed in release.json are skipped on purpose: they are renders of world.json.
VENDORED = {"world.json", "notes.md", "release.json"}
GENERATED_DOCS = {"README.md", "world.svg", "world-light.svg", "world-dark.svg"}


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
            "fixture": {"id": self.fixture, "content_version": self.version},
            "content_version": self.version,
            "feature": self.feature,
        }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins() -> dict:
    return json.loads((WORLDS / "pins.json").read_text())


def same_byte_releases(fixture: str, version: str) -> tuple[tuple[str, str], ...]:
    """``(tag, commit)`` pairs whose world.json matches the vendored file.

    ``worlds/same-bytes.json`` lists them. The pin in ``worlds/pins.json`` is
    accepted as well.
    """
    document = json.loads((WORLDS / "same-bytes.json").read_text())
    found = document.get(f"{fixture}/{version}", [])
    return tuple((tag, commit) for tag, commit in found)


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
        if name in GENERATED_DOCS:
            # Generated docs (README, SVG renders) are not vendored; skipped on purpose.
            continue
        if not path.is_file():
            raise FixtureError(f"{key}: missing released file {name}")
        got = sha256(path)
        if got != expected:
            raise FixtureError(
                f"{key}: checksum mismatch for {name}: expected {expected}, got {got}"
            )
    unknown = sorted(n for n in manifest["files"] if n not in VENDORED and n not in GENERATED_DOCS)
    if unknown:
        raise FixtureError(
            f"{key}: release.json lists files this adapter does not know: {', '.join(unknown)}"
        )
    extras = sorted(p.name for p in directory.iterdir() if p.name not in VENDORED)
    if extras:
        raise FixtureError(f"{key}: files not vendored from release.json: {', '.join(extras)}")
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


# Only these keys may appear in a per-world MuJoCo extension; anything else is UNSUPPORTED.
EXTENSION_KEYS = {
    "": {"fixture", "content_version", "note", "option", "contact", "probe"},
    "option": {"timestep", "integrator", "gravity", "cone"},
    "contact": {"solref", "solimp", "friction", "condim", "margin"},
    "probe": {"radius", "density", "friction", "drive", "kv", "force_limit", "max_speed"},
}


def check_extension(extension: dict, fixture: str, version_: str) -> None:
    """Reject unknown extension keys and any setting that would move a collision boundary."""
    for section, allowed in EXTENSION_KEYS.items():
        block = extension if section == "" else extension.get(section)
        if not isinstance(block, dict):
            raise Unsupported(fixture, version_, f"extension section '{section}' missing")
        for key in sorted(set(block) - allowed):
            where = f"{section}.{key}" if section else key
            raise Unsupported(fixture, version_, f"extension key '{where}'")
    shared = contract.probe()["probe"]
    pr, opt = extension["probe"], extension["option"]
    try:
        contract.sampling(opt["timestep"])  # the step must give the shared sample rate
    except contract.ContractError as exc:
        raise Unsupported(fixture, version_, f"extension option.timestep: {exc}") from None
    for key, have, want in (
        ("probe.radius", pr["radius"], shared["radius_m"]),
        ("probe.density", pr["density"], shared["density_kg_per_m3"]),
        ("probe.friction[0]", pr["friction"][0], shared["friction"]),
        ("probe.max_speed", pr["max_speed"], shared["max_speed_m_per_s"]),
        ("option.gravity", list(opt["gravity"]), contract.probe()["gravity_m_per_s2"]),
    ):
        if have != want:
            raise Unsupported(
                fixture,
                version_,
                f"extension {key} = {have!r} contradicts hallway-probe.json ({want!r})",
            )
    if extension["contact"]["margin"] != 0:
        raise Unsupported(
            fixture,
            version_,
            f"extension contact.margin {extension['contact']['margin']} (must be 0; "
            "a margin moves the collision boundary)",
        )


def to_mjcf(world: dict, extension: dict, probe: bool = False) -> str:
    """Translate world.json to MJCF. Geometry is copied verbatim; places become sites."""
    supported(world)
    check_extension(extension, world["id"], world["content_version"])
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
            f'    <body name="probe" pos="0 0 {start_z()!r}">',
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


def start_z() -> float:
    return float(contract.probe()["checks"]["floor"]["start_z_m"])


def compile_model(xml: str):
    import mujoco

    return mujoco.MjModel.from_xml_string(xml)


def quantity(name, kind, canonical, measured, tolerance, unit, method, **extra):
    """A 0.2 quantity; kind/unit/comparison must agree with hallway-quantities.json."""
    q = contract.quantity(name, canonical, measured, tolerance, method, **extra)
    if (q["kind"], q["unit"]) != (kind, unit):
        raise contract.ContractError(
            f"{name}: adapter says {kind}/{unit}, contract {q['kind']}/{q['unit']}"
        )
    return q


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
    stride = contract.sampling(float(model.opt.timestep))["sample_stride"]
    for step in range(int(round(seconds / model.opt.timestep))):
        pos = data.qpos[j[axis]]
        command = speed if target is None else max(-speed, min(speed, 10.0 * (target - pos)))
        data.ctrl[act] = command
        data.ctrl[other] = 0.0
        mujoco.mj_step(model, data)
        if any(w.number for w in data.warning):
            raise RuntimeError("MuJoCo raised a simulation warning")
        if (step + 1) % stride:
            continue  # sample at the shared rate: every stride-th physics step
        samples.append(
            {
                "t": float(data.time),
                # world = joint position + body offset (0, 0, start_z); qpos is post-step
                "x": float(data.qpos[j["x"]]),
                "y": float(data.qpos[j["y"]]),
                "z": float(data.qpos[j["z"]]) + start_z(),
                "vx": float(data.qvel[j["x"]]),
                "vy": float(data.qvel[j["y"]]),
                "contacts": sorted(_contacts(model, data, g)),
            }
        )
    return samples


def _reset(model, data, settle=1.5):
    """mj_resetData restores the initial state: probe at rest at spawn-centre, z = start_z."""
    import mujoco

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return _drive(model, data, 0.0, "x", settle, 0.0)


def _snapshot(model, data) -> dict:
    """World positions of every static geom, every site and the probe, read from MjData."""
    import mujoco

    def name(kind, i):
        return mujoco.mj_id2name(model, kind, i)

    return {
        "geoms": {
            name(mujoco.mjtObj.mjOBJ_GEOM, i): [float(v) for v in data.geom_xpos[i]]
            for i in range(model.ngeom)
            if name(mujoco.mjtObj.mjOBJ_GEOM, i) != "probe"
        },
        "sizes": {
            name(mujoco.mjtObj.mjOBJ_GEOM, i): [float(v) for v in model.geom_size[i]]
            for i in range(model.ngeom)
        },
        "places": {
            name(mujoco.mjtObj.mjOBJ_SITE, i): [float(v) for v in data.site_xpos[i]]
            for i in range(model.nsite)
        },
        "probe": [float(v) for v in data.geom_xpos[_probe_ids(model)[1]]],
    }


def collision_checks(world: dict, extension: dict) -> tuple[list[dict], dict]:
    import mujoco

    model = compile_model(to_mjcf(world, extension, probe=True))
    data = mujoco.MjData(model)
    tol = world["tolerances"]["distance_m"]
    shared = contract.probe()
    r = shared["probe"]["radius_m"]
    speed = shared["probe"]["max_speed_m_per_s"]
    checks = shared["checks"]
    half = world["dimensions"]["clear_width"] / 2
    out, notes = [], {}

    mujoco.mj_forward(model, data)
    before = _snapshot(model, data)
    drop = _reset(model, data, settle=checks["floor"]["settle_s"])
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
        _reset(model, data, settle=checks["wall"]["settle_s"])
        push = _drive(model, data, None, "y", checks["wall"]["drive_s"], sign * speed)
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
            quantity(
                f"collision/wall{label}/max_speed",
                "drive",
                speed,
                max_speed,
                0.0,
                "m/s",
                "max |vy| over steps; at_most the 0.25 m/s cap",
            )
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

    for target in checks["open_end"]["targets_x_m"]:
        end = "+x" if target > 0 else "-x"
        _reset(model, data, settle=checks["open_end"]["settle_s"])
        run = _drive(model, data, target, "x", checks["open_end"]["drive_s"], speed)
        touched = sorted({c for s in run for c in s["contacts"]})
        others = [c for c in touched if c != "floor"]
        out.append(
            quantity(
                f"collision/open_end/{end}/final_x",
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
                    f"collision/open_end/{end}/non_floor_contacts",
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
                f"collision/open_end/{end}/max_abs_y",
                "collision",
                0.0,
                max(abs(s["y"]) for s in run),
                tol,
                "m",
                "max |y| during the drive",
            )
        )
    # hallways.md: reset restores the geometry, frame and initial world state.
    _reset(model, data, settle=0.0)
    after = _snapshot(model, data)
    out += reset_checks(before, after, tol)
    notes.update(
        gravity=[float(v) for v in model.opt.gravity],
        contacts_enabled=not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT),
        timestep=float(model.opt.timestep),
        **contract.sampling(float(model.opt.timestep)),
        sampling_hz=contract.sampling(float(model.opt.timestep))["sample_rate"],
        drive_method=f"{extension['probe']['drive']}, |command| <= {speed} m/s, kv {extension['probe']['kv']}, "
        f"force limit {extension['probe']['force_limit']} N; no teleporting",
        measurement_method="joint positions and contact list sampled at sample_rate (every sample_stride-th physics step)",
        probe={
            **shared["probe"],
            "density": extension["probe"]["density"],
            "source": "hallway-probe.json",
        },
        reset_method="mj_resetData + mj_forward after all runs, then readback from MjData",
    )
    return out, notes


def reset_checks(before: dict, after: dict, tol: float) -> list[dict]:
    """reset/* quantities: geometry and places unchanged, probe back at its start."""

    def shift(a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    out = [
        quantity(
            f"reset/geometry/{n}/max_shift",
            "reset",
            0.0,
            max(
                shift(before["geoms"][n], after["geoms"][n]),
                shift(before["sizes"][n], after["sizes"][n]),
            ),
            tol,
            "m",
            "max |change| of world centre and half-size, before the runs vs after reset",
        )
        for n in sorted(before["geoms"])
    ]
    out.append(
        quantity(
            "reset/places/max_shift",
            "reset",
            0.0,
            max(shift(before["places"][n], after["places"][n]) for n in before["places"]),
            tol,
            "m",
            "max |change| of any place marker, before the runs vs after reset",
        )
    )
    start = [0.0, 0.0, start_z()]
    out += [
        quantity(
            f"reset/probe/{a}",
            "reset",
            c,
            v,
            tol,
            "m",
            "probe centre after reset; start is spawn-centre at start_z_m",
        )
        for a, c, v in zip("xyz", start, after["probe"])
    ]
    return out


def adapter_identity() -> dict:
    from livingproof_sim_mujoco.cli import identity

    ident = identity()
    return contract.adapter(
        "livingproof-sim-mujoco",
        Path(__file__).resolve().parent,
        ident["version"],
        ident["provenance"],
        package_version=ident["package_version"],
        mujoco=version("mujoco"),
    )


def run_checks(fixture: str, content_version: str, root: Path = WORLDS) -> dict:
    """Return one engine-neutral result record (livingproof-world-result/0.2, contract/hallway-result.schema.json)."""
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


# Per-trial walking observations live in observations.py. They are re-exported lazily
# so `hallway.observe` keeps working without importing mujoco at module import.
_OBSERVATIONS = {"observe", "WallDistance", "FORMAT", "DISTMAX", "COLUMNS"}


def __getattr__(name):
    if name in _OBSERVATIONS:
        from livingproof_sim_mujoco import observations

        return getattr(observations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
