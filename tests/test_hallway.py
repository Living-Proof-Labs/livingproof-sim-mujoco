# SPDX-License-Identifier: Apache-2.0
"""Pinned hallway-open fixture: checksums, MJCF translation, UNSUPPORTED and hallways.md checks."""

import copy
import json
from pathlib import Path
import shutil

import pytest

from livingproof_sim_mujoco import hallway

FIXTURE, VERSION = "hallway-open", "0.1.0"
SCHEMA = json.loads(Path("docs/hallway-result.schema.json").read_text())


@pytest.fixture(scope="module")
def loaded():
    return hallway.load_fixture(FIXTURE, VERSION)


@pytest.fixture(scope="module")
def result():
    return hallway.run_checks(FIXTURE, VERSION)


@pytest.fixture
def copy_root(tmp_path):
    root = tmp_path / "worlds"
    shutil.copytree(hallway.WORLDS, root)
    return root


def by_name(result):
    return {q["name"]: q for q in result["quantities"]}


def test_pin_names_tag_and_full_commit(loaded):
    pin = loaded["pin"]
    assert pin["repository"] == "Living-Proof-Labs/livingproof-worlds"
    assert pin["tag"] == "v1.1.0"
    assert pin["commit"] == "9b3e7119aa93a3370383227fb30bd39a9dd72b4b"
    assert loaded["world"]["specification"]["tag"] == "v1.3.0"


def test_tampered_world_file_fails_with_named_mismatch(copy_root):
    path = copy_root / FIXTURE / VERSION / "world.json"
    path.write_text(path.read_text().replace('"clear_width": 1.5', '"clear_width": 1.6'))
    with pytest.raises(
        hallway.FixtureError, match=r"checksum mismatch for world.json: expected d60bb8"
    ):
        hallway.load_fixture(FIXTURE, VERSION, copy_root)


def test_tampered_release_manifest_fails(copy_root):
    path = copy_root / FIXTURE / VERSION / "release.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(
        hallway.FixtureError, match=r"release.json checksum mismatch.*v1.1.0 9b3e7119"
    ):
        hallway.load_fixture(FIXTURE, VERSION, copy_root)


def test_missing_and_unlisted_files_fail(copy_root):
    directory = copy_root / FIXTURE / VERSION
    (directory / "extra.txt").write_text("x")
    with pytest.raises(hallway.FixtureError, match="files not in release.json: extra.txt"):
        hallway.load_fixture(FIXTURE, VERSION, copy_root)
    (directory / "extra.txt").unlink()
    (directory / "notes.md").unlink()
    with pytest.raises(hallway.FixtureError, match="missing released file notes.md"):
        hallway.load_fixture(FIXTURE, VERSION, copy_root)


@pytest.mark.parametrize(
    "fixture,version_", [("hallway-closed", "0.1.0"), ("hallway-open", "0.2.0")]
)
def test_unknown_fixture_or_version_is_reported_not_substituted(fixture, version_):
    with pytest.raises(hallway.FixtureError, match=f"unknown fixture {fixture}/{version_}"):
        hallway.load_fixture(fixture, version_)


def test_extension_holds_no_geometry_or_places(loaded):
    assert not {"geometry", "places", "dimensions"} & set(loaded["extension"])


def test_mjcf_compiles_with_geometry_verbatim_and_places_as_sites(loaded):
    import mujoco

    model = hallway.compile_model(hallway.to_mjcf(loaded["world"], loaded["extension"]))
    for g in loaded["world"]["geometry"]:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g["name"])
        assert list(model.geom_pos[i]) == pytest.approx(g["centre"])
        assert list(2 * model.geom_size[i]) == pytest.approx(g["size"])
    sites = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)}
    assert {p["name"] for p in loaded["world"]["places"] if p["type"] != "polyline"} <= sites
    assert {"passage-centreline/0", "passage-centreline/1", "entrance/0", "exit/1"} <= sites
    assert model.ngeom == 3  # no robot, no probe in the world model


@pytest.mark.parametrize(
    "edit,feature",
    [
        (
            lambda w: w.update(format="livingproof-world/0.2"),
            "world format 'livingproof-world/0.2'",
        ),
        (lambda w: w.update(lights=[]), "top-level field 'lights'"),
        (lambda w: w["geometry"][1].update(type="mesh"), "geometry 'wall-left' type 'mesh'"),
        (
            lambda w: w["geometry"][2].update(rotation=[0, 0, 5]),
            "geometry 'wall-right' field 'rotation'",
        ),
        (lambda w: w["geometry"][0].update(role="ceiling"), "geometry 'floor' role 'ceiling'"),
        (
            lambda w: w["places"].append({"name": "door", "type": "door"}),
            "place 'door' type 'door'",
        ),
        (lambda w: w["places"][4].update(pitch_deg=5), "place 'spawn-entrance' field 'pitch_deg'"),
        (lambda w: w.update(units="ft"), "units 'ft'"),
    ],
)
def test_unmapped_feature_is_unsupported_naming_fixture_version_feature(loaded, edit, feature):
    world = copy.deepcopy(loaded["world"])
    edit(world)
    with pytest.raises(hallway.Unsupported) as info:
        hallway.to_mjcf(world, loaded["extension"])
    assert str(info.value) == f"UNSUPPORTED: hallway-open 0.1.0: {feature}"
    assert info.value.record() == {
        "status": "UNSUPPORTED",
        "fixture": {"id": "hallway-open", "content_version": "0.1.0"},
        "content_version": "0.1.0",
        "feature": feature,
    }


def test_run_checks_reports_unsupported_and_never_passes(monkeypatch, loaded):
    world = copy.deepcopy(loaded["world"])
    world["geometry"][1]["type"] = "cylinder"
    monkeypatch.setattr(hallway, "load_fixture", lambda *a: {**loaded, "world": world})
    record = hallway.run_checks(FIXTURE, VERSION)
    assert record["status"] == "UNSUPPORTED"
    assert record["feature"] == "geometry 'wall-left' type 'cylinder'"
    assert record["quantities"] == []
    assert record["fixture"] == {"id": FIXTURE, "content_version": VERSION}
    assert schema_errors(record, SCHEMA) == []


def test_geometry_readback_detects_a_wrong_model(loaded):
    xml = hallway.to_mjcf(loaded["world"], loaded["extension"]).replace(
        'pos="0.0 0.825 1.25"', 'pos="0.0 0.845 1.25"'
    )
    q = {
        x["name"]: x for x in hallway.measure_geometry(hallway.compile_model(xml), loaded["world"])
    }
    assert not q["clear_width"]["pass"] and q["clear_width"]["deviation"] == pytest.approx(0.02)


def test_every_quantity_passes(result):
    failed = [q["name"] for q in result["quantities"] if not q["pass"]]
    assert failed == [] and result["status"] == "PASS"


def test_dimensions_places_and_yaws_are_covered(result):
    q = by_name(result)
    for name in [
        "clear_width",
        "length",
        "exterior_width",
        "floor_thickness",
        "origin/z",
        "wall_height/wall-left",
        "wall_thickness/wall-right",
        "axes/floor",
        "place/entrance/x",
        "place/exit/1/y",
        "place/passage/y_min",
        "place/passage-centreline/1/x",
        "place/spawn-exit/yaw",
    ]:
        assert name in q, name
    assert q["place/spawn-exit/yaw"]["canonical"] == 180.0
    assert q["place/spawn-exit/yaw"]["tolerance"] == 0.5
    assert q["clear_width"]["tolerance"] == 0.01


def test_collision_checks_are_real_and_bounded(result):
    q = by_name(result)
    assert q["collision/floor/penetration"]["measured"] <= 0.01
    for side, wall in (("+y", "wall-left"), ("-y", "wall-right")):
        assert q[f"collision/wall{side}/reached_wall"]["contacts"] == [wall]
        assert q[f"collision/wall{side}/max_speed"]["measured"] <= 0.25
        assert q[f"collision/wall{side}/surface_past_inner_face"]["measured"] <= 0.01
    for end in ("+2.80", "-2.80"):
        assert q[f"collision/open_end/{end}/non_floor_contacts"]["contacts"] == ["floor"]
    s = result["settings"]
    assert s["gravity"] == [0.0, 0.0, -9.81] and s["contacts_enabled"] and s["sampling_hz"] == 500.0


def test_result_matches_documented_schema(result):
    for key in SCHEMA["required"]:
        assert key in result, key
    assert result["format"] == SCHEMA["properties"]["format"]["const"]
    item = SCHEMA["properties"]["quantities"]["items"]
    for q in result["quantities"]:
        assert set(item["required"]) <= set(q), q["name"]
        assert q["kind"] in item["properties"]["kind"]["enum"]
        assert q["unit"] in item["properties"]["unit"]["enum"]
    assert len(result["catalog"]["commit"]) == 40 and len(result["worlds"]["commit"]) == 40
    assert result["contract"]["world_contract_revision"] == "0.2.0"
    assert (
        result["adapter"]["name"] == "livingproof-sim-mujoco"
        and result["engine"]["name"] == "mujoco"
    )


def test_adapter_version_is_full_commit():
    import re as _re

    ident = hallway.adapter_identity()
    assert _re.fullmatch(r"[0-9a-f]{40}", ident["version"]), ident["version"]
    pattern = SCHEMA["properties"]["adapter"]["properties"]["version"]["pattern"]
    assert pattern == "^[0-9a-f]{40}$"
    assert ident["describe"] != ident["version"]


def schema_errors(value, schema, path="$"):
    """Minimal JSON Schema subset (type, const, enum, pattern, required, properties, items)."""
    import re as _re

    kinds = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "null": type(None),
        "boolean": bool,
        "integer": int,
    }
    errs = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(value, kinds[t]) for t in types):
            return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: != const")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: not in enum")
    if "pattern" in schema and isinstance(value, str) and not _re.search(schema["pattern"], value):
        errs.append(f"{path}: pattern")
    if isinstance(value, dict):
        errs += [f"{path}.{k}: required" for k in schema.get("required", []) if k not in value]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += schema_errors(value[k], sub, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errs += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errs


def test_full_result_matches_schema(result):
    assert schema_errors(result, SCHEMA) == []


def test_schema_helper_rejects_string_fixture():
    assert schema_errors({"fixture": "hallway-open"}, SCHEMA["properties"]["fixture"]) != []


@pytest.mark.parametrize(
    "edit, feature",
    [
        (lambda e: e.__setitem__("ccd", {}), "extension key 'ccd'"),
        (lambda e: e["contact"].__setitem__("gap", 0.0), "extension key 'contact.gap'"),
        (lambda e: e["probe"].__setitem__("roll", True), "extension key 'probe.roll'"),
    ],
)
def test_unknown_extension_key_is_unsupported(loaded, edit, feature):
    extension = copy.deepcopy(loaded["extension"])
    edit(extension)
    with pytest.raises(hallway.Unsupported) as info:
        hallway.to_mjcf(loaded["world"], extension)
    assert info.value.feature == feature


def test_nonzero_contact_margin_is_unsupported(loaded, monkeypatch):
    extension = copy.deepcopy(loaded["extension"])
    extension["contact"]["margin"] = 0.005
    with pytest.raises(hallway.Unsupported) as info:
        hallway.to_mjcf(loaded["world"], extension)
    assert info.value.feature.startswith("extension contact.margin 0.005")
    monkeypatch.setattr(hallway, "load_fixture", lambda *a: {**loaded, "extension": extension})
    record = hallway.run_checks(FIXTURE, VERSION)
    assert record["status"] == "UNSUPPORTED" and schema_errors(record, SCHEMA) == []


def test_adapter_commit_in_checkout():
    import re as _re

    ident = hallway.adapter_identity()
    pattern = SCHEMA["properties"]["adapter"]["properties"]["commit"]["pattern"]
    assert pattern == SCHEMA["properties"]["catalog"]["properties"]["commit"]["pattern"]
    if ident["commit"] is not None:
        assert _re.fullmatch(r"[0-9a-f]{40}", ident["commit"])
        assert ident["version"] == ident["commit"]


def test_adapter_commit_is_null_without_checkout(tmp_path, monkeypatch):
    assert hallway.checkout_commit(tmp_path) is None
    monkeypatch.setattr(hallway, "checkout_commit", lambda here=None: None)
    ident = hallway.adapter_identity()
    assert ident["commit"] is None
    assert schema_errors(ident, SCHEMA["properties"]["adapter"]) == []
