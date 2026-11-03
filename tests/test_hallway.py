# SPDX-License-Identifier: Apache-2.0
"""Pinned hallway-open fixture: checksums, MJCF translation, UNSUPPORTED and hallways.md checks."""

import copy
import json
import shutil

import pytest

from livingproof_sim_mujoco import contract, hallway

FIXTURE, VERSION = "hallway-open", "0.1.0"


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
    with pytest.raises(
        hallway.FixtureError, match="files not vendored from release.json: extra.txt"
    ):
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
    contract.validate(record)


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
    for end in ("+x", "-x"):
        assert q[f"collision/open_end/{end}/non_floor_contacts"]["contacts"] == ["floor"]
    s = result["settings"]
    assert s["gravity"] == [0.0, 0.0, -9.81] and s["contacts_enabled"] and s["sampling_hz"] == 500.0


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
    assert record["status"] == "UNSUPPORTED"
    contract.validate(record)


@pytest.mark.parametrize(
    "edit, feature",
    [
        (
            lambda e: e["probe"].update(radius=0.12),
            "extension probe.radius = 0.12 contradicts hallway-probe.json (0.1)",
        ),
        (
            lambda e: e["probe"].update(friction=[0.3, 0.005, 0.0001]),
            "extension probe.friction[0] = 0.3 contradicts hallway-probe.json (0.1)",
        ),
        (
            lambda e: e["probe"].update(max_speed=0.5),
            "extension probe.max_speed = 0.5 contradicts hallway-probe.json (0.25)",
        ),
        (
            lambda e: e["option"].update(timestep=0.003),
            "extension option.timestep: physics step 0.003 s does not divide the 500 Hz sample period",
        ),
        (
            lambda e: e["probe"].update(density=500.0),
            "extension probe.density = 500.0 contradicts hallway-probe.json (1000.0)",
        ),
        (
            lambda e: e["option"].update(gravity=[0.0, 0.0, -9.8]),
            "extension option.gravity = [0.0, 0.0, -9.8] contradicts hallway-probe.json ([0.0, 0.0, -9.81])",
        ),
    ],
)
def test_extension_contradicting_shared_probe_is_unsupported(loaded, edit, feature):
    extension = copy.deepcopy(loaded["extension"])
    edit(extension)
    with pytest.raises(hallway.Unsupported) as info:
        hallway.to_mjcf(loaded["world"], extension)
    assert info.value.feature == feature


def test_only_world_notes_release_are_vendored_and_generated_docs_skipped_on_purpose(copy_root):
    directory = copy_root / FIXTURE / VERSION
    assert sorted(p.name for p in directory.iterdir()) == ["notes.md", "release.json", "world.json"]
    manifest = json.loads((directory / "release.json").read_text())
    skipped = set(manifest["files"]) - hallway.VENDORED
    # release.json lists the generated README and SVG renders; they are deliberately not vendored or checked.
    assert (
        skipped
        == hallway.GENERATED_DOCS
        == {"README.md", "world.svg", "world-light.svg", "world-dark.svg"}
    )
    hallway.load_fixture(FIXTURE, VERSION, copy_root)
    (directory / "world.svg").write_text("<svg/>")  # a generated doc reappearing is still rejected
    with pytest.raises(
        hallway.FixtureError, match="files not vendored from release.json: world.svg"
    ):
        hallway.load_fixture(FIXTURE, VERSION, copy_root)


def test_sampling_records_physics_step_and_rate_every_step_at_0_002(result):
    s = result["settings"]
    assert (s["physics_step"], s["sample_rate"], s["sample_stride"]) == (0.002, 500, 1)
    assert s["timestep"] == s["physics_step"]


def test_finer_physics_step_samples_every_second_step(loaded):
    import mujoco

    extension = copy.deepcopy(loaded["extension"])
    extension["option"]["timestep"] = 0.001
    hallway.check_extension(extension, FIXTURE, VERSION)  # 0.001 s is allowed: stride 2
    model = mujoco.MjModel.from_xml_string(hallway.to_mjcf(loaded["world"], extension, probe=True))
    data = mujoco.MjData(model)
    samples = hallway._drive(model, data, None, "x", 0.1, 0.0)
    assert len(samples) == 50  # 100 physics steps, 500 Hz
    assert samples[1]["t"] - samples[0]["t"] == pytest.approx(0.002)


def test_observations_are_reexported_from_hallway():
    from livingproof_sim_mujoco import observations

    assert hallway.observe is observations.observe
    assert hallway.FORMAT == "livingproof-hallway-observation/2.0.0"
