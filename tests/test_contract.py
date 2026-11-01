# SPDX-License-Identifier: Apache-2.0
"""Shared livingproof-world-result/0.2 contract: vendored files, names, passes, adapter, reset.

Identical in livingproof-sim-mujoco and livingproof-sim-gazebo apart from the package name.
"""

import json
from pathlib import Path
import re
import shutil

import pytest

from livingproof_sim_mujoco import contract, hallway

FIXTURE, VERSION = "hallway-open", "0.1.0"
SPEC_COMMIT = json.loads((contract.CONTRACT / "pins.json").read_text())["commit"]


@pytest.fixture(scope="module")
def result():
    return hallway.run_checks(FIXTURE, VERSION)


@pytest.fixture
def fresh_contract(monkeypatch, tmp_path):
    root = tmp_path / "contract"
    shutil.copytree(contract.CONTRACT, root)
    monkeypatch.setattr(contract, "CONTRACT", root)
    contract.load.cache_clear()
    contract.pin.cache_clear()
    yield root
    contract.load.cache_clear()
    contract.pin.cache_clear()


def test_contract_files_are_pinned_to_the_specs_commit():
    pin = contract.pin()
    assert pin["repository"] == "Living-Proof-Labs/robot-capability-specs"
    assert pin["commit"] == SPEC_COMMIT and re.fullmatch(r"[0-9a-f]{40}", SPEC_COMMIT)
    assert re.fullmatch(r"v\d+\.\d+\.\d+", pin["tag"])  # pinned to a released tag
    assert sorted(pin["files"]) == [
        "hallway-probe.json",
        "hallway-quantities.json",
        "hallway-result.schema.json",
    ]
    for name in pin["files"]:
        contract.load(name)  # checksum verified


def test_tampered_contract_file_is_rejected(fresh_contract):
    path = fresh_contract / "hallway-probe.json"
    path.write_text(path.read_text().replace('"radius_m": 0.1', '"radius_m": 0.2', 1))
    with pytest.raises(
        contract.ContractError, match=f"checksum mismatch for hallway-probe.json.*{SPEC_COMMIT}"
    ):
        contract.probe()


def test_local_schema_copy_is_gone():
    assert not Path("docs/hallway-result.schema.json").exists()


def test_result_validates_against_the_shared_schema_and_names(result):
    contract.validate(result)
    assert result["format"] == "livingproof-world-result/0.2"
    assert result["status"] == "PASS"
    names = [q["name"] for q in result["quantities"]]
    allowed = set(contract.names())
    assert contract.required_names() <= set(names)
    assert all(n in allowed or n.startswith("reset/") for n in names)


def test_open_end_names_carry_the_target_in_canonical(result):
    q = {x["name"]: x for x in result["quantities"]}
    assert q["collision/open_end/+x/final_x"]["canonical"] == 2.8
    assert q["collision/open_end/-x/final_x"]["canonical"] == -2.8
    assert not [n for n in q if "2.80" in n]


def test_kind_unit_comparison_and_pass_follow_the_contract(result):
    spec = contract.names()
    for q in result["quantities"]:
        want = spec.get(q["name"], contract.RESET)
        assert (q["kind"], q["unit"], q["comparison"]) == (
            want["kind"],
            want["unit"],
            want["comparison"],
        ), q["name"]
        assert q["pass"] == contract.passes(
            q["comparison"], q["canonical"], q["measured"], q["tolerance"]
        )
    at_most = {q["name"] for q in result["quantities"] if q["comparison"] == "at_most"}
    assert at_most == {
        "collision/floor/penetration",
        "collision/wall+y/surface_past_inner_face",
        "collision/wall-y/surface_past_inner_face",
        "collision/wall+y/max_speed",
        "collision/wall-y/max_speed",
    }


def test_at_most_and_within_rules():
    assert contract.passes("at_most", 0.25, 0.25, 0.0)
    assert not contract.passes("at_most", 0.25, 0.2500001, 0.0)
    assert contract.passes("at_most", 0.0, -5.0, 0.01)
    assert not contract.passes("within", 0.0, -5.0, 0.01)
    assert not contract.passes("within", 0.0, None, 0.01)


def test_unknown_quantity_name_is_refused():
    with pytest.raises(contract.ContractError, match="not in hallway-quantities.json"):
        contract.quantity("collision/open_end/+2.80/final_x", 2.8, 2.8, 0.01, "x")


def test_reset_quantities_confirm_geometry_places_and_probe_start(result):
    reset = {q["name"]: q for q in result["quantities"] if q["name"].startswith("reset/")}
    geometry = {n for n in reset if n.startswith("reset/geometry/")}
    assert geometry == {
        f"reset/geometry/{g}/max_shift" for g in ("floor", "wall-left", "wall-right")
    }
    assert {"reset/places/max_shift", "reset/probe/x", "reset/probe/y", "reset/probe/z"} <= set(
        reset
    )
    assert reset["reset/probe/z"]["canonical"] == contract.probe()["checks"]["floor"]["start_z_m"]
    assert all(q["pass"] and q["kind"] == "reset" for q in reset.values())


def test_probe_settings_come_from_the_shared_probe_file(result):
    shared = contract.probe()
    probe = result["settings"]["probe"]
    for key, value in shared["probe"].items():
        assert probe[key] == value, key
    assert result["settings"]["sampling_hz"] == shared["sampling_hz"]
    assert result["settings"]["sample_rate"] == shared["sampling_hz"]


@pytest.mark.parametrize("step,stride", [(0.002, 1), (0.001, 2), (0.0005, 4)])
def test_sampling_stride_hits_the_shared_rate(step, stride):
    s = contract.sampling(step)
    assert s == {"physics_step": step, "sample_rate": 500, "sample_stride": stride}


@pytest.mark.parametrize("step", [0.003, 0.0015, 0.004])
def test_step_that_cannot_sample_at_the_shared_rate_is_refused(step):
    with pytest.raises(contract.ContractError, match="does not divide"):
        contract.sampling(step)


def test_adapter_is_semver_plus_full_commit_from_git(result):
    a = result["adapter"]
    assert contract.SEMVER.match(a["version"]) and not contract.SHA.match(a["version"])
    if (Path(__file__).parents[1] / ".git").exists():
        assert a["provenance"] == "git" and re.fullmatch(r"[0-9a-f]{40}", a["commit"])


def test_adapter_outside_a_checkout_uses_package_provenance_or_null(tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    pkg = tmp_path / "site-packages" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "_build_provenance.json").write_text(json.dumps({"identity": {"commit": sha}}))
    a = contract.adapter("x", pkg, "1.2.0", "recorded-build")
    assert (a["version"], a["commit"], a["provenance"]) == ("1.2.0", sha, "package")
    b = contract.adapter("x", pkg, "0.0.0-unknown", "unavailable")
    assert b["commit"] is None and "provenance" not in b
    with pytest.raises(contract.ContractError, match="not a semantic version"):
        contract.adapter("x", pkg, sha, "checkout")
    full = contract.schema()
    schema = {**full["properties"]["adapter"], "$defs": full["$defs"]}
    import jsonschema

    for record in (a, b):
        jsonschema.Draft202012Validator(schema).validate(record)


def test_unsupported_record_validates(result):
    record = {**result, "status": "UNSUPPORTED", "feature": "x", "quantities": []}
    contract.validate(record)
    bad = {k: v for k, v in record.items() if k != "feature"}
    with pytest.raises(contract.ContractError):
        contract.validate(bad)
