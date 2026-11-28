# SPDX-License-Identifier: Apache-2.0
"""numpy is a range, not an exact pin, so robot repos can take any simulator release (#236).

The robot repositories pin their own numpy (2.5.4 when this was written) and depend on
this package. An exact numpy here makes `uv sync` unresolvable in each of them whenever
the two pins differ. The development lock still holds one tested version.
"""

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
import pytest

ROOT = Path(__file__).resolve().parents[1]
# What the robot repositories pin today (2.5.4) and the neighbours Renovate moves them to.
ROBOT_PINS = ["2.5.4", "2.5.5", "2.6.0"]
TESTED = "2.4.6"


def _project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _numpy(dependencies):
    found = [Requirement(item) for item in dependencies if Requirement(item).name == "numpy"]
    assert len(found) == 1
    return found[0]


def _lock():
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def test_runtime_numpy_is_a_range_that_admits_the_robot_pins_and_the_tested_version():
    requirement = _numpy(_project()["project"]["dependencies"])
    specifier = requirement.specifier
    assert not any(item.operator in ("==", "===") for item in specifier), str(specifier)
    for version in [TESTED, *ROBOT_PINS]:
        assert Version(version) in specifier, version


def test_runtime_numpy_keeps_a_floor_and_a_major_ceiling():
    specifier = _numpy(_project()["project"]["dependencies"]).specifier
    assert Version("2.4.5") not in specifier
    assert Version("1.26.4") not in specifier
    assert Version("3.0.0") not in specifier


def test_the_lock_records_the_same_requirement_and_one_version_inside_it():
    requirement = _numpy(_project()["project"]["dependencies"])
    lock = _lock()
    package = next(p for p in lock["package"] if p["name"] == "livingproof-sim-mujoco")
    recorded = next(r for r in package["metadata"]["requires-dist"] if r["name"] == "numpy")
    assert Requirement(f"numpy{recorded['specifier']}").specifier == requirement.specifier
    locked = [p for p in lock["package"] if p["name"] == "numpy"]
    assert len(locked) == 1
    assert Version(locked[0]["version"]) in requirement.specifier


def test_renovate_widens_numpy_instead_of_rewriting_the_range_to_a_pin():
    """The preset's rangeStrategy pin rewrote the Pillow floor (#197, #201); not numpy too."""
    rules = json.loads((ROOT / "renovate.json").read_text(encoding="utf-8"))["packageRules"]
    matching = [
        rule
        for rule in rules
        if "numpy" in rule.get("matchPackageNames", [])
        and "pep621" in rule.get("matchManagers", [])
        and "project.dependencies" in rule.get("matchDepTypes", [])
    ]
    assert len(matching) == 1
    assert matching[0]["rangeStrategy"] == "widen"


@pytest.mark.parametrize("extra", ["torch", "media"])
def test_no_extra_adds_another_numpy_pin(extra):
    extras = _project()["project"]["optional-dependencies"][extra]
    assert [item for item in extras if Requirement(item).name == "numpy"] == []
