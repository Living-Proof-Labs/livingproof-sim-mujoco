# SPDX-License-Identifier: Apache-2.0
"""Vera review checks for the livingproof-unitree-go1 v2.0.2 pin (#52).

These confirm this package is the published v2.0.2 tag, that the lock commit
matches the tag, that the other robots keep their own revisions (without
hard-coding those tags, so a later sibling bump still passes), and that
dependency-metadata still drops the robot's own livingproof-sim-mujoco pin
so this checkout is what runs.
"""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import distribution
from pathlib import Path

from locked_source import locked_robot_source

ROOT = Path(__file__).resolve().parents[1]
GO1 = "livingproof-unitree-go1"
GO1_TAG = "v2.0.2"
GO1_COMMIT = "b89bfa5caa5c14fd612868e8a1e19777a3e03a01"
OTHER_ROBOTS = ("livingproof-unitree-g1", "livingproof-booster-t1")


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _source_tag(package: str) -> str:
    return _pyproject()["tool"]["uv"]["sources"][package]["tag"]


def test_pyproject_and_lock_name_the_published_go1_v2_0_2_release():
    assert _source_tag(GO1) == GO1_TAG
    assert locked_robot_source(GO1) == (GO1_TAG, GO1_COMMIT)


def test_the_other_robots_keep_their_own_lock_revisions():
    go1 = locked_robot_source(GO1)
    for package in OTHER_ROBOTS:
        tag, commit = locked_robot_source(package)
        assert _source_tag(package) == tag, package
        assert (tag, commit) != go1, package


def test_dependency_metadata_omits_the_robots_sim_mujoco_pin():
    metas = [
        item for item in _pyproject()["tool"]["uv"]["dependency-metadata"] if item["name"] == GO1
    ]
    assert len(metas) == 1
    assert all("livingproof-sim-mujoco" not in req for req in metas[0]["requires-dist"])


def test_installed_go1_wheel_is_the_locked_revision():
    dist = distribution("livingproof-unitree-go1")
    recorded = json.loads(Path(dist._path, "direct_url.json").read_text())
    assert recorded["url"] == f"https://github.com/Living-Proof-Labs/{GO1}"
    assert recorded["vcs_info"]["requested_revision"] == GO1_TAG
    assert recorded["vcs_info"]["commit_id"] == GO1_COMMIT
