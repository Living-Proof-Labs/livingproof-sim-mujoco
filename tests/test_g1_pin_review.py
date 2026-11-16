# SPDX-License-Identifier: Apache-2.0
"""Vera review checks for the livingproof-unitree-g1 v2.0.2 pin (#54).

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
PKG = "livingproof-unitree-g1"
PKG_TAG = "v2.0.2"
PKG_COMMIT = "07ab4f6a9f880f1f61f1abf4104d64bc6f933263"
OTHER_ROBOTS = ("livingproof-unitree-go1", "livingproof-booster-t1")


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _source_tag(package: str) -> str:
    return _pyproject()["tool"]["uv"]["sources"][package]["tag"]


def test_pyproject_and_lock_name_the_published_g1_v2_0_2_release():
    assert _source_tag(PKG) == PKG_TAG
    assert locked_robot_source(PKG) == (PKG_TAG, PKG_COMMIT)


def test_the_other_robots_keep_their_own_lock_revisions():
    mine = locked_robot_source(PKG)
    for package in OTHER_ROBOTS:
        tag, commit = locked_robot_source(package)
        assert _source_tag(package) == tag, package
        assert (tag, commit) != mine, package


def test_dependency_metadata_omits_the_robots_sim_mujoco_pin():
    metas = [
        item for item in _pyproject()["tool"]["uv"]["dependency-metadata"] if item["name"] == PKG
    ]
    assert len(metas) == 1
    assert all("livingproof-sim-mujoco" not in req for req in metas[0]["requires-dist"])


def test_installed_g1_wheel_is_the_locked_revision():
    dist = distribution("livingproof-unitree-g1")
    recorded = json.loads(Path(dist._path, "direct_url.json").read_text())
    assert recorded["url"] == f"https://github.com/Living-Proof-Labs/{PKG}"
    assert recorded["vcs_info"]["requested_revision"] == PKG_TAG
    assert recorded["vcs_info"]["commit_id"] == PKG_COMMIT
