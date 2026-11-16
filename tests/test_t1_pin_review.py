# SPDX-License-Identifier: Apache-2.0
"""Vera review checks for the livingproof-booster-t1 pin (#53).

These confirm pyproject, the lock and the installed package agree on one tag
and commit, that the tag is at least the reviewed floor, that the other robots
keep their own revisions (without hard-coding those tags, so a later sibling
bump still passes), and that dependency-metadata still drops the robot's own
livingproof-sim-mujoco pin so this checkout is what runs. LIVINGPROOF_NETWORK=1
also checks that the locked commit is the peeled published tag.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest
import tomllib
from importlib.metadata import distribution
from pathlib import Path

from locked_source import locked_robot_source

ROOT = Path(__file__).resolve().parents[1]
PKG = "livingproof-booster-t1"
# Reviewed floor: do not freeze the exact current tag or commit.
MIN_TAG = (2, 0, 2)
OTHER_ROBOTS = ("livingproof-unitree-g1", "livingproof-unitree-go1")
REPO = f"https://github.com/Living-Proof-Labs/{PKG}.git"


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _source_tag(package: str) -> str:
    return _pyproject()["tool"]["uv"]["sources"][package]["tag"]


def _parse_tag(tag: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
    assert match, tag
    return tuple(int(part) for part in match.groups())


def _peeled_tag_commit(tag: str) -> str:
    listing = subprocess.check_output(
        ["git", "ls-remote", "--tags", REPO, f"refs/tags/{tag}"],
        text=True,
        timeout=30,
    )
    peeled = None
    direct = None
    for line in listing.splitlines():
        sha, ref = line.split("\t", 1)
        if ref.endswith("^{}"):
            peeled = sha
        else:
            direct = sha
    assert peeled or direct, listing
    return peeled or direct


def test_pyproject_lock_and_installed_release_agree():
    tag = _source_tag(PKG)
    assert _parse_tag(tag) >= MIN_TAG, tag
    lock_tag, lock_commit = locked_robot_source(PKG)
    assert lock_tag == tag
    assert re.fullmatch(r"[0-9a-f]{40}", lock_commit), lock_commit
    dist = distribution(PKG)
    recorded = json.loads(Path(dist._path, "direct_url.json").read_text())
    assert recorded["url"] == f"https://github.com/Living-Proof-Labs/{PKG}"
    assert recorded["vcs_info"]["requested_revision"] == tag
    assert recorded["vcs_info"]["commit_id"] == lock_commit


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


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
def test_the_locked_commit_is_the_peeled_published_tag():
    tag, commit = locked_robot_source(PKG)
    assert commit == _peeled_tag_commit(tag)
