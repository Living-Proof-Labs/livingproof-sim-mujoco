# SPDX-License-Identifier: Apache-2.0
"""Review probes for tests/locked_source.py beyond the follow-the-lock cases.

The head's test_locked_robot_revision_follows_the_lock covers three synthetic
tags for livingproof-unitree-g1. These cover a missing or ambiguous lock entry,
invalid TOML, a source without a tag or commit, a git URL for the wrong package,
and a mixed-version lock that must return the right pin for each robot.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from locked_source import lock_pinning, locked_robot_source

G1 = "livingproof-unitree-g1"
GO1 = "livingproof-unitree-go1"
T1 = "livingproof-booster-t1"
ROBOTS = (G1, GO1, T1)


def test_repository_lock_reads_each_robot_pin():
    """The real uv.lock pin for each robot is that package's git tag and commit."""
    raw = (Path(__file__).resolve().parents[1] / "uv.lock").read_text()
    data = tomllib.loads(raw)
    for package in ROBOTS:
        tag, commit = locked_robot_source(package)
        assert tag
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
        urls = [
            item["source"]["git"]
            for item in data.get("package", [])
            if isinstance(item, dict) and item.get("name") == package
        ]
        assert len(urls) == 1, package
        assert f"?tag={tag}#{commit}" in urls[0]


def test_mixed_lock_returns_the_named_package_only():
    """G1 v2.0.1 next to Go1 and T1 at v2.0.0 does not leak another package's pin."""
    text = (
        lock_pinning(GO1, "v2.0.0", "8" * 40)
        + lock_pinning(G1, "v2.0.1", "d" * 40)
        + lock_pinning(T1, "v2.0.0", "3" * 40)
    )
    assert locked_robot_source(G1, text) == ("v2.0.1", "d" * 40)
    assert locked_robot_source(GO1, text) == ("v2.0.0", "8" * 40)
    assert locked_robot_source(T1, text) == ("v2.0.0", "3" * 40)


def test_missing_package_fails_rather_than_passing_or_skipping():
    with pytest.raises(
        AssertionError, match="uv.lock has 0 git sources for missing-robot, expected one"
    ):
        locked_robot_source("missing-robot", lock_pinning(G1, "v2.0.1", "a" * 40))


def test_empty_lock_fails_as_a_missing_package():
    with pytest.raises(
        AssertionError,
        match="uv.lock has 0 git sources for livingproof-unitree-g1, expected one",
    ):
        locked_robot_source(G1, "")


def test_registry_source_is_not_a_git_pin():
    text = (
        "[[package]]\n"
        f'name = "{G1}"\n'
        'version = "0.1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )
    with pytest.raises(
        AssertionError,
        match="uv.lock has 0 git sources for livingproof-unitree-g1, expected one",
    ):
        locked_robot_source(G1, text)


def test_ambiguous_git_sources_fail():
    text = lock_pinning(G1, "v2.0.1", "d" * 40) + lock_pinning(G1, "v2.0.0", "e" * 40)
    with pytest.raises(
        AssertionError,
        match="uv.lock has 2 git sources for livingproof-unitree-g1, expected one",
    ):
        locked_robot_source(G1, text)


def test_invalid_toml_fails_clearly():
    with pytest.raises(AssertionError, match="uv.lock is not valid TOML"):
        locked_robot_source(G1, "[[package]\nname = ")


@pytest.mark.parametrize(
    "git",
    [
        f"https://github.com/Living-Proof-Labs/{G1}#d4ba8b51b6f87b621d4902c40b0eeb0b60b94f80",
        f"https://github.com/Living-Proof-Labs/{G1}?tag=v2.0.1",
        f"https://github.com/Living-Proof-Labs/{G1}?branch=main#d4ba8b51b6f87b621d4902c40b0eeb0b60b94f80",
        f"https://github.com/Living-Proof-Labs/{G1}?tag=v2.0.1#d4ba8b5",
    ],
)
def test_source_without_a_tag_and_commit_fails(git):
    text = f'[[package]]\nname = "{G1}"\nsource = {{ git = "{git}" }}\n'
    with pytest.raises(
        AssertionError,
        match=rf"uv.lock git source for {re.escape(G1)} has no tag and commit:",
    ):
        locked_robot_source(G1, text)


def test_git_url_for_the_wrong_package_fails():
    text = (
        "[[package]]\n"
        f'name = "{G1}"\n'
        "source = { git = "
        '"https://github.com/Living-Proof-Labs/livingproof-unitree-go1'
        '?tag=v2.0.0#8a20f7cee4118644536c5186fed05e97ff986927" }\n'
    )
    with pytest.raises(
        AssertionError,
        match=rf"uv.lock git source for {re.escape(G1)} points at",
    ):
        locked_robot_source(G1, text)
