# SPDX-License-Identifier: Apache-2.0
"""Keep robot composition pin-only updates from issuing consumer releases."""
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROBOTS = {f"livingproof-{name}" for name in ("unitree-g1", "unitree-go1", "booster-t1")}
TAG = r"v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def normalized(text, path):
    """Ignore only existing robot source identities, never runtime/lock metadata."""
    data = tomllib.loads(text)
    if path == "pyproject.toml":
        runtime = data["project"].get("dependencies", []) + data["build-system"]["requires"]
        for requirements in data["project"].get("optional-dependencies", {}).values():
            runtime += requirements
        if any(any(requirement.startswith(name) for name in ROBOTS) for requirement in runtime):
            raise ValueError("robot is a runtime/build dependency, not test-only")
        sources = data["tool"]["uv"]["sources"]
        for name in ROBOTS:
            source = sources[name]
            if source.get("git") != f"https://github.com/Living-Proof-Labs/{name}":
                raise ValueError("unexpected robot origin")
            if not re.fullmatch(TAG, source["tag"]):
                raise ValueError("robot pin is not a release tag")
            source["tag"] = "<robot-release>"
    else:
        found = set()
        for package in data["package"]:
            name = package["name"]
            if name == "livingproof-sim-mujoco":
                for requirement in package["metadata"]["requires-dev"]["robots"]:
                    robot = requirement["name"]
                    if robot not in ROBOTS or not re.fullmatch(
                        rf"https://github.com/Living-Proof-Labs/{robot}\?tag={TAG}",
                        requirement["git"],
                    ):
                        raise ValueError("unexpected robot test requirement")
                    requirement["git"] = "<robot-release>"
            if name not in ROBOTS:
                continue
            source = package["source"]
            pattern = rf"https://github.com/Living-Proof-Labs/{name}\?tag={TAG}#[0-9a-f]{{40}}"
            if not re.fullmatch(pattern, source["git"]):
                raise ValueError("unexpected robot lock source")
            source["git"] = "<robot-release>"
            found.add(name)
        if found != ROBOTS:
            raise ValueError("missing robot lock entries")
    return data


def pin_only(before, after):
    """Classify a pair of snapshots; uncertainty always retains normal release."""
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed or not changed <= {"pyproject.toml", "uv.lock"}:
        return False
    try:
        return all(
            normalized(before[p], p) == normalized(after[p], p)
            for p in ("pyproject.toml", "uv.lock")
        )
    except (KeyError, TypeError, ValueError):
        return False


def classify(head="HEAD"):
    result = {"head": git("rev-parse", head), "release_needed": True}
    try:
        base = git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", head)
        result["base"] = base
        # First-parent edges include the complete merged PR delta. Inspect every
        # edge since the release so a mixed/reverted runtime change cannot hide.
        commits = git("rev-list", "--first-parent", f"{base}..{head}").splitlines()
        result["commits"] = commits
        if not commits:
            result["reason"] = "No unreleased commits; retain semantic-release's no-op decision."
            return result
        for commit in commits:
            parent = git("rev-parse", f"{commit}^1")
            paths = git("diff", "--name-only", parent, commit).splitlines()
            if not paths or not set(paths) <= {"pyproject.toml", "uv.lock"}:
                result["reason"] = "Unreleased changes extend beyond robot composition pins."
                return result
            before = {p: git("show", f"{parent}:{p}") for p in ("pyproject.toml", "uv.lock")}
            after = {p: git("show", f"{commit}:{p}") for p in ("pyproject.toml", "uv.lock")}
            if not pin_only(before, after):
                result["reason"] = "Unreleased metadata includes more than robot source pins."
                return result
        result.update(release_needed=False, reason="Only robot composition source pins changed.")
    except (subprocess.CalledProcessError, ValueError):
        result["reason"] = "History unavailable; retain normal release handling."
    return result


if __name__ == "__main__":
    result = classify()
    print(json.dumps(result, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(f"release-needed={str(result['release_needed']).lower()}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write("## Consumer release boundary\n```json\n")
            stream.write(json.dumps(result, indent=2) + "\n```\n")
