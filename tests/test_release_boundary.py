# SPDX-License-Identifier: Apache-2.0
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_boundary", ROOT / "tools/release_boundary.py")
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


def snapshots():
    project = '[build-system]\nrequires = []\n[project]\ndependencies = ["mujoco==3.17.0"]\n'
    project += '[dependency-groups]\nrobots = ["livingproof-unitree-g1", "livingproof-unitree-go1", "livingproof-booster-t1"]\n'
    project += '[tool.uv.sources]\n'
    lock = '[[package]]\nname = "numpy"\nversion = "2.4.6"\n'
    lock += '[[package]]\nname = "livingproof-sim-mujoco"\n[package.metadata.requires-dev]\nrobots = [\n'
    for name in sorted(boundary.ROBOTS):
        url = f"https://github.com/Living-Proof-Labs/{name}"
        project += f'{name} = {{ git = "{url}", tag = "v2.2.8" }}\n'
        lock += f'{{ name = "{name}", git = "{url}?tag=v2.2.8" }},\n'
    lock += ']\n'
    for name in sorted(boundary.ROBOTS):
        url = f"https://github.com/Living-Proof-Labs/{name}"
        lock += f'[[package]]\nname = "{name}"\nversion = "0.1.0"\n'
        lock += f'source = {{ git = "{url}?tag=v2.2.8#{"a" * 40}" }}\n'
    return {"pyproject.toml": project, "uv.lock": lock}


def bumped(before):
    return {p: s.replace("v2.2.8", "v2.2.99") for p, s in before.items()}


def test_only_robot_pin_refresh_is_not_a_consumer_release():
    assert boundary.pin_only(snapshots(), bumped(snapshots()))


@pytest.mark.parametrize("path,old,new", [
    ("pyproject.toml", 'mujoco==3.17.0', 'mujoco==3.17.1'),
    ("uv.lock", 'version = "2.4.6"', 'version = "2.4.7"'),
    ("pyproject.toml", 'livingproof-unitree-g1",', 'new-runtime",'),
    ("pyproject.toml", 'tag = "v2.2.8"', 'tag = "main"'),
    ("uv.lock", 'Living-Proof-Labs/livingproof-unitree-g1', 'attacker/livingproof-unitree-g1'),
])
def test_runtime_metadata_and_unknown_sources_still_release(path, old, new):
    before = snapshots()
    after = bumped(before)
    # The branch fixture should replace the bumped tag.
    if new == 'tag = "main"':
        old = 'tag = "v2.2.99"'
    assert old in after[path]
    after[path] = after[path].replace(old, new)
    assert not boundary.pin_only(before, after)


def test_unknown_and_malformed_files_do_not_suppress_release():
    before = snapshots()
    after = bumped(before) | {"src/runtime.py": "changed"}
    assert not boundary.pin_only(before, after)
    assert not boundary.pin_only(before, {"pyproject.toml": "invalid["})
    assert not boundary.pin_only(before, before)


def test_two_repository_release_chain_stops_but_mixed_runtime_work_releases(tmp_path, monkeypatch):
    """Robot release -> simulator composition update -> no simulator release/pin back."""
    original = snapshots()
    simulator = tmp_path / "simulator"
    robot = tmp_path / "robot"
    for repo in (simulator, robot):
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    def commit(repo, files, message):
        for path, text in files.items():
            (repo / path).write_text(text)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", message], check=True)
    commit(robot, {"simulator-pin": "v3.4.1"}, "chore: adopt simulator")
    subprocess.run(["git", "-C", str(robot), "tag", "v2.2.99"], check=True)
    commit(simulator, original, "feat: simulator runtime")
    subprocess.run(["git", "-C", str(simulator), "tag", "v3.4.1"], check=True)
    commit(simulator, bumped(original), "chore(deps): robot v2.2.99")
    monkeypatch.chdir(simulator)
    decision = boundary.classify()
    assert decision["release_needed"] is False
    # No new simulator release means no reciprocal runtime pin update is produced.
    assert boundary.git("tag") == "v3.4.1"
    assert (robot / "simulator-pin").read_text() == "v3.4.1"
    commit(simulator, {"runtime.py": "new runtime"}, "fix: runtime")
    assert boundary.classify()["release_needed"] is True
    (simulator / "runtime.py").unlink()
    commit(simulator, {}, "revert: runtime")
    assert boundary.classify()["release_needed"] is True


def test_workflow_retains_tests_and_gates_only_release():
    import yaml
    jobs = yaml.safe_load((ROOT / ".github/workflows/verify.yml").read_text())["jobs"]
    assert jobs["release"]["needs"] == ["verify", "publish-tests", "release-boundary"]
    assert "release-needed == 'true'" in jobs["release"]["if"]
    assert "release-boundary" not in jobs["verify"].get("needs", [])
    assert jobs["release-boundary"]["permissions"] == {"contents": "read"}
    assert jobs["release"]["with"]["draft-release"] is True


def test_lock_only_refresh_respects_unchanged_runtime_robot_dependency():
    before = snapshots()
    before["pyproject.toml"] = before["pyproject.toml"].replace(
        'dependencies = ["mujoco==3.17.0"]',
        'dependencies = ["mujoco==3.17.0", "livingproof-unitree-g1"]',
    )
    after = dict(before)
    after["uv.lock"] = bumped(before)["uv.lock"]
    assert not boundary.pin_only(before, after)
