# SPDX-License-Identifier: Apache-2.0
"""Preserve the host contract while routing prepared-container commands directly."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("task,build", [("setup", True), ("test", False), ("package", False)])
def test_host_dispatch_keeps_compose(tmp_path, task, build):
    """Given a host, contributor tasks still invoke the Compose tooling service."""
    log = tmp_path / "calls"
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TASK_TEST_LOG"\n')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "TASK_TEST_LOG": str(log)}
    env.pop("LIVINGPROOF_TOOLING", None)
    subprocess.run(["bash", "tools/run-task.sh", task], env=env, check=True)
    assert log.read_text().splitlines() == [
        "compose",
        "run",
        "--rm",
        *(["--build"] if build else []),
        "tooling",
        "task",
        "-t",
        "Taskfile.runtime.yml",
        task,
    ]


def test_bare_task_lists_commands_without_docker(tmp_path):
    """Given only Task on PATH, bare task succeeds and advertises the preview."""
    shutil.copy2(shutil.which("task"), tmp_path / "task")
    env = {**os.environ, "PATH": str(tmp_path)}
    result = subprocess.run([str(tmp_path / "task")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "preview" in result.stdout
    assert "setup" in result.stdout


def test_devcontainer_exposes_view_port_without_public_visibility():
    """Given the container definition, the view port is forwarded without public access configuration."""
    config = json.loads(Path(".devcontainer/devcontainer.json").read_text())
    assert config["forwardPorts"] == [8000]
    assert config["postCreateCommand"] == "task setup"
    assert "visibility" not in config["portsAttributes"]["8000"]


def _runtime_test_commands():
    import yaml

    tasks = yaml.safe_load(Path("Taskfile.runtime.yml").read_text())["tasks"]
    return [command for command in tasks["test"]["cmds"] if isinstance(command, str)]


def test_the_zero_command_stand_runs_in_ci_with_egl():
    """#38: the evidence tests, the zero-command stand included, run headless on EGL."""
    evidence = [
        command for command in _runtime_test_commands() if "tests/test_evidence.py" in command
    ]
    assert evidence == [
        "MUJOCO_GL=egl uv run --no-sync pytest tests/test_evidence.py "
        "--junitxml=artifacts/evidence.xml"
    ]
    assert not any("zero_command" in command for command in _runtime_test_commands())
    reference = Path("docs/tasks.md").read_text()
    assert evidence[0] in reference and "zero_command" not in reference


def test_the_development_image_has_a_headless_egl_driver():
    """MUJOCO_GL=egl needs libEGL, Mesa's EGL driver and a GL library PyOpenGL can load.

    PyOpenGL's EGL platform loads libOpenGL or libGL; without one, ``import mujoco``
    fails with "'NoneType' object has no attribute 'glGetError'" (CI run 43861442755).
    """
    dockerfile = Path("Dockerfile").read_text()
    development = dockerfile.split("AS development", 1)[1].split("\nFROM ", 1)[0]
    packages = set(development.replace("\\\n", " ").split())
    assert {"libegl1", "libegl-mesa0", "libgl1-mesa-dri", "libopengl0", "libgl1"} <= packages


# These modules have their own Taskfile tasks, not the main CI ``test`` task.
# Same collection guard as livingproof-sim-mujoco#88 (423d428), landed here
# because that PR is still open. A glob rewrite of task test would fight #88's
# explicit list and the tests that require filenames in the task commands.
_SEPARATE_TEST_MODULES = frozenset(
    {
        "tests/test_preview.py",
    }
)


def test_runtime_test_task_collects_every_unit_test_module():
    """A new tests/test_*.py must be listed in the CI test task.

    CI calls pytest on an explicit file list. A module that is not on that
    list can fail locally and still leave MuJoCo tests green (livingproof-sim-mujoco#88).
    tests/docs/ is collected by ``task test:docs``; test_preview.py by ``task test:preview``.
    """
    commands = "\n".join(_runtime_test_commands())
    missing = []
    for path in sorted(Path("tests").glob("test_*.py")):
        relative = path.as_posix()
        if relative in _SEPARATE_TEST_MODULES:
            continue
        if relative not in commands:
            missing.append(relative)
    assert missing == []
