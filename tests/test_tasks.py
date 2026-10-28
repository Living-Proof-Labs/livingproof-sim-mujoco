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
