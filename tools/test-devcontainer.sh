#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p artifacts
devcontainer up --workspace-folder "$PWD" --log-level debug 2>&1 | tee artifacts/devcontainer-setup.log
# A successful `up` alone is insufficient: missing home/user entries can cause
# lifecycle marker writes to fail and silently skip postCreateCommand.
devcontainer exec --workspace-folder "$PWD" sh -ec '
  test "$(id -u)" -ne 0
  getent passwd "$(id -u)"
  test -w "$HOME"
  test -f "$HOME/.devcontainer/.postCreateCommandMarker"
  test -f /opt/mujoco-venv/pyvenv.cfg
  uv run --no-sync pytest tests/test_cli.py --junitxml=artifacts/devcontainer.xml
'
