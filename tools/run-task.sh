#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The development image marks its prepared toolchain explicitly; do not infer
# it from Docker availability (a misconfigured host should still fail clearly).
set -euo pipefail
if [[ ${LIVINGPROOF_TOOLING:-} == 1 ]]; then
  if [[ $1 == test:docs || $1 == test:preview ]]; then
    task -t Taskfile.runtime.yml docs:browser
  fi
  exec task -t Taskfile.runtime.yml "$@"
fi
if ! command -v docker >/dev/null; then
  echo 'Host tasks require Docker. Inside an already prepared environment use: task -t Taskfile.runtime.yml setup (then verify).' >&2
  exit 1
fi
args=(compose run --rm)
[[ $1 != setup ]] || args+=(--build)
if [[ $1 == test:docs || $1 == test:preview ]]; then
  args+=(-e PLAYWRIGHT_BROWSERS_PATH=/opt/mujoco-cache/browsers)
  docker "${args[@]}" tooling task -t Taskfile.runtime.yml docs:browser
fi
exec docker "${args[@]}" tooling task -t Taskfile.runtime.yml "$@"
