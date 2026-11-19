#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# Reuse the pinned development image for its Python/uv and headless drivers.
# Mount only built wheels and this check, never the checkout or dependency volumes.
image=$(docker compose config --images | head -n 1)
test -n "$image"
args=(--rm --entrypoint bash -e MUJOCO_GL=osmesa
  -v "$PWD/dist:/wheels:ro"
  -v "$PWD/tools/check_media_install.py:/check_media_install.py:ro")
if [[ -n ${BUILD_CA_FILE:-} ]]; then
  args+=(-v "$BUILD_CA_FILE:/run/ca.pem:ro" -e SSL_CERT_FILE=/run/ca.pem -e GIT_SSL_CAINFO=/run/ca.pem)
fi
docker run "${args[@]}" "$image" -eu -o pipefail -c '
  cd /tmp
  uv --version
  uv venv /tmp/media
  wheel=$(find /wheels -maxdepth 1 -name "*.whl")
  test -n "$wheel"
  uv pip install --python /tmp/media/bin/python "$wheel[media]"
  uv pip check --python /tmp/media/bin/python
  /tmp/media/bin/python /check_media_install.py
  # The allowed imageio release from the original reproduction also works.
  uv pip install --python /tmp/media/bin/python "imageio==2.37.0"
  /tmp/media/bin/python /check_media_install.py
  uv venv /tmp/rejected
  printf "pillow==11.3.0\n" > /tmp/vulnerable.txt
  if uv pip install --dry-run --python /tmp/rejected/bin/python -c /tmp/vulnerable.txt "$wheel[media]" > /tmp/conflict.log 2>&1; then
    cat /tmp/conflict.log
    echo "ERROR: vulnerable Pillow was accepted" >&2
    exit 1
  fi
  cat /tmp/conflict.log
  grep -q "pillow>=13.0.0" /tmp/conflict.log
  grep -q "pillow==11.3.0" /tmp/conflict.log
  echo "Vulnerable Pillow constraint rejected by the built media extra"
' 2>&1 | tee artifacts/media-install.log
