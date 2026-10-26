#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p artifacts
build=(docker build --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy)
if [ -n "${BUILD_CA_FILE:-}" ]; then build+=(--secret "id=ca,src=$BUILD_CA_FILE"); fi
# Python wheel install has no access to the source checkout or network.
"${build[@]}" --target runtime --build-arg VERSION="${VERSION:-0.0.0}" -t "livingproof-sim-mujoco:${VERSION:-0.0.0}" .
docker run --rm --network none --entrypoint python "livingproof-sim-mujoco:${VERSION:-0.0.0}" -c 'import json,subprocess; d=json.loads(subprocess.check_output(["livingproof-sim-mujoco","--steps","100"])); assert d["steps"]==100 and d["final_qpos"][2]<d["initial_qpos"][2]; assert d["identity"]["provenance"]=="recorded-build"; print(json.dumps(d))' > artifacts/runtime.json
# Verify that Docker stop reaches the command running as PID 1.
cid=$(docker run -d --network none "livingproof-sim-mujoco:${VERSION:-0.0.0}" --steps 1000000000 --timeout 30)
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
for attempt in {1..100}; do
    if docker logs "$cid" 2>&1 | grep -q 'model loaded'; then break; fi
    sleep 0.1
done
docker logs "$cid" 2>&1 | grep -q 'model loaded'
docker stop --time 3 "$cid" >/dev/null
test "$(docker wait "$cid")" = 143
docker logs "$cid" > artifacts/runtime-signal.json 2> artifacts/runtime-signal.log
docker rm "$cid" >/dev/null
trap - EXIT
# Real Debian install in a clean distribution, then network disabled execution.
"${build[@]}" -f tools/Dockerfile.debian -t livingproof-sim-mujoco-debian-test .
docker run --rm --network none livingproof-sim-mujoco-debian-test --steps 100 > artifacts/debian.json
python_check='import json,sys; d=json.load(open(sys.argv[1])); assert d["steps"]==100 and d["final_qpos"][2]<d["initial_qpos"][2]; assert d["identity"]["provenance"]=="recorded-build"'
docker run --rm --network none --user "$(id -u):$(id -g)" -v "$PWD/artifacts:/evidence:ro" --entrypoint python "livingproof-sim-mujoco:${VERSION:-0.0.0}" -c "$python_check" /evidence/debian.json
# Full CLI tests against the installed wheel, without source checkout or network.
"${build[@]}" --target runtime-test -t livingproof-sim-mujoco-runtime-test .
docker run --rm --network none -v "$PWD/artifacts:/reports" --user "$(id -u):$(id -g)" livingproof-sim-mujoco-runtime-test --junitxml=/reports/installed.xml
"${build[@]}" -f tools/Dockerfile.debian-test -t livingproof-sim-mujoco-debian-tests .
docker run --rm --network none -v "$PWD/artifacts:/reports" --user "$(id -u):$(id -g)" livingproof-sim-mujoco-debian-tests --junitxml=/reports/debian-tests.xml
