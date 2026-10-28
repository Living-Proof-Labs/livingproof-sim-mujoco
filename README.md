# livingproof-sim-mujoco

**Start here: [documentation](https://docs.livingprooflabs.com/livingproof-sim-mujoco/)**

The manual follows `main`; its build context identifies the exact source revision.
[Markdown source](docs/index.md) and 14-day CI preview artifacts remain available.

A small MuJoCo-native command for loading MJCF XML / MJB models and advancing
headless physics. Original code and documentation: Apache-2.0 (see LICENSE).

```sh
livingproof-sim-mujoco --help
livingproof-sim-mujoco --model builtin:falling --steps 100 --headless
livingproof-sim-mujoco --model ./scene.xml --steps 1000 --timeout 10
livingproof-sim-mujoco --version
```

The built-in sphere falls freely under gravity: 100 steps advance 0.2 simulation
seconds and lower its z coordinate. This is engine evidence, never a robot
capability PASS. The installed CLI has no viewer, robot control, adapter or general runner.
The development environment includes a separate rendered playback example below.
Use trusted local models; native model parsing is not a security sandbox.

The wall deadline covers startup, loading and stepping. Shutdown first requests
cooperative exit, then terminates/kills the isolated worker with at most two
seconds of grace. SIGINT/SIGTERM return 130/143, timeout returns 124, invalid
input or engine errors return 2. JSON includes actual steps, simulation time,
first 16 initial/final qpos values, model dimensions, engine and build identity.
Native warnings fail execution. XML includes resolve using MuJoCo's native loader.

## Development

Docker and Task 3.54.0 are host prerequisites. `task setup`, `task verify`,
`task package`, and `task smoke` retain the repository Docker workflow.
Inside the devcontainer, ordinary `task setup`, `task check`, `task test`, and
`task verify` use the prepared tooling directly, without Docker. Bare `task`
shows help successfully. `task image` and `task smoke` remain Docker-host tasks;
`task package` can build products inside the prepared container.

### Try it in browser-based VS Code / Codespaces

Use a Linux amd64 Codespace; simulation and rendering run remotely, so your
local ARM laptop only needs a browser. No ARM product support is claimed.
After opening this branch in a devcontainer, wait for creation setup to finish:

```sh
task
task test
task preview
```

In VS Code's **Ports** tab, find port **8000** (add it if absent), confirm its
visibility is **Private**, and select **Open in Browser**. Use the forwarded
HTTPS URL supplied by Codespaces while signed into the same GitHub account;
your laptop's `localhost:8000` is not the remote service. No public port is needed.
The page shows actual MuJoCo-rendered frames of the built-in sphere falling,
with Play/Pause and a frame slider. This is **recorded playback**, slowed to
one tenth speed, not live interactive simulation, a robot demo, or a desktop
MuJoCo viewer. It records 200 steps / 0.4 simulation seconds; the model has no floor.
Ctrl+C stops serving. Re-run `task preview` to regenerate the frames.

Run `task render` to generate the same 41 PNG frames, `trajectory.json`
(including engine identity and recorded positions), and `index.html` under
`artifacts/preview` without starting a server. CI retains these in the
`mujoco-evidence-amd64` artifact for 14 days. The HTML playback requires a
local HTTP server; individual PNGs and the JSON can be inspected directly.

For JSON physics output:

```sh
uv run --no-sync livingproof-sim-mujoco --model builtin:falling --steps 100 --headless
```

For an existing container built before this change, the immediate command
workaround is `task -t Taskfile.runtime.yml setup`, then
`task -t Taskfile.runtime.yml test`. To use the new ordinary task routing and
CPU graphics library, run **Codespaces: Rebuild Container** after checking out
this branch. Setup needs no sudo. If setup fails, inspect the creation log,
resolve the reported error and rerun `task setup`.

`task preview` serves only generated assets on container loopback port 8000;
it does not serve the checkout. If that port is occupied, stop the other server
or run `uv run --no-sync python tools/preview.py --port 8001` and forward 8001
privately instead. `task test:preview` installs pinned Chromium and tests real
rendering, HTTP isolation and playback controls. `task docs` builds the manual;
`task docs:reference` refreshes generated references.

See [development and verification](docs/verification.md) for tested scope and
[GitHub's port-forwarding guide](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace)
for private browser access. No hosted Codespace or its authentication flow has
been exercised by these local container tests.

Runtime dependencies are hash-locked in uv.lock; MuJoCo 3.17.0 is exact-pinned.
The engine wheel bundles the native engine. Shared identity uses the released
[livingproof-common v1.0.0 wheel](https://github.com/Living-Proof-Labs/livingproof-common/releases/tag/v1.0.0),
verified as the latest release on 2026-10-26. No copied version algorithm.
See docs/evidence.md for actual tested platforms and remaining release gates.
