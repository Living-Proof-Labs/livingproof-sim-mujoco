# livingproof-sim-mujoco

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
capability PASS. There is no viewer, robot control, adapter or general runner.
Use trusted local models; native model parsing is not a security sandbox.

The wall deadline covers startup, loading and stepping. Shutdown first requests
cooperative exit, then terminates/kills the isolated worker with at most two
seconds of grace. SIGINT/SIGTERM return 130/143, timeout returns 124, invalid
input or engine errors return 2. JSON includes actual steps, simulation time,
first 16 initial/final qpos values, model dimensions, engine and build identity.
Native warnings fail execution. XML includes resolve using MuJoCo's native loader.

## Development

Docker and Task 3.54.0 are host prerequisites. `task setup`, `task verify`,
`task package`, and `task smoke` use the repository Docker environment. The
.devcontainer configuration reuses its development target. In that prepared
environment use `task -t Taskfile.runtime.yml setup` and `verify`.
`task docs` generates [the command reference](docs/command.md) from real help.

Runtime dependencies are hash-locked in uv.lock; MuJoCo 3.17.0 is exact-pinned.
The engine wheel bundles the native engine. Shared identity uses the released
[livingproof-common v1.0.0 wheel](https://github.com/Living-Proof-Labs/livingproof-common/releases/tag/v1.0.0),
verified as the latest release on 2026-10-26. No copied version algorithm.
See docs/evidence.md for actual tested platforms and remaining release gates.
