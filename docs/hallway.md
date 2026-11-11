# Hallway fixture adapter

`livingproof_sim_mujoco.hallway` loads `hallway-open` 0.1.0 from
[livingproof-worlds v1.1.0](https://github.com/Living-Proof-Labs/livingproof-worlds/tree/v1.1.0/worlds/hallway-open/0.1.0)
(commit `9b3e7119aa93a3370383227fb30bd39a9dd72b4b`) and checks it against
[hallways.md](https://github.com/Living-Proof-Labs/robot-capability-specs/blob/v1.3.0/specs/worlds/hallways.md)
from robot-capability-specs v1.3.0. No robot is involved.

## Pinning and checksums

Only `world.json`, `notes.md` and `release.json` are vendored, unchanged, under
`src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/`. `worlds/pins.json` records the tag, the full
commit and the SHA-256 of `release.json`. On every load the adapter checks:

- `release.json` against the pinned SHA-256;
- `world.json` and `notes.md` against `release.json`.

`release.json` also lists the generated docs (`README.md`, `world.svg`,
`world-light.svg`, `world-dark.svg`). These are renders of `world.json`, so they are
**skipped on purpose**: they are not vendored and not checksum-checked. A test states
this. Loading fails with a `FixtureError` when:

- a vendored file's checksum does not match;
- a vendored file is missing;
- any other file is present (including a generated doc);
- `release.json` lists a file the adapter doesn't know.

An unknown fixture or version is reported, never substituted.

## Translation

`world.json` box geometry is copied verbatim into MJCF geoms. Named places become sites:
segments as a reference site plus `name/0` and `name/1`, polylines as `name/i`, the region as a
flat box site, and poses as sites whose quaternion encodes the yaw.
MuJoCo-only settings (timestep, integrator, solref, solimp, friction, probe drive) live in
`worlds/hallway-open-0.1.0.mujoco.json` and never change geometry or places.
Any declared feature without a mapping (another format, a non-box geometry, an unknown role,
place type or field) raises `UNSUPPORTED: <fixture> <version>: <feature>`.

## Checks

- **Compile:** `mujoco.MjModel.from_xml_string`.
- **Dimensions, frame and places:** read back from the compiled model (`geom_*`, `site_*`),
  within 0.01 m and 0.5°.
- **Floor:** a 0.10 m sphere is released from rest at `spawn-centre`, centre `z = 0.75`;
  penetration may not exceed 0.01 m.
- **Side walls:** the sphere is driven by velocity servos on slide joints, at most 0.25 m/s,
  toward `+y` then `-y` for 5 s each; its surface may not pass the inner face by more than
  0.01 m, and it must actually touch the wall.
- **Open ends:** the sphere is driven along the centreline to `x = ±2.80` and may touch only
  the floor. The quantities are named `collision/open_end/+x/…` and `…/-x/…`, with the target
  in `canonical`.
- **Reset:** after all runs, `mj_resetData` + `mj_forward` restores the initial state. The
  `reset/*` quantities read the world back from `MjData` and confirm that:
  - every geom's world centre and size is unchanged (`reset/geometry/<name>/max_shift`);
  - every place marker is unchanged (`reset/places/max_shift`);
  - the probe is back at its start, `spawn-centre` at `z = 0.75` (`reset/probe/x|y|z`).

  All are within 0.01 m (hallways.md: "Reset restores the geometry, frame, and initial world state").

**Probe parameters** are read from the shared `hallway-probe.json`. They cover the radius,
density, friction, the 0.25 m/s cap, gravity, start height, settle and drive times, the targets
and the 500 Hz sample rate. The MuJoCo extension may not contradict them: a different
`probe.radius`, `probe.density`, `probe.friction[0]`, `probe.max_speed` or `option.gravity` is
UNSUPPORTED.

**Sampling.** Probe samples are taken at the shared 500 Hz, which is every `sample_stride`-th
physics step. With MuJoCo's 0.002 s step, that is every step. A step that does not divide
the 2 ms sample period, such as 0.003 s, is UNSUPPORTED. The result `settings` records
`physics_step`, `sample_rate` and `sample_stride`.

**Probe modelling (demo choices).** The sphere slides rather than rolls: it is mounted on x,
y and z slide joints with no rotational freedom. Its friction is 0.1 (with priority 1, so this
value wins over the floor's 1.0). It is moved by speed control (velocity servos) capped at
0.25 m/s. These are demonstration choices made to probe collision boundaries cleanly, not a
model of any robot or of rolling physics. Every physics step is sampled (500 Hz).

## Result format

`task hallway` writes `artifacts/hallway/result.json` in the shared format
`livingproof-world-result/0.2`. The contract lives in robot-capability-specs, and three files
are vendored from it, unchanged, into `src/livingproof_sim_mujoco/contract/`:

- `hallway-result.schema.json`
- `hallway-quantities.json`
- `hallway-probe.json`

`contract/pins.json` is the single place that holds the specs release tag, its commit and each file's
SHA-256. Every load checks them. To move to a newer specs commit, run
`python tools/pin_contract.py <specs checkout> <tag>`; `--check` confirms the vendored files match the tag. Currently pinned to v2.0.2 (16524cbf). The three hallway contract files are unchanged from v1.4.0.

Each quantity uses its shared name, `kind`, `unit` and `comparison`, and records
`canonical`, `measured`, `deviation` and `tolerance`. The comparison decides `pass`:

- `within`: |measured − canonical| ≤ tolerance;
- `at_most`: measured ≤ canonical + tolerance. This is used for penetration, distance past a
  wall face and the speed cap.

`adapter.version` is a semantic version. `adapter.commit` is the full commit: from git in a
checkout (`provenance: git`), or from `_build_provenance.json` in an installed package
(`provenance: package`). It is `null` only when neither knows it. The record is validated
against the vendored schema and the required names before it is written.
The bundle also contains `model.xml`, `versions.json`, `overhead.png` (orthographic, straight
down, 5.0 m visible height, so the parallel walls render parallel) and `entrance-view.png`
(perspective, from the entrance).
Screenshot failures are recorded as `BLOCKED` with the error, never faked.
