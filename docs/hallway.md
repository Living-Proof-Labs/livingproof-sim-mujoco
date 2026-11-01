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
  the floor.

**Probe modelling (demo choices).** The sphere slides rather than rolls: it is mounted on x,
y and z slide joints with no rotational freedom. Its friction is 0.1 (with priority 1, so this
value wins over the floor's 1.0). It is moved by speed control (velocity servos) capped at
0.25 m/s. These are demonstration choices made to probe collision boundaries cleanly, not a
model of any robot or of rolling physics. Every physics step is sampled (500 Hz).

## Result format

`task hallway` writes `artifacts/hallway/result.json` in format
`livingproof-world-result/0.1` ([schema](hallway-result.schema.json)). Each quantity
records `canonical`, `measured`, `deviation` (measured − canonical), `tolerance` and `pass`,
under an engine-neutral `name`, so a Gazebo record with the same names can be joined for the
cross-engine comparison. The record also names the fixture, world-contract revision,
catalog tag and commit, pinned worlds release, engine and adapter versions.
The bundle also contains `model.xml`, `versions.json`, `overhead.png` (orthographic, straight
down, 5.0 m visible height, so the parallel walls render parallel) and `entrance-view.png`
(perspective, from the entrance).
Screenshot failures are recorded as `BLOCKED` with the error, never faked.
