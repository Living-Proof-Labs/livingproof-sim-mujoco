# Hallway fixture adapter

`livingproof_sim_mujoco.hallway` loads `hallway-open` 0.1.0 from
[livingproof-worlds v1.1.0](https://github.com/Living-Proof-Labs/livingproof-worlds/tree/v1.1.0/worlds/hallway-open/0.1.0)
(commit `9b3e7119aa93a3370383227fb30bd39a9dd72b4b`) and checks it against
[hallways.md](https://github.com/Living-Proof-Labs/robot-capability-specs/blob/v1.3.0/specs/worlds/hallways.md)
from robot-capability-specs v1.3.0. No robot is involved.

## Pinning and checksums

The released files are vendored unchanged under `src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/`.
`worlds/pins.json` records the tag, full commit and the SHA-256 of `release.json`.
Loading fails with a `FixtureError` naming the file, expected and actual hash when
`release.json` or any listed file differs, when a listed file is missing, or when an
unlisted file is present. Unknown identifiers or versions are reported, never substituted.

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

The probe uses x, y and z slide joints (no rotation) and its own friction (priority 1), so the
drive is not fighting floor friction. Every physics step is sampled (500 Hz).

## Result format

`task hallway` writes `artifacts/hallway/result.json` in format
`livingproof-world-result/0.1` ([schema](hallway-result.schema.json)). Each quantity
records `canonical`, `measured`, `deviation` (measured − canonical), `tolerance` and `pass`,
under an engine-neutral `name`, so a Gazebo record with the same names can be joined for the
cross-engine comparison. The record also names the fixture, world-contract revision,
catalog tag and commit, pinned worlds release, engine and adapter versions.
The bundle also contains `model.xml`, `versions.json`, `overhead.png` and `entrance-view.png`.
Screenshot failures are recorded as `BLOCKED` with the error, never faked.
