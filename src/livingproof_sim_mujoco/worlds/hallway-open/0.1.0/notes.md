## Purpose

The first shared world for the MuJoCo and Gazebo simulators: one straight
hallway, open at both ends, with no doors. It lets a verifier show that both
engines loaded the same dimensions, places and collision boundaries using a
plain sphere probe. No robot or controller is involved; "G1 walks from the
entrance to the exit" is a later story.

Stories that use it (robot-capability-specs v1.3.0):

1. An integrator selects the open hallway by name and version.
2. A scenario author uses the declared hallway places.
3. A verifier compares MuJoCo and Gazebo hallway evidence.
4. A reviewer inspects retained hallway results from both engines.
5. A consumer meets an unsupported hallway feature.

## Changelog

- `0.1.0`: first release, from `specs/worlds/hallways.md` in
  robot-capability-specs v1.3.0.

## Simulator support

| Simulator | Status | Known differences |
| --- | --- | --- |
| MuJoCo | Not yet loaded; adapter pending in livingproof-sim-mujoco | None recorded |
| Gazebo | Not yet loaded; adapter pending in livingproof-sim-gazebo | None recorded |

All geometry is boxes, which both engines support, so no feature is expected to
be `UNSUPPORTED`. An adapter that cannot load a declared feature must fail with
`UNSUPPORTED` naming it. Equivalent geometry does not imply identical physics.
