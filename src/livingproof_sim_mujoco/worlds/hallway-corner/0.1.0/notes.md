## Purpose

The second shared world, one step harder than `hallway-open`: the robot walks
down one hallway, turns left where it meets a second hallway, and reaches the far
end of the second hallway. Both hallways have the same 1.50 m clear width,
2.50 m walls and 0.15 m wall thickness as `hallway-open`, and the route from
`spawn-entrance` to `exit` is 5.5 m long, the same as in `hallway-open`, so the
turn is the only thing that changes.

The `corner` region is the 1.50 m square where the two hallways meet. The
`route-centreline` runs down the middle of the first hallway to the centre of
the corner square, then down the middle of the second hallway to the exit.

Stories that use it ([robot-capability-specs #40](https://github.com/Living-Proof-Labs/robot-capability-specs/issues/40)):

1. A scenario author selects the corner hallway by name and version.
2. A scenario uses the declared places: `entrance`, `spawn-entrance`, `corner`,
   `exit` and `route-centreline`.
3. A verifier checks that the base passes through `corner` before it crosses
   `exit`, and measures its distance from `route-centreline`.

## Changelog

- `0.1.0`: first release, for the `walk-corner` scenario (robot-capability-specs
  #40, Part B step 10), meeting world contract 0.3.0 "Any layout" in
  robot-capability-specs v2.2.2.

## Simulator support

| Simulator | Status | Known differences |
| --- | --- | --- |
| MuJoCo | Not yet pinned; livingproof-sim-mujoco pins it in #40 Part B step 11 | None recorded |
| Gazebo | Not yet pinned | None recorded |

All geometry is axis-aligned boxes, which both engines support, so no feature is
expected to be `UNSUPPORTED`. An adapter that cannot load a declared feature must
fail with `UNSUPPORTED` naming it. Equivalent geometry does not imply identical
physics.
