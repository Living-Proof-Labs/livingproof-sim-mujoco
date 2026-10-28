# MuJoCo manual

Start with real headless physics:

```sh
livingproof-sim-mujoco --model builtin:falling --steps 100 --headless
```

The sphere falls from height 1 to approximately 0.801838 in 0.2 simulated seconds.
The command emits JSON; a successful exit is engine evidence, never robot capability evidence.
Use trusted local models. Native parsing is not a security sandbox.
Native engine availability does not establish capability-pipeline integration.

For a browser-visible experience in Codespaces, run `task preview` inside the
devcontainer and open private port 8000. This is rendered playback, not live
interactive simulation; see the [browser playback instructions](verification.md#browser-simulation-playback).

Read the [command reference](command.md), [feature scenarios](features.md),
[development and review guide](verification.md), and [distribution guide](release.md).
[Build context](context.md) identifies this development manual's exact source and installed command.
Historical manuals are not maintained; consult the matching source tag for an older release.

The [public manual](https://docs.livingprooflabs.com/livingproof-sim-mujoco/) follows
`main`; pull-request artifacts are review previews retained for 14 days.
Related manuals: [Living Proof](https://docs.livingprooflabs.com/livingproof/) and
[collaboration guidelines](https://github.com/Living-Proof-Labs/collaboration-guidelines/tree/v6.0.0).
[Source repository](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco).

For contribution, support and license details, see [Contribute](contribute.md).
[Documentation home](https://docs.livingprooflabs.com/) owns the complete manual directory.
