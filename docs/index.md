# MuJoCo manual

Start with real headless physics:

```sh
livingproof-sim-mujoco --model builtin:falling --steps 100 --headless
```

The sphere falls from height 1 to approximately 0.801838 in 0.2 simulated seconds.
The command emits JSON; a successful exit is engine evidence, never robot capability evidence.
Use trusted local models. Native parsing is not a security sandbox.

Read the [command reference](command.md), [feature scenarios](features.md),
[development and review guide](verification.md), and [distribution guide](release.md).
[Build context](context.md) identifies this development manual's exact source and installed command.
Historical manuals are not maintained; consult the matching source tag for an older release.

This manual is a review preview. Public Pages deployment and live acceptance are pending.
Related manuals: [Living Proof](https://docs.livingprooflabs.com/livingproof/) and
[collaboration guidelines](https://github.com/Living-Proof-Labs/collaboration-guidelines/tree/v6.0.0).
[Source repository](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco).

[Documentation home](https://docs.livingprooflabs.com/) ·
[Contribute](contribute.md) ·
[Help / report an issue](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/issues) ·
[License: Apache-2.0](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/blob/main/LICENSE)
