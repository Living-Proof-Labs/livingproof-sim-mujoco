# Agent instructions

Read [collaboration-guidelines v6.0.0](https://github.com/Living-Proof-Labs/collaboration-guidelines/blob/v6.0.0/AGENTS.md)
and its applicable procedures before changing this repository.
Local additions: keep this a concrete native MuJoCo CLI; no robot adapters,
umbrella runner, Unitree SDK or motor control. Use released livingproof-common
v1.0.0. Original repository content is Apache-2.0, including documentation.
Substantive implementation stays in a draft PR until independent review.
Do not enable new registry/security permissions or alter organization settings.

Release exception to the pinned shared maintenance-release default: updates
limited to the three released robot composition-test source pins and their lock
identities run normal verification but do not issue a new simulator release.
See `docs/releases.md`. Runtime/build dependency changes and all other maintenance
remain eligible for the shared semantic-release policy.
