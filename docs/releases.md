<!-- SPDX-License-Identifier: Apache-2.0 -->
# Robot composition updates and consumer releases

G1, Go1 and Booster T1 are released-tag inputs to this repository's `robots`
dependency group, used for composition and parity tests. They are not simulator
runtime or build dependencies. The robot repositories, in contrast, depend on the
simulator at runtime. Releasing a simulator for every robot test-pin refresh can
therefore cause reciprocal release and update traffic without a runtime change.

`tools/release_boundary.py` makes a narrow local exception to the shared
maintenance-release policy. It reads every first-parent change since the nearest
reachable version tag. Only changes confined to the three robot source tags and
resolved lock source identities qualify for suppression. Source origins must
remain the expected repositories and pins must remain version tags. Dependency
requirements, package versions, transitive locks, runtime code, documentation,
workflow changes, malformed or unavailable history, and mixed changes all retain
normal semantic-release handling. Unknown classifications do not silently skip a
release. Empty history leaves semantic-release to make its normal no-op decision.

This does not disable or remove any tests, packaging checks or documentation
previews. Stable documentation follows a successful simulator release. Only the consumer release job is conditioned on the classification;
its existing verification prerequisites still apply. No existing tag or release
is rewritten. A later genuine release includes the intervening test compositions.
The exact composition remains reproducible from the merged commit's `pyproject.toml`
and `uv.lock`; its normal workflow run, test artifacts and classification job
summary identify the commit, compared release and decision, even when no consumer
tag is created. Workflow artifact retention is finite; the committed lock and
repository history remain the durable source identities.

Run the focused boundary tests with the repository's development environment:

```sh
uv run --no-sync pytest tests/test_release_boundary.py tests/test_release_workflow.py
```

The tests include a two-repository release-history fixture: a robot release
refreshes simulator test pins without creating the next simulator version, while
runtime, mixed and reverted-runtime changes still allow normal releases. This
fixture proves classification behavior, not a claim that a production Renovate
cycle has already been observed stopping.

## Released manual delivery

Verification retains its unversioned `documentation-preview`. Production Pages
is built in `deploy-docs` only after the draft finalizer succeeds. That job checks
out the returned tag and reads its release record, requiring a public, immutable,
non-prerelease release and exact equality between the tag, checkout and verified
workflow commit. Missing or mismatched release evidence fails closed.

The same manual builder receives the confirmed version through `DOCS_VERSION`;
it records the version and full source commit in provenance and renders them in
the manual. The installed-wheel help and source-bound CLI report remain part of
the build. The Pages payload is checked and uploaded under a version, commit and
attempt-specific name. Deployment retains the existing serialization and current
main check. Test-pin-only changes still verify and produce a preview without a
simulator release or stable Pages update.

If delivery fails after publication, rerun only the failed `deploy-docs` job of
that workflow run. It revalidates the same existing release and rebuilds from its
exact source; it does not call the writer or draft finalizer. Do not rerun all
jobs to retry delivery: immutable release assets cannot be replaced. A stale
revision continues to skip deployment under the existing current-main policy.
