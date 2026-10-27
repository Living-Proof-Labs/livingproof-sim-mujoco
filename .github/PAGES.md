# Pages deployment

Issue #3's scoped approval permits this repository's deployment job to use
`pages: write` and `id-token: write`. Only `deploy-docs` receives these grants.
Build and pull-request execution use `contents: read` only. The existing
`checks: write` grant moves to a separate result-publication job, which reads
same-run reports and runs no repository code. The deploy job executes no repository code.

Use the Actions build source and restrict the `github-pages` environment to
`main`. The public URL is
https://docs.livingprooflabs.com/livingproof-sim-mujoco/ through the existing
organization domain. Do not set a repository custom domain or commit CNAME.
No domain, DNS, mail, organization, or unrelated security change is needed.

`verify` builds and tests the site before uploading the same-run Pages artifact.
Only a main push can deploy. The deployment job serializes on `mujoco-pages` and
checks current main inside that lock, skipping stale revisions. PRs only produce
14-day preview artifacts. An unsuccessful build or test blocks deployment.

The first documentation deployment followed independent review and merge at
`5ecc8d516bf547e7d267ce4e5a3917a6bffce1ea` in
[main run 40453277204](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/actions/runs/40453277204).
Live acceptance passed on 2026-10-27: normal TLS, styles, navigation, anchors,
native CLI reference, the bounded scenario inventory, and exact clean source/run
provenance. The `timeout` search returned three results and navigated to the
physics fragment through reload; dark-theme persistence and system theme passed.

For later deployments, verify the actual public source revision and exercise
desktop/mobile navigation, search results, themes, command reference, assets and
fragments at the project subpath. A successful upload alone is not live acceptance.
The shared documentation directory is maintained separately.
