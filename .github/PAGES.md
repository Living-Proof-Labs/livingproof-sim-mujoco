# Pages deployment preparation

Issue #3's scoped approval permits this repository's deployment job to use
`pages: write` and `id-token: write`. Only `deploy-docs` receives these grants.
Build and pull-request execution use `contents: read` only. The existing
`checks: write` grant moves to a separate result-publication job, which reads
same-run reports and runs no repository code. The deploy job executes no repository code.

The parent settings worker must enable Pages with the Actions build source and
restrict the `github-pages` environment to `main`. The expected URL is
https://docs.livingprooflabs.com/livingproof-sim-mujoco/ through the existing
organization domain. Do not set a repository custom domain or commit CNAME.
No domain, DNS, mail, organization, or unrelated security change is needed.

`verify` builds and tests the site before uploading the same-run Pages artifact.
Only a main push can deploy. The deployment job serializes on `mujoco-pages` and
checks current main inside that lock, skipping stale revisions. PRs only produce
14-day preview artifacts. An unsuccessful build or test blocks deployment.

After independent review and merge, record the successful deployment run and
source revision. Open the public site and exercise desktop/mobile navigation,
search result navigation, theme switching, command reference and build context.
Check assets and fragments at the project subpath. Then replace the README's
preview notice with the prominent verified live-manual link and ask the parent
to add the verified URL to the shared documentation directory. Until then,
settings, deployment and live acceptance must not be described as complete.
