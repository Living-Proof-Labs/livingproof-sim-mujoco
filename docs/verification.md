# Development and review

Docker and Task 3.54.0 are host prerequisites. Run `task setup`, `task verify`,
`task package`, and `task smoke` for the existing product checks. The devcontainer
uses the same development image. Its `LIVINGPROOF_TOOLING=1` marker makes
ordinary `task setup`, `task check`, `task test`, and `task verify` run directly.
Explicit `task -t Taskfile.runtime.yml` commands also remain available. Bare
`task` lists commands without an error. Image and smoke tasks still require a
Docker host; no Docker-in-Docker is needed for development tooling.

## Browser simulation playback

Inside the Linux amd64 devcontainer run `task preview`. In the browser editor's
Ports tab, add port 8000 if needed, confirm **Private** visibility, and use
**Open in Browser** while signed into the same GitHub account. Codespaces supplies
the forwarded HTTPS URL; localhost on your laptop is not the remote server.
[GitHub documents private forwarding](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).
No public port, GPU, desktop, or local ARM execution is required.

The development-only helper renders the unchanged built-in falling model with
MuJoCo's CPU OSMesa backend: 200 steps, 0.4 simulation seconds, 41 frames.
Play/Pause and a slider inspect recorded playback at one tenth speed; they do
not control live physics. The sphere has no floor. This is engine evidence,
not a robot capability claim. The installed CLI and its runtime dependencies
remain headless. The helper serves only its generated assets on loopback and
stops with Ctrl+C. Run `task preview` again to regenerate.

Existing containers need **Codespaces: Rebuild Container** for the tooling marker
and OSMesa library. Until then, `task -t Taskfile.runtime.yml setup` followed by
`task -t Taskfile.runtime.yml test` bypasses Docker. For a busy port use
`uv run --no-sync python tools/preview.py --port 8001` and privately forward 8001.

`tools/test-devcontainer.sh` exercises creation setup, bare task, setup, physics
tests, and rendered playback tests as a non-root user with no Docker executable.
`task test:preview` checks physics samples, changed rendered frames, HTTP asset
isolation, and Play/Pause/scrub/replay in Chromium. These local tests do not
establish a hosted Codespaces authentication or port-forwarding session.

## Manual checks

Run `task docs` to build the manual and `task test:docs` to check its HTML and
browser behavior. The browser task installs the Chromium revision pinned by
Playwright 1.64.1 in the disposable tooling container; its Linux libraries are in the development image.
The manual uses Zensical 0.0.73; exact tooling dependencies are in `uv.lock`.
`task docs:reference` updates the committed command reference after intentional CLI changes.

Download the documentation-preview artifact from the PR's Actions run (14-day
retention), extract it, and serve `html/` with `python -m http.server 8000`.
Open http://localhost:8000/. Browser search needs HTTP; opening a file directly
is not a supported preview path. `context.md` and `provenance.json` identify the
source, installed wheel digest, dependency versions, and CI run. Local uncommitted
builds are explicitly marked dirty. This is development documentation, not a release.

The generated feature inventory uses the Given/When/Then descriptions on actual
CLI tests and maps all parameterized cases to their JUnit outcomes. Missing cases
are untested, never passed. This bounded inventory covers physics, native model
loading, errors, deadlines, signals, help and identity. It does not cover every
MuJoCo model, renderer, platform, or robot capability. Line coverage is not measured.

Independent review should install the candidate wheel or container in a fresh
environment, run the first example and an invalid model, then explore shutdown.
Serve the downloaded manual, follow navigation and search results, toggle themes,
and inspect it at mobile width. Record exact artifact, revision, commands and
outcomes using the [released review procedure](https://github.com/Living-Proof-Labs/collaboration-guidelines/blob/v6.0.0/docs/independent-review.md).
An implementer's browser test is not independent review or live-site acceptance.

## Initial gaps and adoption

Issue #3 found generated argparse help and real physics/lifecycle tests, but no
rendered manual, scenario-to-result inventory, or browser preview tests. Existing
CI already publishes individual JUnit cases; this change reuses that publisher.
The local agent pin deliberately advances from v4.3.0 to v5.0.0, resolved as
`dd4b0391a9a66fa7944f6655aac57c67d3f8c002`. No local `.agents/skills` were present.
The release workflow stays on its existing v4.3.0 pin; no release upgrade is needed.

The [public manual](https://docs.livingprooflabs.com/livingproof-sim-mujoco/) was
published from revision `5ecc8d516bf547e7d267ce4e5a3917a6bffce1ea` by
[main run 40453277204](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/actions/runs/40453277204).
Live acceptance passed on 2026-10-27: normal TLS access, styles, navigation,
anchors, native CLI reference and the bounded 23 behavioral cases in 11 scenario
groups. The live context identified the exact clean revision and run above.
Searching for `timeout` returned three results; navigation to the physics
fragment and reload passed. Dark-theme persistence and system theme passed.
The shared documentation-directory entry is coordinated separately. Documentation
deployment does not publish the simulator release; v1.0.1 remains a draft.
See [Pages deployment](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/blob/main/.github/PAGES.md).

## Manual recipe adoption

Issue #5 adopts collaboration-guidelines v6.0.0, resolved to
`52a2f274576dab20e96ee1152b6c8d187cfbf377`, including its documentation recipe
and ownership guidance. Existing published page paths and fragments remain;
common navigation and generated Task reference are additive. Initial publication
cleanup from issue #3 is reconciled here, with no separate cleanup PR.
Task reference generation uses only the two reviewed local Taskfiles and native
Task 3.54.0 listing/summary commands. The release workflow remains pinned to v4.3.0.
No public library API or capability contract is introduced; those references are
inapplicable. Line coverage remains unmeasured. A private security route is an
explicit gap, not an invented destination.
