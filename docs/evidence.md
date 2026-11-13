# Verification record

Guidance: collaboration-guidelines v4.3.0, resolved commit
`583ada2c1cd7cfc9832678316c401b801e8a9f6f`. Python and container Task templates,
licensing, developer-interface, tool-usability, release and GitHub settings
procedures were read at that tag. No relevant local `.agents/skills` files were
present in the selected workspace.

Shared identity: livingproof-common v1.0.0, resolved commit
`64d45dca16c640df56cfe5e0a5a2a4ad8afedf57`. GitHub's unauthenticated latest-release
API reported v1.0.0 on 2026-10-26. The public released wheel downloaded without
authentication and matched its release SHA256SUMS:
`8b7795748b54eeb89610f3716245f45db48ac0474465a5abbf974e7075c1476c`.

Linux x86_64 / Python 3.12.14: 17 source tests passed and 17 independently
installed-wheel tests passed outside the checkout, with real MuJoCo 3.17.0.
Docker development / Python 3.11.14: Task checks and all 17 tests passed.
The container used a hash-verified prefilled uv cache because online uv setup
encountered intermittent proxy DNS errors. No DNS, proxy service, firewall or
organization settings were changed. Hosted devcontainer setup is not inferred.
Two consecutive Python wheel/source archive builds with SOURCE_DATE_EPOCH=315532800
were byte-identical; raw hashes are in artifacts/python-reproducibility.json.
The explicit reporting failure fixture exited 1 and retained one named JUnit
failure. This establishes local failure-report generation, not hosted rendering.
Raw reports are written to artifacts; CI retains reports for 30 days.

The public repository was initialized with Apache-2.0 after explicit owner
approval. Implementation remains on a draft pull request for independent review.
No organization settings, package visibility or credentials were changed.

Final local product checks at source commit
`31bf8e1a0ff6dd2bc078cc32774cc2d0c10b22bf`:

- Development container, Python 3.11.14: checks and 17 tests passed.
- Installed runtime image, Python 3.11.14: 17 tests passed without network.
- Installed Debian package, Debian 12 / Python 3.11.2: 17 tests passed without network.
- Docker stop reached the command as PID 1, returned 143, and the worker exited
  cooperatively with code 0 after advancing physics.
- Two consecutive builds produced identical wheel, source archive, Debian,
  wheelhouse archive and requirements checksums. Installer cache timestamps and
  local installation paths are omitted from the Debian payload; source identity
  and third-party licenses remain present.

Raw results: artifacts/tests.xml, installed.xml, debian-tests.xml,
verification.json, package-reproducibility.json and logs/. These are local
review evidence, not hosted check publication or a released product.

Hosted verification subsequently passed at
[f8dac0e](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/actions/runs/40408178025):
[51 published results](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/runs/121872110784),
17 per development, installed runtime and installed Debian environment. The
[separate intentional failure](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/actions/runs/40408236991)
retained its nonzero workflow result and
[rendered named failure](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/runs/121871766878).
Both public check pages were inspected for individual cases and diagnostics.
Artifacts have 30-day retention. See [PR #2](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/pull/2)
for the latest head and runs.

Independent review, actual release assembly/publication and unauthenticated
product downloads remain pending. ARM, macOS, Windows and hosted devcontainer
execution remain unverified. Engine startup and software tests are not robot
capability evidence. No substantive merge or release has occurred.

## Evidence command

`python -m livingproof_sim_mujoco.evidence <module:adapter> <configuration> <out> [--trials N]`
records observations and does not grade the robot. The manifest format is
`livingproof-hallway-evidence/2.0.0`. Its `plan` records `trials.count` and does not
copy `seeds` or `required_passes` from the lock.

The world file and the scenario parameter file come from the robot's
`verification.lock.json` when `sources.world` or `sources.parameters` names a
path and a sha256. A mismatch is an ERROR and the pins file is not used. With
no lock entry, the file is taken from the robot's `pins.json`. That fallback
remains until robot repositories stop pinning those files (robot-capability-specs
issue 40, step 7). If the pins do not list the parameter file either, the
runner uses its copy of robot-capability-specs v2.2.0 `parameters.json`, checked
by sha256. The velocity command is that file's `command`. Only mode `constant`
is accepted; any other mode is an ERROR. On the straight open hallway the
command is `[0.5, 0, 0]`.

The `world` in `run-manifest.json` and `manifest.json` names the world file
that was loaded. When the file came from `sources.world` or `pins.json`, its
`sha256` is the digest of those bytes, and its `id` and `version` come from the
lock `world` entry or, when the lock has none, from the file's `id` and
`content_version`. A lock `world.id`, `world.version` or `world.sha256` that
disagrees with the loaded file is an ERROR before anything is written. Only a
world supplied by the robot adapter, with no file read by the runner, still
takes its `sha256` from lock `world.sha256` or the robot's `pins.json`.

A world whose format is `livingproof-world/0.2` is an ERROR. This release does
not yet write a neutral observation for that format, and it does not write the
frozen hallway observation for it. Limits in the frozen hallway record stay as
they were recorded before: `lateral_limit` to 6 decimal places and `time_limit`
unrounded. The parameter file's own rounding is what `resolve` returns and what
a later neutral record will store.

- Exit 0: the observations were recorded. An unmet precondition is also exit 0: no trial runs, and the manifest says so.
- Exit 1: the lock, plan, or manifest is unusable. Those are `SystemExit` messages. One case is an unknown adapter commit: no manifest is written.
- Exit 2: the command line is unusable (argparse).
- Exit 4: `trials.count` is greater than 1 and the trials disagree. The manifest records that as a test-system reproducibility finding (`trials.reproducibility.finding`). It is not a robot grade. The physics command's exits (2, 124, 130, 143) and the hallway fixture tool's exit 1 (a non-PASS world result) are different programs and are not this code.

`simulator.adapter_commit` in `run-manifest.json` is the 40-character commit of this
adapter. The field is required by `livingproof-run-manifest/0.1`; the object has no
provenance field. When this module is the package's own source tree (a checkout or an
editable install) and git HEAD resolves, that live HEAD is used, so a later commit is
seen without reinstalling. The toplevel must contain this package's `pyproject.toml` or
`src/livingproof_sim_mujoco` source. If that git lookup fails, or this is an installed
copy, the commit is read from the packaged `_build_provenance.json` `identity.commit`,
then from the installed distribution's PEP 610 `direct_url.json` `vcs_info.commit_id`.
An enclosing repository is not used. When none of those yields a 40-character commit,
the command exits 1 and names each source and why it failed. The Debian package and the
runtime image call this same function. The Debian payload drops `direct_url.json`, so
that install uses the build snapshot. The runtime image pip-installs the same wheel and
has no separate commit lookup.

The generated command reference in `docs/command.md` is the physics command `livingproof-sim-mujoco`. It does not list this module, so those exits stay as they are.

## Evidence media

`python -m livingproof_sim_mujoco.media <module:adapter> <configuration> <evidence dir>
<verification.json> <out>` (install the `media` extra) re-runs every trial of an evidence
run, in sequence, with a read-only recorder. It writes an MCAP file per trial (`/base_pose` as
`foxglove.PoseInFrame` and `/joint_states`, JSON, 50 Hz, zstd) and an mp4 screencast of
the verifier's first PASS trial and first FAIL trial only, plus `media.json` with each
file's size and sha256. Each re-run observation must hash to that trial's evidence
sha256, or nothing is kept (recording goes to a staging folder, so a mismatch leaves no
partial files from any trial), so every recording is of the trial that was graded. It
fetches and verifies the pinned upstream files itself, before the adapter loads anything,
and re-running into an existing folder replaces only its `mcap/`, `video/` and
`media.json`. Media never grades and never changes evidence. An adapter may set the
optional `camera = (distance, elevation, azimuth)` for the tracking camera of media videos
and the `walk stand` clip (Go1 uses its own; default `walk.CAMERA`). The verifier's reusable walk-hallway
workflow uploads these as named Actions artifacts and hands their URLs to the reporter.
