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

`python -m livingproof_sim_mujoco.evidence <module:adapter> <configuration> <out> --lock <lock.json> [--trials N]`
records observations and does not grade the robot. `--lock` is the caller's verification
lock (a `livingproof-verification-lock/0.2.0` or the `1.0.0` projection of one). It selects
the scenario and the world. Wall geometry and the named start place come from the world's
release vendored in this package: `release.json` is checked, the world is translated with
the same mapping as the hallway fixture, and those wall boxes and place sites are compiled
onto the robot model. The robot package supplies the robot model, the controller, the
nominal pose and the profiles. It does not supply the world. A world this package does not
vendor is UNSUPPORTED. If the world has walls and the composed model is missing any of
them, or the robot base is at the origin when the start place is not, the command exits 1
with `INCOMPATIBLE` and writes nothing.

The scenario parameter file comes only from `sources.parameters`, which names a
path and a sha256. A mismatch is an ERROR. A missing `sources.parameters` entry is
also an ERROR; the robot's `pins.json` and the vendored parameter file are not
used. The velocity command is that file's `command`. Only mode `constant` is accepted;
any other mode is an ERROR. On the straight open hallway the command is `[0.5, 0, 0]`.
A parameter file with no `command` states only the outcome. This runner does not invent
a velocity for it.

`manifest.json` records the file it loaded as `sources.parameters`, with `path` (the lock's
`sources.parameters` path) and `sha256`. That digest is of the one read that was checked
against the lock and parsed; the file is not read again. It is also recorded when the
scenario is UNSUPPORTED and no trial runs. The field is additive, so the evidence format
stays `livingproof-hallway-evidence/2.0.0`, and `run-manifest.json` does not change.

Before any trial, a scenario that lists the precondition `scenario_supported` is checked
against the robot repository's `manifest.json` (`livingproof-robot-manifest/0.1`).
That is robot-capability-specs v2.3.0, Scenario support. The optional `scenarios` list
names the scenarios the adapter declares. A list that does not name the requested
scenario, an empty list, or a manifest with no `scenarios` list records
`UNSUPPORTED: precondition scenario_supported unmet` and runs no trial. A missing
manifest file names nothing, the same as a manifest with no list. That decision is
made as soon as the lock and the parameter file are read, before the world fixture
or the route. The record is the evidence manifest: the precondition is unmet,
`trials.results` is empty, and no robot pass or fail is written. The requested
world is named from the lock and is not loaded. The planned `trials.count` stays at least 1 so the evidence
keeps the shape the capability verifier grades as `UNSUPPORTED`. A scenario that does
not list the precondition keeps running, including one whose parameter file still has
its command and whose manifest has no list. A declared scenario that has a command
runs the trials. Invalid manifest data is a test-system `ERROR` and writes nothing.

A declared scenario with no command, such as walk-corner 0.1.0 on hallway-corner 0.1.0,
is driven by the robot adapter. The adapter's optional
`drive(name, model, ctrl, policy, places)` returns `step(data)`, which advances one
physics timestep its own way. `places` holds the world's own `start`, `exit`, `route`
and `through` places, as the parameter file names them. The runner sends no velocity
and writes each trial as `livingproof-route-observation/0.1.0`:
- the base trace, with distance to the walls and how far the robot reaches past each
  wall's inner face;
- the first exit crossing and the first sample inside each `through` region;
- the largest distance from the route and the lowest base height;
- every wall contact.

The trial ends `crossed_exit`, `time_limit`, `non_finite_state`, or `left_world`
(through a wall, or across the exit line outside the exit segment). No threshold is
applied; the capability verifier grades. A declared adapter with no `drive` is a
test-system `ERROR` that writes nothing. Walk-corner uses the same
`sources.parameters` rule: a missing entry is an ERROR, not a vendored default.
The walk-hallway observation and its bytes are unchanged.

The `world` in `run-manifest.json` names the `world.json` that was loaded: its
`id` and `version` come from the lock, and its `sha256` is the digest of those bytes. A
lock `world.sha256` that disagrees is an ERROR before anything is written. A `world.tag`
or `world.commit` that is not a release of that same file is the same ERROR. The
published verification locks name livingproof-worlds v1.1.0; that release's
`world.json` is the vendored file byte for byte, so those locks are accepted.
When the lock declares `sources.world`, those bytes are the simulated world. A sha256
that does not match the file it names is an ERROR, and a mismatched vendored copy is
not substituted. The pins file is not used. The declared document is checked before
use: it must include `geometry`, `places`, and `dimensions`, and be mappable. A world
that fails that check is `INCOMPATIBLE` (or `UNSUPPORTED` for an unmappable feature)
and nothing is written. `world.tag` and `world.commit` are not compared with a
vendored release when `sources.world` is set; `world.sha256` pins the bytes. Locks
without `sources.world` still load the vendored release named by `lock.world`.

## Standing check

`python -m livingproof_sim_mujoco.walk stand <module:adapter> --configuration NAME --lock <lock.json> [--out DIR]`
is the zero-command standing check at the world's start place. `--lock` is required. It
is the same caller verification lock as `evidence`, and the world and the start place
come from `prepare_run`: `sources.world` when the lock declares it, otherwise the
vendored release that lock names. With `--out`, the check writes
`stand.json` and `stand-zero-command.mp4` in that directory. It does not grade the robot.
A lock missing `world`, `releases`, `configurations` or `plan` exits 1 and names the
missing field.

A world whose format is `livingproof-world/0.2` is an ERROR when that is the loaded
document. This release does not yet write a neutral observation for that format. Limits
in the frozen hallway record stay as they were recorded before: `lateral_limit` to 6
decimal places and `time_limit` unrounded.

- Exit 0: the observations were recorded. An unmet precondition is also exit 0: no trial runs, and the manifest says so.
- Exit 1: the lock, plan, manifest, or composed world is unusable (`INCOMPATIBLE` or `UNSUPPORTED` are this exit). An unknown adapter commit is one case: no manifest is written.
- Exit 2: the command line is unusable (argparse), including a missing `--lock`.
- Exit 4: `trials.count` is greater than 1 and the trials disagree. The manifest records that as a test-system reproducibility finding (`trials.reproducibility.finding`). It is not a robot grade. The physics command's exits (2, 124, 130, 143) and the hallway fixture tool's exit 1 (a non-PASS world result) are different programs and are not this code.
- Exit 5: a trial raised an exception. `manifest.json` is still written for the trials that ran, with the failed trial recorded in `trials.results[]` as `{"index", "exception": {"type", "message"}, "wall_clock_s"}` (a single-line message up to 240 characters) and no `outcome`, `path`, `sha256`, or `trajectory_sha256`, and later trials do not run. Because the exit is non-zero and not 4, `scenario.yml` stops before `verify`, and the verifier refuses to grade a manifest whose trial result omits `path` and `sha256`.

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

## Declared hashes

Every sha256 in `manifest.json` and `run-manifest.json` resolves, and
`evidence.DECLARED_HASHES` lists each field with what it covers:

- `trials.results[].sha256` is the sha256 of the file at that result's `path`
  (`trials/trial-NN.json`) in the evidence folder.
- `trials.results[].trajectory_sha256` names no file. It is the sha256 of the trial
  observation's `trace` as compact, key-sorted UTF-8 JSON:
  `json.dumps(obs["trace"], separators=(",", ":"), sort_keys=True)`. Recompute it from the
  published trial with `evidence.trajectory_sha256_of("trials/trial-NN.json")`. Identical
  trials share it; the reproducibility check compares it.
- `lock.sha256`, `world.sha256`, `lock.content.world.sha256`,
  `lock.content.configurations[].sha256`, `sources.parameters.sha256`,
  `configuration.sha256` and `robot.profile_sha256` are pinned inputs: the lock file, the
  loaded `world.json`, the loaded scenario parameter file, the robot's configuration file
  and its `robot.json`, at the commits the manifest records. They are not files in the
  evidence artifact.
- Every sha256 under `robot.profile.` or `configuration.profile.` (in either manifest) and
  under `lock.content.sources.` (in `manifest.json`) is a pinned source too, whatever its
  name: the embedded profiles cite pinned files in the robot repository or upstream, such
  as `robot.profile.model_source.sha256`, `configuration.profile.controller.artifact.sha256`
  and each robot's own `configuration.profile.settings.<file>.sha256`, and a lock's
  `sources` pin scenario inputs. `evidence.DECLARED_SOURCE_PREFIXES` lists the prefixes.

`evidence.unresolved_hashes(folder)` returns one message per declared hash that does not
resolve (an artifact file that differs, a trajectory that does not recompute, or a field
`DECLARED_HASHES` and `DECLARED_SOURCE_PREFIXES` do not document). It is empty for a run
this module wrote, including real G1, T1 and Go1 runs (`tests/fixtures/real-runs`).

## Evidence media

`python -m livingproof_sim_mujoco.media <module:adapter> <configuration> <evidence dir>
<verification.json> <out>` (install the `media` extra) re-runs every trial of an evidence
run, in sequence, with a read-only recorder. On Linux, importing this module sets
`MUJOCO_GL` to `osmesa` when `DISPLAY` is unset or empty and `MUJOCO_GL` is not already
set, so a headless host uses the OSMesa software renderer. An existing `MUJOCO_GL` is
not replaced. Other platforms, and a Linux session with `DISPLAY` set, keep MuJoCo's
own choice. It writes an MCAP file per trial (`/base_pose` as
`foxglove.PoseInFrame`, `/joint_states` at 50 Hz, and `/wall_contacts` with one message per
`body_contact` event with a hallway wall, `{wall, start, end, depth}`, where `depth` is the
intrusion past the wall's inner face, not a penetration depth; JSON, zstd) with every message in `log_time` order (`mcap doctor` is clean) and an mp4
screencast of the verifier's first PASS trial and first FAIL trial only. Next to each MCAP it
writes the per-trial `livingproof-observation-binding/1.0.0` file, `trial-NN.binding.json`: the
producer binding shared by every robot, merged with the robot's base body and nominal base
height (a term the robot cannot supply is `unbound`, never guessed). The nominal's `source` is
the configuration's own (`measured`, `published`, `chosen`), carried as is, and a nominal with no
source in the configuration is not offered; its `ref` is the configuration file at the robot's
pinned commit, `<repo>@<commit>:<path>`, never a branch. The repo part may be a full URL (an `https://` URL, or an scp-style `git@host:path`), so a ref splits at the last `@`; any `user:password@` userinfo in a `scheme://` repo URL is removed first, in the ref and in the run manifest's `robot.repo`, so a credentialed remote never lands in a file. The one exception is the lock itself: `lock.content` keeps the lock's `robot.repo` verbatim, because the lock is recorded exactly as given, so a credential written in the lock stays in the manifest. Do not put one there. `media.json` is
`livingproof-hallway-media/2.1.0`: 2.0.0 plus an additive `binding` file entry on each trial
(a minor bump; readers of 2.0.0 fields are unaffected). It lists each
file's size and sha256. Each re-run observation must hash to that trial's evidence
sha256, or nothing is kept (recording goes to a staging folder, so a mismatch leaves no
partial files from any trial), so every recording is of the trial that was graded. It
fetches and verifies the pinned upstream files itself, before the adapter loads anything,
and re-running into an existing folder replaces only its `mcap/`, `video/` and
`media.json`. Media never grades and never changes evidence. An adapter may set the
optional `camera = (distance, elevation, azimuth)` for the tracking camera of media videos
and the `walk stand` clip (Go1 uses its own; default `walk.CAMERA`). The verifier's reusable walk-hallway
workflow uploads these as named Actions artifacts and hands their URLs to the reporter.
