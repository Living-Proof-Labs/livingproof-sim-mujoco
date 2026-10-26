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
