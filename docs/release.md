# Distribution and release

[Package release v1.0.0](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/releases/tag/v1.0.0)
is the published release verified on 2026-10-27; v1.0.1 is still a draft.
This development manual and its tested candidate metadata are separate from
that package release. Historical manuals are not maintained; consult the
[source release tags](https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/tags).

Python wheels, a Debian package for Debian 12 amd64, and a versioned Docker
runtime image are the initial products. Wheels are pure Python wrappers but
require native MuJoCo and NumPy wheels. ARM, Windows and macOS are unverified.
The Debian package bundles locked Python dependencies, preserving their license
metadata, and depends on Debian's Python 3.11 and native libraries.

`task package` builds a wheel, source archive, Debian package, and an offline
wheelhouse. `task smoke` installs the products and exercises real physics.
Install offline with `pip install --no-index --no-deps wheelhouse/*.whl` in a
Python 3.11 Linux amd64 environment. Install Debian with
`sudo apt install ./livingproof-sim-mujoco_*_amd64.deb`. Do not use sudo for Task.

The workflow reuses collaboration-guidelines v4.3.0 semantic-release to create
a draft after main verification. It assembles and tests products from that tag
and attaches all assets before publication. Runtime distribution is a Docker
image archive (`docker load -i runtime-amd64.tar.gz`) attached to the release;
there is no registry grant or new public package setting. An authorized release
operator must inspect the draft, tests and checksums before publishing it. No
release or image is published by this implementation branch.

Release gates: independent PR review; hosted passing and deliberately failing
JUnit case publication; public unauthenticated downloads of all assets; two
identical clean package builds; dependency license audit; and all product tests.
Development tooling and Debian system dependencies follow the Debian repository
at build time. The headless runtime adds no system packages to its digest-pinned
base; its Python dependencies come from the locked wheelhouse. Image build
timestamps still prevent a claim of bit-for-bit image reproducibility without a
separate uncached comparison. Capture final image digest and inventories for release.
