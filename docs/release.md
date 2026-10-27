# Distribution and release

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

## Registry migration gate

[Shared distribution issue 29](https://github.com/Living-Proof-Labs/collaboration-guidelines/issues/29)
prefers public registry delivery for future routine releases. MuJoCo retains its
archive until the same version and Linux amd64 image has an independently verified
anonymous digest-pinned pull and physics run. The current workflow has no registry
publisher; this documentation change grants no publishing permissions. Existing
assets and non-image packages remain unchanged. A registry manifest digest must
not be substituted with the local Docker image ID.
