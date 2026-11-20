# SPDX-License-Identifier: Apache-2.0
"""The development image carries a pinned mcap CLI, so the `mcap doctor` test runs in CI.

Without it that test skipped. The release is pinned by version and each architecture's sha256
(checked with the release's published digests), and the install verifies the hash before use.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
DEVELOPMENT = DOCKERFILE.split("AS development", 1)[1].split("AS runtime", 1)[0]

PINNED = {
    "amd64": "3ace13bc8f0273c69742f25695f7f2df84aed57ddd7c36d5f9a59561dc719fb1",
    "arm64": "94fd84e40ed1593205caf9b9ccee863181256a2fe1ad051ee55d31b4d1b4aca7",
}


def test_the_development_image_pins_the_mcap_cli_version() -> None:
    assert re.search(r"^ARG MCAP_CLI_VERSION=0\.4\.0$", DEVELOPMENT, re.M)
    assert "releases%2Fmcap-cli%2Fv${MCAP_CLI_VERSION}/mcap-linux-${arch}" in DEVELOPMENT


@pytest.mark.parametrize("arch", sorted(PINNED))
def test_each_architecture_has_its_sha256_pinned(arch) -> None:
    assert re.search(rf"^ARG MCAP_CLI_SHA256_{arch.upper()}={PINNED[arch]}$", DEVELOPMENT, re.M)


def test_the_download_is_verified_before_it_is_installed() -> None:
    install = DEVELOPMENT.split("MCAP_CLI_VERSION", 1)[1]
    assert install.index("sha256sum -c") < install.index("install ")


def test_the_doctor_test_cannot_skip_in_the_tooling_image() -> None:
    text = (ROOT / "tests" / "test_robot_topics_verification.py").read_text()
    assert 'os.environ.get("LIVINGPROOF_TOOLING")' in text
