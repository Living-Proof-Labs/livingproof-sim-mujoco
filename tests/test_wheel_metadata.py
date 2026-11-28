# SPDX-License-Identifier: Apache-2.0
"""The distributed media extra must enforce the patched Pillow floor."""

import email
import os
from pathlib import Path
import subprocess
import zipfile

from packaging.requirements import Requirement
import pytest


@pytest.fixture(scope="module")
def metadata(tmp_path_factory):
    wheel = os.environ.get("TEST_WHEEL")
    if not wheel:
        out = tmp_path_factory.mktemp("wheel")
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], check=True)
        wheel = next(out.glob("*.whl"))
    with zipfile.ZipFile(Path(wheel)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(archive.read(name))


def test_media_requires_patched_pillow_in_built_wheel(metadata):
    requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist")]
    pillow = [r for r in requirements if r.name.lower() == "pillow"]
    assert len(pillow) == 1
    requirement = pillow[0]
    assert str(requirement.specifier) == ">=13.0.0"
    assert "13.0.0" in requirement.specifier
    assert "11.3.0" not in requirement.specifier
    assert "12.3.0" not in requirement.specifier
    for platform in ("linux", "darwin", "win32"):
        for python in ("3.11", "3.12", "3.13", "3.14"):
            environment = {"sys_platform": platform, "python_version": python}
            assert requirement.marker.evaluate({**environment, "extra": "media"})
            for extra in ("", "torch"):
                assert not requirement.marker.evaluate({**environment, "extra": extra})


def test_media_stays_optional_and_python_support_is_unchanged(metadata):
    assert metadata["Requires-Python"] == ">=3.11"
    requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist")]
    media = {"pillow", "imageio", "imageio-ffmpeg", "mcap"}
    assert media <= {r.name.lower() for r in requirements}
    for requirement in requirements:
        if requirement.name.lower() in media:
            assert requirement.marker is not None
            assert requirement.marker.evaluate({"extra": "media"})
            assert not requirement.marker.evaluate({"extra": ""})


def test_built_wheel_requires_a_numpy_range_robot_repos_can_resolve(metadata):
    requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist")]
    numpy = [r for r in requirements if r.name.lower() == "numpy" and r.marker is None]
    assert len(numpy) == 1
    specifier = numpy[0].specifier
    assert not any(item.operator in ("==", "===") for item in specifier), str(specifier)
    for version in ("2.4.6", "2.5.4", "2.5.5"):
        assert specifier.contains(version), version
    assert not specifier.contains("3.0.0")
