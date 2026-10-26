# SPDX-License-Identifier: Apache-2.0
"""Build offline products using hash-checked wheels from the committed lock."""

import csv
import gzip
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    if sys.version_info[:2] != (3, 11) or platform.machine() != "x86_64":
        raise SystemExit(
            "Debian packaging is verified only with Python 3.11 on Linux amd64. Use task package in Docker."
        )
    out = Path("dist")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    # Hatch VCS derives metadata; no manually asserted release version.
    run("uv", "build", "--out-dir", str(out), env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"})
    wheel = next(out.glob("*.whl"))
    package_version = wheel.name.split("-")[1]
    run(
        "uv",
        "export",
        "--locked",
        "--no-dev",
        "--no-emit-project",
        "--format",
        "requirements-txt",
        "--output-file",
        "dist/requirements.txt",
    )
    wheelhouse = out / "wheelhouse"
    wheelhouse.mkdir()
    run(
        sys.executable,
        "-m",
        "pip",
        "download",
        "--require-hashes",
        "--only-binary=:all:",
        "-r",
        "dist/requirements.txt",
        "-d",
        str(wheelhouse),
    )
    test_wheels = out / "test-wheelhouse"
    test_wheels.mkdir()
    run(
        "uv",
        "export",
        "--locked",
        "--only-group",
        "dev",
        "--format",
        "requirements-txt",
        "--output-file",
        "dist/test-requirements.txt",
    )
    run(
        sys.executable,
        "-m",
        "pip",
        "download",
        "--require-hashes",
        "--only-binary=:all:",
        "-r",
        "dist/test-requirements.txt",
        "-d",
        str(test_wheels),
    )
    shutil.copy2(wheel, wheelhouse)
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        stage.chmod(0o755)
        library = stage / "usr/lib/livingproof-sim-mujoco"
        run(
            "uv",
            "pip",
            "install",
            "--no-compile-bytecode",
            "--no-index",
            "--no-deps",
            "--target",
            str(library),
            *map(str, sorted(wheelhouse.glob("*.whl"))),
        )
        # Console script paths from target installs are not relocatable.
        shutil.rmtree(library / "bin", ignore_errors=True)
        # Installer cache timestamps and local wheel paths are not runtime provenance.
        # Keep the common library's recorded source identity and all license notices.
        for metadata in library.glob("*.dist-info"):
            for name in ("uv_cache.json", "direct_url.json"):
                (metadata / name).unlink(missing_ok=True)
            record = metadata / "RECORD"
            with record.open(newline="") as source:
                rows = [
                    row
                    for row in csv.reader(source)
                    if not row[0].startswith("../") and (library / row[0]).is_file()
                ]
            with record.open("w", newline="") as target:
                csv.writer(target).writerows(rows)
        binary = stage / "usr/bin/livingproof-sim-mujoco"
        binary.parent.mkdir(parents=True)
        binary.write_text(
            "#!/usr/bin/python3\nimport sys\nsys.path.insert(0, '/usr/lib/livingproof-sim-mujoco')\nfrom livingproof_sim_mujoco.cli import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
        )
        binary.chmod(0o755)
        control = stage / "DEBIAN"
        control.mkdir()
        deb_version = package_version.replace(".dev", "~dev")
        (control / "control").write_text(
            f"Package: livingproof-sim-mujoco\nVersion: {deb_version}\nArchitecture: amd64\nDepends: python3 (>= 3.11), python3 (<< 3.12), libc6 (>= 2.36), libstdc++6, libgcc-s1\nMaintainer: Living Proof Labs <rivet@rollerbot.ai>\nDescription: Native MuJoCo headless physics command with bundled locked engine\n"
        )
        notice = stage / "usr/share/doc/livingproof-sim-mujoco/copyright"
        notice.parent.mkdir(parents=True)
        notice.write_text(
            Path("LICENSE").read_text()
            + "\nBundled dependencies retain their licenses in /usr/lib/livingproof-sim-mujoco/*.dist-info/.\n"
        )
        # Fix archive timestamps; retain dependency license files untouched.
        epoch = 315532800
        for path in stage.rglob("*"):
            os.utime(path, (epoch, epoch), follow_symlinks=False)
        os.utime(stage, (epoch, epoch))
        run(
            "dpkg-deb",
            "--root-owner-group",
            "--build",
            str(stage),
            str(out / f"livingproof-sim-mujoco_{deb_version}_amd64.deb"),
            env={**os.environ, "SOURCE_DATE_EPOCH": str(epoch)},
        )
    with (out / "wheelhouse.tar.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=315532800) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(wheelhouse.iterdir()):
                    info = archive.gettarinfo(str(path), arcname="wheelhouse/" + path.name)
                    info.mtime = 315532800
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    (out / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted(out.iterdir())
            if p.is_file() and not p.name.startswith(".")
        )
    )


if __name__ == "__main__":
    main()
