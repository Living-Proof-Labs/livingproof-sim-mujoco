"""Installed-wheel adapter commit: the build snapshot, not the enclosing repository."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from livingproof_sim_mujoco import evidence

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json, sys
from livingproof_sim_mujoco.evidence import adapter_commit
try:
    commit = adapter_commit()
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
    print(json.dumps({"exit": code, "message": str(exc)}))
    sys.exit(code)
print(json.dumps({"commit": commit, "exit": 0}))
"""


def _run(python: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(
        [str(python), "-c", PROBE],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def _init_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "other",
        ],
        check=True,
    )
    return _git(path, "rev-parse", "HEAD")


def _wheel_commit(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith("_build_provenance.json"))
        saved = json.loads(archive.read(name))
    commit = saved["identity"]["commit"]
    assert isinstance(commit, str) and len(commit) == 40
    return commit


def _install(python: Path, wheel: Path) -> None:
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
    )


def _site(python: Path) -> Path:
    found = subprocess.check_output(
        [
            str(python),
            "-c",
            "import livingproof_sim_mujoco, pathlib; "
            "print(pathlib.Path(livingproof_sim_mujoco.__file__).parent)",
        ],
        text=True,
    ).strip()
    return Path(found)


def _direct_urls(site: Path) -> list[Path]:
    return list(site.parent.glob("*.dist-info/direct_url.json"))


def _run_manifest_validator():
    """Published livingproof-run-manifest/0.1 schema. The specs pin is contract/pins.json."""
    import urllib.request

    from referencing import Registry, Resource
    from jsonschema import Draft202012Validator

    pins = json.loads((ROOT / "src/livingproof_sim_mujoco/contract/pins.json").read_text())
    base = (
        "https://raw.githubusercontent.com/Living-Proof-Labs/robot-capability-specs/"
        f"{pins['commit']}/specs/robots/"
    )
    run = json.loads(urllib.request.urlopen(base + "run-manifest.schema.json", timeout=30).read())
    profile = json.loads(
        urllib.request.urlopen(base + "robot-profile.schema.json", timeout=30).read()
    )
    profile_uri = (
        "https://github.com/Living-Proof-Labs/robot-capability-specs/"
        "specs/robots/robot-profile.schema.json"
    )
    registry = Registry().with_resource(profile_uri, Resource.from_contents(profile))
    return Draft202012Validator(run, registry=registry)


def _commit(repo: Path, message: str, *, allow_empty: bool = False) -> str:
    command = [
        "git",
        "-C",
        str(repo),
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        message,
    ]
    if allow_empty:
        command.append("--allow-empty")
    subprocess.run(command, check=True)
    return _git(repo, "rev-parse", "HEAD")


def test_editable_install_sees_a_later_commit_without_reinstall(tmp_path):
    """An editable install keeps using git HEAD after a commit, not the install snapshot."""
    if shutil.which("uv") is None:
        pytest.skip("uv is required to editable-install the package")
    project = tmp_path / "project"
    shutil.copytree(
        ROOT / "src",
        project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "_build_provenance.json", "*.pyc"),
    )
    for name in ("pyproject.toml", "hatch_build.py", "README.md", ".gitignore"):
        shutil.copy(ROOT / name, project / name)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "-C", str(project), "add", "."], check=True)
    installed = _commit(project, "install")
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "-e", str(project)],
        check=True,
    )
    loaded = subprocess.check_output(
        [
            str(python),
            "-c",
            "import livingproof_sim_mujoco, pathlib; "
            "print(pathlib.Path(livingproof_sim_mujoco.__file__).resolve())",
        ],
        text=True,
    ).strip()
    assert loaded.startswith(str((project / "src").resolve()))
    snapshot = project / "src" / "livingproof_sim_mujoco" / "_build_provenance.json"
    if not snapshot.is_file():
        snapshot.write_text(json.dumps({"identity": {"commit": installed}}), encoding="utf-8")
    stale = json.loads(snapshot.read_text(encoding="utf-8"))["identity"]["commit"]
    assert stale == installed
    live = _commit(project, "later", allow_empty=True)
    assert live != stale
    assert json.loads(snapshot.read_text(encoding="utf-8"))["identity"]["commit"] == stale
    seen = _run(python, project)
    assert seen.returncode == 0, seen.stderr
    assert json.loads(seen.stdout)["commit"] == live


def test_run_manifest_records_checkout_provenance(tmp_path, monkeypatch):
    from test_evidence import Toy, _repo

    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "out", trials=1)
    run = json.loads((tmp_path / "out" / "run-manifest.json").read_text())
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert set(run["simulator"]) == {"engine", "version", "adapter", "adapter_commit"}
    assert "provenance" not in run["simulator"]
    assert "provenance" not in manifest["environment"]["simulator"]
    assert (
        run["simulator"]["adapter_commit"] == manifest["environment"]["simulator"]["adapter_commit"]
    )
    assert len(run["simulator"]["adapter_commit"]) == 40
    _run_manifest_validator().validate(run)


def test_installed_wheel_records_the_build_commit(tmp_path):
    """A wheel installed inside another git repo records the build commit.

    The same wheel outside any repository still returns that commit. Dropping
    ``direct_url.json`` is the Debian payload; a file-install ``direct_url.json``
    without ``vcs_info.commit_id`` is the runtime image. Neither may replace the
    packaged ``identity.commit``.
    """
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build and install the wheel")
    out = tmp_path / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=ROOT, check=True)
    wheel = next(out.glob("*.whl"))
    build_commit = _wheel_commit(wheel)

    foreign = tmp_path / "foreign-repo"
    foreign_head = _init_repo(foreign)
    assert foreign_head != build_commit
    foreign_venv = foreign / ".venv"
    subprocess.run(["uv", "venv", str(foreign_venv)], check=True)
    foreign_python = foreign_venv / "bin" / "python"
    _install(foreign_python, wheel)
    inside = _run(foreign_python, foreign)
    assert inside.returncode == 0, inside.stderr
    got = json.loads(inside.stdout)
    assert got["commit"] == build_commit, got
    assert got["commit"] != foreign_head
    assert "provenance" not in got

    outside = Path("/tmp") / f"sim-mujoco-outside-{os.getpid()}"
    here = outside.resolve()
    while here != here.parent:
        assert not (here / ".git").exists(), here
        here = here.parent
    outside.mkdir()
    try:
        outside_venv = outside / "venv"
        subprocess.run(["uv", "venv", str(outside_venv)], check=True)
        outside_python = outside_venv / "bin" / "python"
        _install(outside_python, wheel)
        site = _site(outside_python)
        directs = _direct_urls(site)
        assert directs, "a wheel install writes PEP 610 direct_url.json"
        recorded = json.loads(directs[0].read_text(encoding="utf-8"))
        assert not (recorded.get("vcs_info") or {}).get("commit_id")

        image_like = _run(outside_python, outside)
        assert image_like.returncode == 0, image_like.stderr
        image_got = json.loads(image_like.stdout)
        assert image_got["commit"] == build_commit
        assert "provenance" not in image_got

        for path in directs:
            path.unlink()
        deb_like = _run(outside_python, outside)
        assert deb_like.returncode == 0, deb_like.stderr
        deb_got = json.loads(deb_like.stdout)
        assert deb_got["commit"] == build_commit
        assert "provenance" not in deb_got
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_unknown_commit_exits_without_a_manifest(tmp_path):
    """No package snapshot and no direct_url, outside any repository: exit 1, no manifest."""
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build and install the wheel")
    out = tmp_path / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=ROOT, check=True)
    wheel = next(out.glob("*.whl"))
    outside = Path("/tmp") / f"sim-mujoco-unknown-{os.getpid()}"
    here = outside.resolve()
    while here != here.parent:
        assert not (here / ".git").exists(), here
        here = here.parent
    outside.mkdir()
    try:
        venv = outside / "venv"
        subprocess.run(["uv", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
        _install(python, wheel)
        site = _site(python)
        for path in _direct_urls(site):
            path.unlink()
        (site / "_build_provenance.json").unlink()
        destination = outside / "evidence-out"
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        proc = subprocess.run(
            [
                str(python),
                "-m",
                "livingproof_sim_mujoco.evidence",
                "missing:ROBOT",
                "stock",
                str(destination),
            ],
            cwd=outside,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0, proc.stdout
        message = proc.stderr + proc.stdout
        assert "package provenance (_build_provenance.json)" in message
        assert "direct_url.json vcs_info" in message
        assert "no source tree containing this package" in message
        assert not destination.exists()
        assert not (destination / "manifest.json").exists()
        assert not (destination / "run-manifest.json").exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-ra"]))
