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
    result = adapter_commit()
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
    print(json.dumps({"exit": code, "message": str(exc)}))
    sys.exit(code)
if isinstance(result, str):
    print(json.dumps({"commit": result, "provenance": None, "returned": "str"}))
else:
    commit, provenance = result
    print(json.dumps({"commit": commit, "provenance": provenance, "returned": "tuple"}))
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


def test_checkout_uses_its_own_git_commit_not_a_stale_snapshot():
    head = _git(ROOT, "rev-parse", "HEAD")
    path = Path(evidence.__file__).resolve().parent / "_build_provenance.json"
    original = path.read_text(encoding="utf-8") if path.is_file() else None
    try:
        path.write_text(json.dumps({"identity": {"commit": "ab" * 20}}), encoding="utf-8")
        assert evidence.adapter_commit() == (head, "git")
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original, encoding="utf-8")


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
    head = _git(ROOT, "rev-parse", "HEAD")
    assert run["simulator"]["adapter_commit"] == head
    assert run["simulator"]["provenance"] == "git"
    assert manifest["environment"]["simulator"]["adapter_commit"] == head
    assert manifest["environment"]["simulator"]["provenance"] == "git"


def test_installed_wheel_records_the_build_commit(tmp_path):
    """A wheel installed inside another git repo records the build commit.

    The same wheel outside any repository must not exit 1. Dropping
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
    assert got["provenance"] == "package"

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
        assert image_got["provenance"] == "package"

        for path in directs:
            path.unlink()
        deb_like = _run(outside_python, outside)
        assert deb_like.returncode == 0, deb_like.stderr
        deb_got = json.loads(deb_like.stdout)
        assert deb_got["commit"] == build_commit
        assert deb_got["provenance"] == "package"

        (site / "_build_provenance.json").unlink()
        unknown = _run(outside_python, outside)
        assert unknown.returncode == 0, unknown.stderr
        unknown_got = json.loads(unknown.stdout)
        assert unknown_got["commit"] is None
        assert unknown_got["provenance"] == "unknown"
    finally:
        shutil.rmtree(outside, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-ra"]))
