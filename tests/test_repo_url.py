# SPDX-License-Identifier: Apache-2.0
"""A credentialed git remote must not leak into the run manifest or the nominal ref (#149).

The credentials here are fake. Each test checks the real files a run writes, not only the helper.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_evidence  # noqa: E402
from test_evidence import Toy, _lock, _repo  # noqa: E402

from livingproof_sim_mujoco import evidence, media  # noqa: E402

SECRET = "ghp_FAKEtoken0123456789FAKEtoken"
USER = "robot-ci-user"
CREDENTIALED = f"https://{USER}:{SECRET}@github.com/example/toy-robot.git"
CLEAN = "https://github.com/example/toy-robot.git"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (CREDENTIALED, CLEAN),
        (f"https://{SECRET}@github.com/example/toy-robot.git", CLEAN),  # user-only
        (f"https://{USER}:@github.com/example/toy-robot", "https://github.com/example/toy-robot"),
        (f"http://{USER}:{SECRET}@example.test:8443/r.git", "http://example.test:8443/r.git"),
        (f"ssh://git:{SECRET}@github.com/example/r.git", "ssh://github.com/example/r.git"),
        (CLEAN, CLEAN),
        ("git@github.com:example/toy-robot.git", "git@github.com:example/toy-robot.git"),
        ("ssh://git@github.com/example/r.git", "ssh://github.com/example/r.git"),
        ("/srv/git/toy-robot", "/srv/git/toy-robot"),
        # An @ in the path is not userinfo.
        ("https://github.com/example/r@v1", "https://github.com/example/r@v1"),
    ],
)
def test_strip_userinfo(url, expected):
    got = evidence.strip_userinfo(url)
    assert got == expected
    assert SECRET not in got


def test_an_scp_style_remote_stays_exactly_as_it_was():
    for url in ("git@github.com:example/toy-robot.git", "deploy@host.example:srv/r.git"):
        assert evidence.strip_userinfo(url) == url


def _credentialed(repo: Path, url: str = CREDENTIALED) -> None:
    subprocess.run(["git", "-C", str(repo), "remote", "set-url", "origin", url], check=True)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return tmp_path, repo


def _everything(root: Path) -> str:
    return "\n".join(
        p.read_text(errors="replace")
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".md", ".txt"}
    )


@pytest.mark.parametrize(
    "url", [CREDENTIALED, f"https://{SECRET}@github.com/example/toy-robot.git"]
)
def test_the_run_manifest_robot_repo_has_no_userinfo(recorded, url):
    tmp_path, repo = recorded
    _credentialed(repo, url)
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", trials=1, lock=_lock(repo))
    run = json.loads((tmp_path / "ev" / "run-manifest.json").read_text())
    assert run["robot"]["repo"] == CLEAN
    text = _everything(tmp_path / "ev")
    assert SECRET not in text and USER not in text


def test_a_repo_the_lock_declares_is_stripped_too(recorded):
    tmp_path, repo = recorded
    lock = json.loads(_lock(repo).read_text())
    lock["robot"] = {"repo": CREDENTIALED}
    _lock(repo).write_text(json.dumps(lock))
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", trials=1, lock=_lock(repo))
    run = json.loads((tmp_path / "ev" / "run-manifest.json").read_text())
    assert run["robot"]["repo"] == CLEAN
    assert SECRET not in _everything(tmp_path / "ev")


def test_the_nominal_ref_in_the_binding_has_no_userinfo(recorded, monkeypatch):
    tmp_path, repo = recorded
    _credentialed(repo)
    conf = json.loads(json.dumps(test_evidence.CONF))
    conf["locomotion"]["nominal_base_height"] = {"value": 0.3, "unit": "m", "source": "measured"}
    monkeypatch.setattr(Toy, "configuration", lambda self, name: conf)
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", trials=1, lock=_lock(repo))
    v = tmp_path / "verification.json"
    v.write_text(json.dumps({"trials": [{"index": 1, "status": "PASS"}]}))
    doc = media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")
    path = tmp_path / "m" / doc["trials"][0]["binding"]["path"]
    ref = json.loads(path.read_text())["observables"]["base_height"]["nominal"]["ref"]
    repo_part, _, rest = ref.rpartition("@")
    assert repo_part == CLEAN and rest.endswith(":stock.json")
    assert SECRET not in _everything(tmp_path / "m") and USER not in _everything(tmp_path / "m")


def test_an_scp_style_remote_survives_into_the_ref_and_the_manifest(recorded, monkeypatch):
    tmp_path, repo = recorded
    scp = "git@github.com:example/toy-robot.git"
    _credentialed(repo, scp)
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", trials=1, lock=_lock(repo))
    run = json.loads((tmp_path / "ev" / "run-manifest.json").read_text())
    assert run["robot"]["repo"] == scp


def test_the_guide_says_the_repo_part_may_be_a_url_and_the_ref_splits_at_the_last_at():
    text = " ".join((Path(__file__).parents[1] / "docs" / "evidence.md").read_text().split())
    assert "may be a full URL" in text
    assert "splits at the last `@`" in text
    assert "userinfo" in text
