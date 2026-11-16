# SPDX-License-Identifier: Apache-2.0
"""Real robot and configuration profiles carry their own sha256 fields (Vera review of #74).

Every robot repository's profiles cite pinned upstream files by sha256, for example
``robot.json`` ``model_source.sha256`` and a configuration's
``controller.artifact.sha256`` and ``settings.<file>.sha256``. The evidence manifest
embeds both profiles, so those fields are declared hashes too, and a real run must
still resolve with nothing undocumented. So must a lock that pins ``sources``.
"""

import copy
import hashlib
from pathlib import Path

import test_evidence
from livingproof_sim_mujoco import evidence
from test_evidence import TOY, Toy, _lock, _repo  # noqa: F401  (TOY is the adapter)

DIGEST = hashlib.sha256(b"pinned upstream file").hexdigest()


def test_profile_and_lock_source_hashes_of_a_real_robot_resolve(tmp_path, monkeypatch):
    robot = copy.deepcopy(test_evidence.ROBOT)
    robot["model_source"] = {"repository": "example/upstream", "path": "m.xml", "sha256": DIGEST}
    conf = copy.deepcopy(test_evidence.CONF)
    conf.setdefault("controller", {})["artifact"] = {"path": "policy.pt", "sha256": DIGEST}
    conf.setdefault("settings", {})["reference"] = {"path": "deploy.py", "sha256": DIGEST}
    monkeypatch.setattr(test_evidence, "ROBOT", robot)
    monkeypatch.setattr(test_evidence, "CONF", conf)
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    out = tmp_path / "out"
    evidence.evidence("test_evidence:TOY", "stock", out, trials=1, lock=_lock(repo))
    keys = {item["key"] for item in evidence.declared_hashes(out)}
    assert "robot.profile.model_source.sha256" in keys
    assert "configuration.profile.controller.artifact.sha256" in keys
    assert evidence.unresolved_hashes(out) == []
