# SPDX-License-Identifier: Apache-2.0
"""The evidence manifest records the digest of the parameters file the run loaded (#104).

``manifest.json`` stores ``sources.parameters`` = ``{"path", "sha256"}``: the lock's
``sources.parameters`` path and the sha256 of the one read that was parsed and used.
Run Integrity can then compare what the run used with what the lock declares.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import test_scenario_sources as t
import test_scenario_supported as supported

from livingproof_sim_mujoco import evidence, sources

REAL = Path(__file__).parent / "fixtures" / "real-runs"


def _lock(repo: Path) -> dict:
    return json.loads((repo / "caller.lock.json").read_text())


def test_manifest_records_the_loaded_parameters_file(tmp_path, monkeypatch):
    _toy, repo = t._toy_repo(tmp_path, monkeypatch)
    out = tmp_path / "out"
    manifest = t._evidence(repo, out, trials=1)
    declared = _lock(repo)["sources"]["parameters"]
    recorded = manifest["sources"]["parameters"]
    assert recorded == {"path": declared["path"], "sha256": declared["sha256"]}
    on_disk = (repo / declared["path"]).read_bytes()
    assert recorded["sha256"] == hashlib.sha256(on_disk).hexdigest()
    # What was written is what was returned, and it resolves.
    assert json.loads((out / "manifest.json").read_text())["sources"] == manifest["sources"]
    assert evidence.unresolved_hashes(out) == []


def test_the_record_follows_the_read_not_a_later_read(tmp_path, monkeypatch):
    """Change the file after the lock check has passed: the record is the bytes that were read."""
    _toy, repo = t._toy_repo(tmp_path, monkeypatch)
    declared = _lock(repo)["sources"]["parameters"]
    path = repo / declared["path"]
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    reads = []
    real = sources._read_locked

    def read_then_change(root, entry, kind):
        data = real(root, entry, kind)
        if kind == "parameters":
            reads.append(hashlib.sha256(data).hexdigest())
            document = json.loads(data)
            document["note"] = "changed after the lock check"
            path.write_text(json.dumps(document))
        return data

    monkeypatch.setattr(sources, "_read_locked", read_then_change)
    manifest = t._evidence(repo, tmp_path / "out", trials=1)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != original
    assert manifest["sources"]["parameters"]["sha256"] == original
    # One read per run: the support check and the run share it.
    assert reads == [original]


def test_an_unsupported_scenario_records_the_file_it_read_too(tmp_path, monkeypatch):
    repo = supported._repo(tmp_path, monkeypatch)
    supported._use_parameters(repo, supported._require_support(supported._parameters()))
    supported._write_manifest(repo, supported._manifest([]))
    out = tmp_path / "out"
    recorded = evidence.evidence(
        "test_evidence:TOY", "stock", out, trials=1, lock=repo / "caller.lock.json"
    )
    assert recorded["trials"]["results"] == []
    declared = _lock(repo)["sources"]["parameters"]
    assert recorded["sources"]["parameters"] == {
        "path": declared["path"],
        "sha256": declared["sha256"],
    }
    assert evidence.unresolved_hashes(out) == []


def test_run_manifest_is_unchanged(tmp_path, monkeypatch):
    """run-manifest.json follows robot-capability-specs; this slice does not touch it."""
    _toy, repo = t._toy_repo(tmp_path, monkeypatch)
    out = tmp_path / "out"
    t._evidence(repo, out, trials=1)
    assert "sources" not in json.loads((out / "run-manifest.json").read_text())


def test_the_recorded_digest_is_a_documented_source_hash(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(REAL / "g1-stock", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["sources"] = {"parameters": {"path": "parameters.json", "sha256": "ab" * 32}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    [item] = [i for i in evidence.declared_hashes(run) if i["key"].startswith("sources.")]
    assert item["key"] == "sources.parameters.sha256"
    assert item["covers"] == "source"
    assert evidence.unresolved_hashes(run) == []


def test_a_real_run_without_the_field_still_resolves():
    for run in ("g1-stock", "t1-stock", "t1-playground", "go1-playground"):
        manifest = json.loads((REAL / run / "manifest.json").read_text())
        assert "sources" not in manifest
        assert evidence.unresolved_hashes(REAL / run) == []


def test_parameters_source_names_the_bytes_that_were_read(tmp_path):
    root = tmp_path
    document = {"parameters": {}, "k": 1}
    raw = json.dumps(document).encode()
    (root / "p.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    lock = {"sources": {"parameters": {"path": "p.json", "sha256": digest}}}
    selected = sources.parameters_source(root, lock)
    assert selected.document == document
    assert selected.origin == "lock"
    assert selected.sha256 == digest
    assert selected.path == "p.json"
    # The tuple form used elsewhere is unchanged.
    assert sources.parameters_document_for(root, lock) == (document, "lock")


def test_an_unsupported_run_records_the_bytes_it_read_not_a_later_read(tmp_path, monkeypatch):
    """The UNSUPPORTED path has its own manifest writer; it must follow the one read too (#104).

    Change the file right after the lock check passes. A writer that hashed the file again would
    record the changed digest; the record must stay the digest of the bytes that were read.
    """
    repo = supported._repo(tmp_path, monkeypatch)
    supported._use_parameters(repo, supported._require_support(supported._parameters()))
    supported._write_manifest(repo, supported._manifest([]))
    declared = _lock(repo)["sources"]["parameters"]
    path = repo / declared["path"]
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    reads = []
    real = sources._read_locked

    def read_then_change(root, entry, kind):
        data = real(root, entry, kind)
        if kind == "parameters":
            reads.append(hashlib.sha256(data).hexdigest())
            document = json.loads(data)
            document["note"] = "changed after the lock check"
            path.write_text(json.dumps(document))
        return data

    monkeypatch.setattr(sources, "_read_locked", read_then_change)
    recorded = evidence.evidence(
        "test_evidence:TOY", "stock", tmp_path / "out", trials=1, lock=repo / "caller.lock.json"
    )
    assert recorded["trials"]["results"] == []
    assert hashlib.sha256(path.read_bytes()).hexdigest() != original
    assert recorded["sources"]["parameters"]["sha256"] == original
    assert (
        json.loads((tmp_path / "out" / "manifest.json").read_text())["sources"]
        == (recorded["sources"])
    )
    assert reads == [original]


def test_the_record_has_exactly_the_keys_and_digest_format_consumers_compare(tmp_path, monkeypatch):
    """livingproof-evaluator-run compares strings exactly: the lock's own form, nothing added.

    ``sources.parameters.sha256`` is 64 lowercase hex characters with no ``sha256:`` prefix,
    the form of the lock's ``sources.parameters.sha256`` and ``releases.specs.files`` values.
    The new top-level key holds ``parameters`` only, with ``path`` and ``sha256`` only.
    """
    import re

    _toy, repo = t._toy_repo(tmp_path, monkeypatch)
    manifest = t._evidence(repo, tmp_path / "out", trials=1)
    assert set(manifest["sources"]) == {"parameters"}
    assert set(manifest["sources"]["parameters"]) == {"path", "sha256"}
    digest = manifest["sources"]["parameters"]["sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == _lock(repo)["sources"]["parameters"]["sha256"]
