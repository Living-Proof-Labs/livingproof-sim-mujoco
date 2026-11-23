# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_publishes_itself_after_the_tested_assets_are_attached():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text()
    upload = workflow.index('gh release upload "$RELEASE_TAG"')
    publish = workflow.index('gh release edit "$RELEASE_TAG" --draft=false')
    assert upload < publish  # never public without its assets
    assert "if: needs.release.outputs.draft-created == 'true'" in workflow


def test_every_release_upload_replaces_assets_so_a_rerun_still_publishes():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text()
    uploads = [line for line in workflow.splitlines() if "gh release upload" in line]
    assert uploads and all("--clobber" in line for line in uploads)


def test_publish_tests_reads_only_this_run_attempt_evidence():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text()
    evidence = "mujoco-evidence-amd64-${{ github.run_attempt }}"
    assert f"name: {evidence}" in workflow
    publish = workflow.split("publish-tests:", 1)[1].split("deploy-docs:", 1)[0]
    assert f"name: {evidence}" in publish
    assert "name: mujoco-evidence-amd64\n" not in publish
    assert "tools/test_results.py" in publish


def _workflow_jobs():
    import yaml

    return yaml.safe_load((ROOT / ".github/workflows/verify.yml").read_text())["jobs"]


def _taskfile_evidence_command():
    import yaml

    tasks = yaml.safe_load((ROOT / "Taskfile.runtime.yml").read_text())["tasks"]
    commands = [c for c in tasks["test"]["cmds"] if isinstance(c, str) and "test_evidence.py" in c]
    assert len(commands) == 1
    return commands[0]


def test_the_zero_command_stand_runs_on_arm64_with_egl():
    """#38 asks for the zero-command stand on amd64 and arm64: same command, no -k exclusion."""
    arm = [job for job in _workflow_jobs().values() if job.get("runs-on") == "ubuntu-24.04-arm"]
    assert len(arm) == 1, "one MuJoCo tests (arm64) job on the free ubuntu-24.04-arm runner"
    runs = [step.get("run", "") for step in arm[0]["steps"]]
    command = _taskfile_evidence_command()
    assert command.startswith("MUJOCO_GL=egl ") and " -k " not in command
    assert any(command in run for run in runs), runs
    assert any(run.strip() == "task setup" for run in runs)
    uploads = [step.get("with", {}).get("name") for step in arm[0]["steps"]]
    assert "mujoco-evidence-arm64-${{ github.run_attempt }}" in uploads


def test_publish_tests_publishes_an_arm64_check_from_this_attempt():
    jobs = _workflow_jobs()
    arm = next(name for name, job in jobs.items() if job.get("runs-on") == "ubuntu-24.04-arm")
    publish = jobs["publish-tests"]
    assert arm in publish["needs"] and "verify" in publish["needs"]
    downloads = [
        s["with"]["name"] for s in publish["steps"] if "download-artifact" in s.get("uses", "")
    ]
    assert "mujoco-evidence-amd64-${{ github.run_attempt }}" in downloads
    assert "mujoco-evidence-arm64-${{ github.run_attempt }}" in downloads
    runs = "\n".join(step.get("run", "") for step in publish["steps"])
    assert 'tools/test_results.py --name "MuJoCo tests (amd64)"' in runs
    assert 'tools/test_results.py --name "MuJoCo tests (arm64)"' in runs


def test_publish_tests_publishes_the_arm64_limit_tests_as_their_own_check():
    """limits.xml is uploaded with evidence.xml; without its own publish it had no per-test check."""
    publish = _workflow_jobs()["publish-tests"]
    steps = [s for s in publish["steps"] if "MuJoCo limit tests (arm64)" in s.get("run", "")]
    assert len(steps) == 1
    run = steps[0]["run"]
    assert 'tools/test_results.py --name "MuJoCo limit tests (arm64)"' in run
    assert "--path 'limits.xml'" in run and "--root reports-arm64" in run
    assert steps[0].get("if") == "${{ !cancelled() }}"
    assert "continue-on-error" not in steps[0]
    assert steps[0]["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha || github.sha }}"


def test_pages_builds_only_after_release_and_checks_uploaded_bytes():
    jobs = _workflow_jobs()
    assert not any("upload-pages-artifact" in s.get("uses", "") for s in jobs["verify"]["steps"])
    assert any(
        s.get("with", {}).get("name") == "documentation-preview" for s in jobs["verify"]["steps"]
    )
    deploy = jobs["deploy-docs"]
    assert deploy["needs"] == ["release", "assemble"]
    assert "needs.assemble.result == 'success'" in deploy["if"]
    assert "needs.release.outputs.draft-created == 'true'" in deploy["if"]
    assert deploy["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert deploy["concurrency"] == {"group": "mujoco-pages", "cancel-in-progress": False}
    steps = deploy["steps"]
    runs = [s.get("run", "") for s in steps]
    guard = next(i for i, run in enumerate(runs) if "python3 tools/released_manual.py\n" in run)
    build = runs.index("task docs")
    check = runs.index("python3 tools/released_manual.py --site")
    upload = next(i for i, s in enumerate(steps) if s.get("id") == "pages-artifact")
    assert guard < build < runs.index("task test:docs") < check < upload
    artifact = steps[upload]["with"]["name"]
    assert "needs.release.outputs.version" in artifact and "github.sha" in artifact
    assert steps[-1]["with"]["artifact_name"] == artifact
    assert steps[0]["with"]["ref"] == "${{ needs.release.outputs.git-tag }}"
    # Delivery retry has no release mutation and never needs the writer rerun.
    assert not any("gh release " in run or "git tag " in run for run in runs)


def _release_guard():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "released_manual", ROOT / "tools/released_manual.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_gate_rejects_unpublished_mismatched_and_dirty_sources(tmp_path):
    import subprocess
    import pytest

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "fixture",
    )
    revision = git("rev-parse", "HEAD")
    git("tag", "v1.2.3")
    record = dict(
        tag_name="v1.2.3",
        draft=False,
        prerelease=False,
        immutable=True,
        published_at="2026-11-22T00:00:00Z",
    )
    guard = _release_guard().validate_release
    before = git("show-ref", "--tags")
    for _ in range(2):  # initial delivery and retry reuse the existing release/tag
        guard(record, "1.2.3", "v1.2.3", revision, tmp_path)
    assert git("show-ref", "--tags") == before
    for key, value in [
        ("draft", True),
        ("immutable", False),
        ("prerelease", True),
        ("published_at", None),
        ("tag_name", "v1.2.4"),
    ]:
        with pytest.raises(ValueError):
            guard(dict(record, **{key: value}), "1.2.3", "v1.2.3", revision, tmp_path)
    for version, tag, sha in [
        ("", "", revision),
        ("1.2.4", "v1.2.3", revision),
        ("1.2.3", "v1.2.3", "a" * 40),
        ("1.2.3-pr.1", "v1.2.3-pr.1", revision),
    ]:
        with pytest.raises(ValueError):
            guard(record, version, tag, sha, tmp_path)
    with pytest.raises(ValueError):
        guard({}, "1.2.3", "v1.2.3", revision, tmp_path)
    git("tag", "-d", "v1.2.3")
    with pytest.raises(subprocess.CalledProcessError):
        guard(record, "1.2.3", "v1.2.3", revision, tmp_path)
    git("tag", "v1.2.3")
    (tmp_path / "dirty").write_text("dirty")
    with pytest.raises(ValueError):
        guard(record, "1.2.3", "v1.2.3", revision, tmp_path)


def test_pages_guard_rejects_preview_or_mislabeled_bytes(tmp_path):
    import json
    import pytest

    guard = _release_guard().validate_site
    revision = "a" * 40
    record = dict(
        version="1.2.3",
        revision=revision,
        channel="stable",
        dirty=False,
        identity={"package_version": "1.2.3"},
    )
    for name in ("index.html", "context.html"):
        (tmp_path / name).write_text(f"Docs: 1.2.3 · {revision}")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(record))
    guard(tmp_path, "1.2.3", revision)
    for key, value in [
        ("version", "1.2.2"),
        ("revision", "b" * 40),
        ("channel", "development"),
        ("dirty", True),
        ("identity", {"package_version": "1.2.4"}),
    ]:
        provenance.write_text(json.dumps(dict(record, **{key: value})))
        with pytest.raises(ValueError):
            guard(tmp_path, "1.2.3", revision)
    provenance.write_text(json.dumps(record))
    (tmp_path / "index.html").write_text("Docs: development")
    with pytest.raises(ValueError):
        guard(tmp_path, "1.2.3", revision)


def _pages_step(name):
    return next(s for s in _workflow_jobs()["deploy-docs"]["steps"] if s.get("name") == name)


def _run_pages_step(name, root, **env):
    import os
    import subprocess

    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _pages_step(name)["run"]],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )


def test_pages_readback_uses_current_id_and_attempt_name_with_bounded_native_retries():
    deploy = _workflow_jobs()["deploy-docs"]
    steps = deploy["steps"]
    upload = next(s for s in steps if s.get("id") == "pages-artifact")
    assert "upload-artifact@" in upload["uses"]
    assert upload["with"]["path"] == "${{ runner.temp }}/artifact.tar"
    assert upload["with"]["name"] == steps[-1]["with"]["artifact_name"]
    guard = _pages_step("Require matching ID and name readback of the exact tar")
    assert (
        steps.index(upload)
        < steps.index(guard)
        < steps.index(_pages_step("Check current main inside deployment lock"))
    )
    assert guard["env"]["EXPECTED_TAR"] == "${{ steps.pages-package.outputs.tar-sha256 }}"
    assert guard["env"]["EXPECTED_NAME"] == upload["with"]["name"]
    assert _pages_step("Require a valid Pages artifact ID")["env"]["UPLOADED_ID"] == (
        "${{ steps.pages-artifact.outputs.artifact-id }}"
    )
    for route, selector, expected in [
        ("id", "artifact-ids", "${{ steps.pages-artifact.outputs.artifact-id }}"),
        ("name", "name", upload["with"]["name"]),
    ]:
        first = next(s for s in steps if s.get("id") == f"pages-{route}-first")
        retry = next(s for s in steps if s.get("id") == f"pages-{route}-retry")
        assert first["continue-on-error"] is True
        assert retry["if"] == f"steps.pages-{route}-first.outcome == 'failure'"
        assert "continue-on-error" not in retry
        assert first["uses"] == retry["uses"]
        assert "download-artifact@" in first["uses"]
        for step in (first, retry):
            assert set(step["with"]) == {selector, "path"}
            assert step["with"][selector] == expected
        assert first["with"]["path"] != retry["with"]["path"]
        source = guard["env"][f"{route.upper()}_READBACK"]
        assert f"steps.pages-{route}-first.outcome == 'success'" in source
        assert f"steps.pages-{route}-retry.outputs.download-path" in source
    pauses = [s for s in steps if s.get("run") == "sleep 4"]
    assert len(pauses) == 2
    assert {s["if"] for s in pauses} == {
        f"steps.pages-{route}-first.outcome == 'failure'" for route in ("id", "name")
    }


def test_pages_package_hashes_only_the_actual_html_tar(tmp_path):
    import hashlib
    import tarfile

    html = tmp_path / "artifacts/manual/html"
    html.mkdir(parents=True)
    (html / "index.html").write_text("verified manual")
    (html / ".nojekyll").touch()
    output = tmp_path / "outputs"
    result = _run_pages_step(
        "Package the verified Pages HTML",
        tmp_path,
        RUNNER_TEMP=str(tmp_path),
        GITHUB_OUTPUT=str(output),
    )
    assert result.returncode == 0, result.stderr
    archive = tmp_path / "artifact.tar"
    assert output.read_text() == f"tar-sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}\n"
    with tarfile.open(archive) as package:
        assert sorted(package.getnames()) == [".", "./.nojekyll", "./index.html"]
    (html / "linked").symlink_to(output)
    result = _run_pages_step(
        "Package the verified Pages HTML",
        tmp_path,
        RUNNER_TEMP=str(tmp_path),
        GITHUB_OUTPUT=str(output),
    )
    assert result.returncode != 0


def test_pages_readback_rejects_stale_mismatched_and_missing_data(tmp_path):
    import hashlib

    expected_name = "github-pages-1.2.3-" + "a" * 40 + "-2"
    for kind in (
        "exact",
        "missing-id",
        "missing-name",
        "wrong-id-bytes",
        "wrong-name-bytes",
        "stale-attempt",
        "empty-id-path",
        "empty-name-path",
        "invalid-digest",
    ):
        root = tmp_path / kind
        identifier, named = root / "id", root / "name"
        id_location = identifier / (
            expected_name[:-1] + "1" if kind == "stale-attempt" else expected_name
        )
        id_location.mkdir(parents=True)
        named.mkdir()
        if kind != "missing-id":
            (id_location / "artifact.tar").write_bytes(
                b"other" if kind == "wrong-id-bytes" else b"sealed"
            )
        if kind != "missing-name":
            (named / "artifact.tar").write_bytes(
                b"other" if kind == "wrong-name-bytes" else b"sealed"
            )
        result = _run_pages_step(
            "Require matching ID and name readback of the exact tar",
            root,
            ID_READBACK="" if kind == "empty-id-path" else str(identifier),
            NAME_READBACK="" if kind == "empty-name-path" else str(named),
            EXPECTED_NAME=expected_name,
            EXPECTED_TAR="" if kind == "invalid-digest" else hashlib.sha256(b"sealed").hexdigest(),
        )
        assert (result.returncode == 0) == (kind == "exact"), (kind, result.stderr)
    for artifact_id in ("12345", "", "0", "-1", "stale", "123,124"):
        result = _run_pages_step(
            "Require a valid Pages artifact ID", tmp_path, UPLOADED_ID=artifact_id
        )
        assert (result.returncode == 0) == (artifact_id == "12345"), artifact_id
