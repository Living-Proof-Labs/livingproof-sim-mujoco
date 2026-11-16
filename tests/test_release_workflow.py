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
