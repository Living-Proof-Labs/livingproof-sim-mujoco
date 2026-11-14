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
