# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_publishes_itself_after_the_tested_assets_are_attached():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text()
    upload = workflow.index('gh release upload "$RELEASE_TAG"')
    publish = workflow.index('gh release edit "$RELEASE_TAG" --draft=false')
    assert upload < publish  # never public without its assets
    assert "if: needs.release.outputs.draft-created == 'true'" in workflow
