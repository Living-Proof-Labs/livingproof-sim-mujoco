# SPDX-License-Identifier: Apache-2.0
"""The principal workflow ends in an always-running completeness gate that the merge queue can require."""

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/verify.yml"
GATE_NEEDS = ["verify", "test-arm64", "publish-tests"]
# Jobs that deliberately sit outside the gate: they run only after a merge to main.
OUTSIDE_GATE = ["deploy-docs", "release-boundary", "release", "assemble"]


def _text():
    return WORKFLOW.read_text()


def _jobs(text):
    body = text.split("\njobs:\n", 1)[1]
    return re.findall(r"^  ([\w-]+):\s*$", body, flags=re.MULTILINE)


def _gate_block(text):
    after = text.split("\n  verification-gate:\n", 1)[1]
    return re.split(r"^  (?=[\w-]+:\s*$)", after, flags=re.MULTILINE)[0]


def test_workflow_runs_on_merge_group():
    assert re.search(
        r"^  merge_group:\n    types: \[checks_requested\]$", _text(), flags=re.MULTILINE
    )


def test_gate_always_runs_and_has_the_required_name():
    gate = _gate_block(_text())
    assert re.search(r"^    name: Repository verification gate$", gate, flags=re.MULTILINE)
    assert "if: always()" in gate
    assert "toJSON(needs)" in gate


def test_gate_needs_every_verification_job():
    text = _text()
    gate = _gate_block(text)
    needs = re.search(r"needs: \[([^\]]+)\]", gate).group(1)
    declared = sorted(n.strip() for n in needs.split(","))
    assert declared == sorted(GATE_NEEDS)
    others = {j for j in _jobs(text) if j != "verification-gate"}
    assert others - set(OUTSIDE_GATE) == set(GATE_NEEDS), (
        "a new job must join the gate or be listed as post-merge only"
    )


def _run_gate(results, event):
    gate = _gate_block(_text())
    script = re.search(r"python3 - <<'PYTHON'\n(.*?)\n\s*PYTHON", gate, flags=re.DOTALL).group(1)
    allowed = re.search(r"ALLOWED_SKIPS: '([^']*)'", gate).group(1)
    env = dict(
        os.environ,
        NEEDS_JSON=json.dumps({k: {"result": v} for k, v in results.items()}),
        ALLOWED_SKIPS=allowed,
        GITHUB_EVENT_NAME=event,
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_gate_passes_only_on_complete_success():
    for event in ("pull_request", "merge_group", "push"):
        assert _run_gate({j: "success" for j in GATE_NEEDS}, event).returncode == 0
        for bad in ("failure", "cancelled", "skipped", "neutral"):
            for job in GATE_NEEDS:
                results = {j: "success" for j in GATE_NEEDS}
                results[job] = bad
                allowed = json.loads(
                    re.search(r"ALLOWED_SKIPS: '([^']*)'", _gate_block(_text())).group(1)
                ).get(event, [])
                expect_ok = bad == "skipped" and job in allowed
                assert (_run_gate(results, event).returncode == 0) is expect_ok, (event, job, bad)
    assert _run_gate({}, "pull_request").returncode != 0
    assert _run_gate({j: "success" for j in GATE_NEEDS}, "issue_comment").returncode != 0
