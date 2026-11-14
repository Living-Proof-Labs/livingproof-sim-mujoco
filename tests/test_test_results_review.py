# SPDX-License-Identifier: Apache-2.0
"""Review probes for tools/test_results.py that are not in tests/test_test_results.py.

These cover malformed and zero-test reports, HTTP 500/403/timeout/network errors,
a missing token, and a 302 that must not be followed. No live GitHub API calls.
"""

from http.client import HTTPMessage
from io import BytesIO
import importlib.util
from pathlib import Path
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("junit_gate", ROOT / "tools/test_results.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

SHA = "e" * 40
COMPLETED_AT = "2026-11-14T07:57:00Z"
HTML = "https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/runs/132382544137"
REPOSITORY = "Living-Proof-Labs/livingproof-sim-mujoco"
CHECK_NAME = "MuJoCo tests (amd64)"


class FakeChecks:
    def __init__(self, *, create_status=201, response_statuses=None, update_statuses=None):
        self.create_status = create_status
        self.response_statuses = list(response_statuses or [])
        self.update_statuses = list(update_statuses or [])
        self.runs = []
        self.updates = []
        self.created_bodies = []

    def create_check_run(self, body):
        if self.create_status != 201:
            raise gate.CheckApiError(self.create_status, "create")
        status = self.response_statuses.pop(0) if self.response_statuses else body["status"]
        run = {
            "id": 132382544137 + len(self.runs),
            "status": status,
            "conclusion": body.get("conclusion") if status == "completed" else None,
            "html_url": HTML,
            "name": body["name"],
        }
        self.runs.append(run)
        self.created_bodies.append(body)
        return dict(run)

    def update_check_run(self, check_id, body):
        self.updates.append({"id": check_id, "body": body})
        status = self.update_statuses.pop(0) if self.update_statuses else 200
        if status != 200:
            raise gate.CheckApiError(status, "update")
        run = next(item for item in self.runs if item["id"] == check_id)
        run["status"] = body.get("status", run["status"])
        if "conclusion" in body:
            run["conclusion"] = body["conclusion"]
        return dict(run)


def write_junit(path, cases):
    suite = ET.Element("testsuite")
    for name, result, detail in cases:
        case = ET.SubElement(suite, "testcase", name=name, classname="tests.example")
        if result in ("failure", "error"):
            ET.SubElement(case, result, message=detail).text = detail
        elif result == "skipped":
            ET.SubElement(case, "skipped", message=detail)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="unicode")


def run_cli(
    tmp_path, monkeypatch, capsys, api, *, publish="true", pattern="*.xml", token="test-token"
):
    if token is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("HEAD_SHA", SHA)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(gate, "client_from_env", lambda env=None: api)
    code = gate.main(
        [
            "--name",
            CHECK_NAME,
            "--path",
            pattern,
            "--publish-check",
            publish,
            "--root",
            str(tmp_path),
            "--completed-at",
            COMPLETED_AT,
        ]
    )
    printed = capsys.readouterr().out
    output = (tmp_path / "output").read_text() if (tmp_path / "output").exists() else ""
    summary = (tmp_path / "summary").read_text() if (tmp_path / "summary").exists() else ""
    if token:
        assert token not in printed + output + summary
    return code, printed, output, summary


def test_malformed_report_fails_without_creating_a_check(tmp_path, monkeypatch, capsys):
    (tmp_path / "tests.xml").write_text("not xml")
    api = FakeChecks()
    code, printed, _output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert printed == ""
    assert api.runs == []
    assert "could not be read" in summary


def test_zero_tests_attribute_fails_without_creating_a_check(tmp_path, monkeypatch, capsys):
    (tmp_path / "tests.xml").write_text('<testsuite tests="0" failures="0"></testsuite>')
    api = FakeChecks()
    code, printed, _output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert printed == ""
    assert api.runs == []
    assert "no test cases were found" in summary


def test_create_500_on_passing_tests_warns_and_exits_0(tmp_path, monkeypatch, capsys):
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", "")])
    api = FakeChecks(create_status=500)
    code, printed, output, _summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 0
    assert "::warning::" in printed
    assert "500" in printed
    assert "create" in printed
    assert api.runs == []
    assert "url_html=\n" in output


def test_create_403_on_passing_tests_warns_and_exits_0(tmp_path, monkeypatch, capsys):
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", "")])
    api = FakeChecks(create_status=403)
    code, printed, output, _summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 0
    assert "::warning::" in printed
    assert "403" in printed
    assert api.runs == []
    assert "url_html=\n" in output


def test_failed_junit_still_fails_when_create_returns_500(tmp_path, monkeypatch, capsys):
    write_junit(
        tmp_path / "tests.xml",
        [("test_reporting_failure_fixture", "failure", "Deliberate reporting fixture failure")],
    )
    api = FakeChecks(create_status=500)
    code, printed, output, _summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert "failed=1" in output
    assert "500" in printed
    assert api.runs == []


def test_missing_token_skips_publish_and_passing_tests_exit_0(tmp_path, monkeypatch, capsys):
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", "")])
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("HEAD_SHA", SHA)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    # Use the real client_from_env, not a stub.
    code = gate.main(
        [
            "--name",
            CHECK_NAME,
            "--path",
            "*.xml",
            "--publish-check",
            "true",
            "--root",
            str(tmp_path),
            "--completed-at",
            COMPLETED_AT,
        ]
    )
    printed = capsys.readouterr().out
    output = (tmp_path / "output").read_text()
    assert code == 0
    assert "::warning::" in printed
    assert "token" in printed.lower()
    assert "passed=1" in output


def test_github_client_maps_timeout_and_http_500_without_a_live_request():
    def timeout_opener(request, timeout):
        raise TimeoutError("timed out")

    client = gate.GitHubChecks("test-token", REPOSITORY, opener=timeout_opener)
    with pytest.raises(gate.CheckApiError) as caught:
        client.create_check_run(
            {"name": CHECK_NAME, "status": "completed", "conclusion": "success"}
        )
    assert caught.value.status == 0
    assert caught.value.operation == "create"
    assert "before the API responded" in str(caught.value)
    assert "test-token" not in str(caught.value)

    def error_500(request, timeout):
        raise HTTPError(request.full_url, 500, "Server Error", HTTPMessage(), BytesIO(b"{}"))

    client = gate.GitHubChecks("test-token", REPOSITORY, opener=error_500)
    with pytest.raises(gate.CheckApiError) as caught:
        client.create_check_run(
            {"name": CHECK_NAME, "status": "completed", "conclusion": "success"}
        )
    assert caught.value.status == 500
    assert "HTTP 500" in str(caught.value)
    assert "test-token" not in str(caught.value)

    def refused(request, timeout):
        raise URLError("Connection refused")

    client = gate.GitHubChecks("test-token", REPOSITORY, opener=refused)
    with pytest.raises(gate.CheckApiError) as caught:
        client.create_check_run(
            {"name": CHECK_NAME, "status": "completed", "conclusion": "success"}
        )
    assert caught.value.status == 0
    assert "test-token" not in str(caught.value)


def test_github_client_does_not_follow_a_redirect():
    seen = []

    def opener(request, timeout):
        seen.append(request)
        raise HTTPError(request.full_url, 302, "Found", HTTPMessage(), BytesIO(b""))

    client = gate.GitHubChecks("test-token", REPOSITORY, opener=opener)
    with pytest.raises(gate.CheckApiError) as caught:
        client.create_check_run(
            {"name": CHECK_NAME, "status": "completed", "conclusion": "success"}
        )
    assert caught.value.status == 302
    assert len(seen) == 1
    assert "test-token" not in str(caught.value)


def test_publish_tests_still_runs_when_verify_fails_and_skips_forks():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text()
    publish = workflow.split("  publish-tests:", 1)[1].split("  deploy-docs:", 1)[0]
    assert "needs: verify" in publish
    assert "!cancelled()" in publish
    assert "head.repo.full_name == github.repository" in publish
    assert "continue-on-error" not in publish
    assert "persist-credentials: false" in publish
    # #45 will add the attempt suffix; this PR still uses the unsuffixed name.
    assert "name: mujoco-evidence-amd64" in publish
    assert "github.run_attempt" not in publish
