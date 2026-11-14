# SPDX-License-Identifier: Apache-2.0
"""Check-run API failures warn; JUnit failures still fail. No live API calls."""

from http.client import HTTPMessage
from io import BytesIO
import importlib.util
from pathlib import Path
from urllib.error import HTTPError
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
    """In-memory checks API. Statuses are consumed in call order."""

    def __init__(self, *, create_status=201, response_statuses=None, update_statuses=None):
        self.create_status = create_status
        self.response_statuses = list(response_statuses or [])
        self.update_statuses = list(update_statuses or [])
        self.runs = []
        self.updates = []
        self.created_bodies = []

    def create_check_run(self, body):
        if self.create_status == 404:
            raise gate.CheckApiError(404, "create")
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
        if status == 404:
            raise gate.CheckApiError(404, "update")
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


def run_cli(tmp_path, monkeypatch, capsys, api, *, publish="true", pattern="*.xml"):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
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
    output = (tmp_path / "output").read_text()
    summary = (tmp_path / "summary").read_text()
    assert "test-token" not in printed + output + summary
    return code, printed, output, summary


def test_update_404_on_passing_tests_warns_and_finishes_the_check(tmp_path, monkeypatch, capsys):
    """The incident shape: tests passed, and the check-run update returns Not Found."""
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", "")])
    api = FakeChecks(response_statuses=["in_progress"], update_statuses=[404, 200])
    code, printed, output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 0
    assert "::warning::" in printed
    assert "404 Not Found" in printed
    assert "update" in printed
    assert api.created_bodies[0]["status"] == "completed"
    assert api.created_bodies[0]["conclusion"] == "success"
    assert api.created_bodies[0]["head_sha"] == SHA
    assert api.created_bodies[0]["name"] == CHECK_NAME
    assert "output" in api.updates[0]["body"]
    assert "output" not in api.updates[1]["body"]
    assert api.updates[1]["body"]["conclusion"] == "success"
    assert api.runs[0]["status"] == "completed"
    assert api.runs[0]["conclusion"] == "success"
    assert len(api.runs) == 1
    assert f"url_html={HTML}" in output
    assert "passed 1, failed 0, skipped 0" in summary
    assert "404 Not Found" in summary


def test_both_updates_404_on_passing_tests_publishes_a_completed_run(tmp_path, monkeypatch, capsys):
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", "")])
    api = FakeChecks(response_statuses=["in_progress"], update_statuses=[404, 404])
    code, printed, output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 0
    assert "::warning::" in printed
    assert "404 Not Found" in printed
    assert api.created_bodies[0]["status"] == "completed"
    assert api.created_bodies[1]["status"] == "completed"
    assert api.created_bodies[1]["conclusion"] == "success"
    assert api.runs[0]["status"] == "in_progress"
    assert api.runs[1]["status"] == "completed"
    assert api.runs[1]["conclusion"] == "success"
    assert f"url_html={HTML}" in output
    assert "passed 1, failed 0, skipped 0" in summary
    assert "not left in progress" in printed


def test_failed_junit_still_fails_when_the_check_update_returns_404(tmp_path, monkeypatch, capsys):
    write_junit(
        tmp_path / "tests.xml",
        [("test_reporting_failure_fixture", "failure", "Deliberate reporting fixture failure")],
    )
    api = FakeChecks(response_statuses=["in_progress"], update_statuses=[404, 200])
    code, printed, output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert "failed=1" in output
    assert "::warning::" in printed
    assert "404 Not Found" in printed
    assert api.created_bodies[0]["conclusion"] == "failure"
    assert "test_reporting_failure_fixture" in api.created_bodies[0]["output"]["summary"]
    assert api.runs[0]["status"] == "completed"
    assert api.runs[0]["conclusion"] == "failure"
    assert "failed 1" in summary


def test_failed_junit_still_fails_when_create_returns_404(tmp_path, monkeypatch, capsys):
    write_junit(
        tmp_path / "tests.xml",
        [("test_reporting_failure_fixture", "failure", "AssertionError: assert 1 == 0")],
    )
    api = FakeChecks(create_status=404)
    code, printed, output, _summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert "failed=1" in output
    assert api.runs == []
    assert "create" in printed
    assert "404 Not Found" in printed


def test_missing_report_fails_without_creating_a_check(tmp_path, monkeypatch, capsys):
    api = FakeChecks()
    code, printed, _output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert printed == ""
    assert api.runs == []
    assert "no test cases were found" in summary


def test_empty_report_fails_without_creating_a_check(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tests.xml"
    path.write_text("<testsuite></testsuite>")
    api = FakeChecks()
    code, printed, _output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 1
    assert printed == ""
    assert api.runs == []
    assert "no test cases were found" in summary


def test_completed_create_publishes_passing_results_without_an_update(
    tmp_path, monkeypatch, capsys
):
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", ""), ("test_skip", "skipped", "")])
    api = FakeChecks()
    code, printed, output, summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 0
    assert printed == ""
    assert api.updates == []
    assert api.created_bodies[0]["status"] == "completed"
    assert api.created_bodies[0]["conclusion"] == "success"
    assert api.runs[0]["status"] == "completed"
    assert "passed=1\nfailed=0\nskipped=1\n" in output
    assert "passed 1, failed 0, skipped 1" in summary


def test_top_level_glob_does_not_read_nested_artifact_xml(tmp_path, monkeypatch, capsys):
    """The evidence artifact also contains manual and sitemap XML. Those are not the suite."""
    write_junit(tmp_path / "tests.xml", [("test_ok", "pass", "")])
    write_junit(
        tmp_path / "manual/tests.xml",
        [("test_hidden", "failure", "nested report is not the suite")],
    )
    (tmp_path / "manual/html").mkdir(parents=True)
    (tmp_path / "manual/html/sitemap.xml").write_text("<urlset></urlset>")
    api = FakeChecks()
    code, printed, output, _summary = run_cli(tmp_path, monkeypatch, capsys, api)
    assert code == 0
    assert printed == ""
    assert "failed=0" in output
    assert "test_hidden" not in api.created_bodies[0]["output"]["summary"]


def test_github_client_maps_not_found_without_a_live_request():
    seen = []

    def opener(request, timeout):
        seen.append(request)
        raise HTTPError(request.full_url, 404, "Not Found", HTTPMessage(), BytesIO(b"{}"))

    client = gate.GitHubChecks("test-token", REPOSITORY, opener=opener)
    with pytest.raises(gate.CheckApiError) as caught:
        client.update_check_run(132382544137, {"status": "completed", "conclusion": "success"})
    assert caught.value.status == 404
    assert caught.value.operation == "update"
    assert "Not Found" in str(caught.value)
    assert "test-token" not in str(caught.value)
    request = seen[0]
    assert request.get_method() == "PATCH"
    assert request.full_url.endswith(f"/repos/{REPOSITORY}/check-runs/132382544137")
    assert request.get_header("Authorization") == "Bearer test-token"
    assert "test-token" not in request.full_url
    assert (
        gate.RejectRedirect().redirect_request(
            None, None, 302, "Found", None, "https://evil.example"
        )
        is None
    )


def _job(workflow, name, following):
    return workflow.split(f"  {name}:", 1)[1].split(f"  {following}:", 1)[0]


def test_publish_tests_job_creates_a_completed_check_without_new_permissions():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text()
    verify = _job(workflow, "verify", "publish-tests")
    publish = _job(workflow, "publish-tests", "deploy-docs")
    assert "dorny/test-reporter@" in verify
    assert "use-actions-summary: 'true'" in verify
    assert "name: mujoco-evidence-amd64" in verify
    assert "retention-days: 14" in verify
    assert "use-actions-summary: 'false'" not in workflow
    assert "dorny/test-reporter" not in publish
    assert "tools/test_results.py" in publish
    assert 'name: "MuJoCo tests (amd64)"' in publish or "--name" in publish
    assert CHECK_NAME in publish
    assert "--path '*.xml'" in publish
    assert "--root reports" in publish
    assert "continue-on-error" not in publish
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in publish
    assert "name: mujoco-evidence-amd64" in publish
    assert "github.event.pull_request.head.sha || github.sha" in publish
    assert "github.token" in publish
    assert "secrets." not in publish
    assert publish.count("permissions:") == 1
    permissions = publish.split("permissions:", 1)[1].split("steps:", 1)[0]
    assert [line.strip() for line in permissions.strip().splitlines()] == [
        "actions: read",
        "contents: read",
        "checks: write",
    ]
    assert "head.repo.full_name == github.repository" in publish
