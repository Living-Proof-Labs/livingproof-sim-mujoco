# SPDX-License-Identifier: Apache-2.0
"""Gate a job on JUnit XML and publish one already-completed check run.

dorny/test-reporter's check mode creates a run with status in_progress and then
updates it. That update returns HTTP 404 Not Found and leaves the run in
progress, which fails a job whose tests passed. This tool does not use that
sequence. It creates the run already completed, in one request, and treats a
create or update failure as a warning. A failed, empty, missing, or unreadable
report still fails the process.

``--path`` accepts one glob or a comma-separated list.
"""

import argparse
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET

SUMMARY_LIMIT = 65000
API_ROOT = "https://api.github.com"


class CheckApiError(Exception):
    def __init__(self, status, operation):
        self.status = status
        self.operation = operation
        if status == 404:
            message = f"{operation} a check-run failed with HTTP 404 Not Found"
        elif status:
            message = f"{operation} a check-run failed with HTTP {status}"
        else:
            message = f"{operation} a check-run failed before the API responded"
        super().__init__(message)


class RejectRedirect(HTTPRedirectHandler):
    """Refuse redirects so the job token is not sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHubChecks:
    def __init__(self, token, repository, opener=None):
        owner, repo = repository.split("/", 1)
        self.token = token
        self.owner = owner
        self.repo = repo
        self.opener = opener or build_opener(RejectRedirect()).open

    def create_check_run(self, body):
        path = f"/repos/{self.owner}/{self.repo}/check-runs"
        return self._request("POST", path, body, "create")

    def update_check_run(self, check_id, body):
        path = f"/repos/{self.owner}/{self.repo}/check-runs/{check_id}"
        return self._request("PATCH", path, body, "update")

    def _request(self, method, path, body, operation):
        request = Request(
            API_ROOT + path,
            data=json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "livingproof-sim-mujoco-test-results",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with self.opener(request, timeout=30) as response:
                payload = response.read(1024 * 1024 + 1)
        except HTTPError as error:
            raise CheckApiError(error.code, operation) from error
        except (URLError, TimeoutError, OSError) as error:
            raise CheckApiError(0, operation) from error
        if len(payload) > 1024 * 1024:
            raise CheckApiError(0, operation)
        try:
            parsed = json.loads(payload.decode())
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CheckApiError(0, operation) from error
        if not isinstance(parsed, dict):
            raise CheckApiError(0, operation)
        return parsed


def warning(message):
    return "::warning::" + " ".join(str(message).split())


def client_from_env(env=None):
    env = os.environ if env is None else env
    token = env.get("GITHUB_TOKEN", "")
    repository = env.get("GITHUB_REPOSITORY", "")
    if not token or repository.count("/") != 1:
        return None
    return GitHubChecks(token, repository)


def matching_reports(pattern, root):
    root = Path(root)
    matched = []
    for piece in pattern.split(","):
        piece = piece.strip()
        if not piece:
            continue
        candidate = Path(piece)
        query = piece if candidate.is_absolute() else str(root / piece)
        matched.extend(Path(item) for item in glob.glob(query, recursive=True))
    return sorted({path for path in matched if path.is_file()})


def case_result(case):
    if case.find("failure") is not None or case.find("error") is not None:
        return "FAIL"
    if case.find("skipped") is not None:
        return "SKIPPED"
    return "PASS"


def case_detail(case):
    node = case.find("failure")
    if node is None:
        node = case.find("error")
    if node is None:
        return ""
    text = node.get("message") or node.text or ""
    return " ".join(text.split())[:500]


def case_name(case):
    classname = case.get("classname", "")
    name = case.get("name", "")
    if classname and name:
        return f"{classname}::{name}"
    return name or classname or "(unnamed)"


def load_junit(pattern, root):
    paths = matching_reports(pattern, root)
    report = {"cases": [], "missing": not paths, "malformed": False}
    if not paths:
        return report
    for path in paths:
        try:
            cases = ET.parse(path).findall(".//testcase")
        except (ET.ParseError, OSError):
            report["malformed"] = True
            report["cases"] = []
            return report
        for case in cases:
            report["cases"].append(
                {
                    "name": case_name(case),
                    "result": case_result(case),
                    "detail": case_detail(case),
                }
            )
    return report


def counts(report):
    cases = report["cases"]
    return {
        "passed": sum(case["result"] == "PASS" for case in cases),
        "failed": sum(case["result"] == "FAIL" for case in cases),
        "skipped": sum(case["result"] == "SKIPPED" for case in cases),
    }


def report_ok(report):
    tally = counts(report)
    return (
        not report["missing"]
        and not report["malformed"]
        and report["cases"]
        and tally["failed"] == 0
    )


def check_summary(report):
    tally = counts(report)
    lines = [
        f"{tally['passed']} passed, {tally['failed']} failed, and {tally['skipped']} skipped.",
        "",
        "| Result | Test |",
        "| --- | --- |",
    ]
    for case in report["cases"]:
        lines.append(f"| {case['result']} | `{case['name']}` |")
        if case["detail"]:
            lines.extend(["", "```", case["detail"], "```"])
    text = "\n".join(lines) + "\n"
    if len(text) > SUMMARY_LIMIT:
        text = text[: SUMMARY_LIMIT - 40] + "\n\nReport truncated.\n"
    return text


def _completed_body(*, name, head_sha, conclusion, title, summary, completed_at):
    return {
        "name": name,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": conclusion,
        "completed_at": completed_at,
        "output": {"title": title, "summary": summary},
    }


def _finish_open_run(api, check_id, *, conclusion, title, summary, completed_at):
    """Finish a run the API did not complete. Return (check or None, warnings)."""
    warnings = []
    try:
        finished = api.update_check_run(
            check_id,
            {
                "status": "completed",
                "conclusion": conclusion,
                "completed_at": completed_at,
                "output": {"title": title, "summary": summary},
            },
        )
    except CheckApiError as error:
        warnings.append(
            warning(
                f"Could not update reporting check-run {check_id} ({error}). "
                "Retrying with a conclusion only so it is not left in progress."
            )
        )
        finished = None
    if finished is not None and finished.get("status") == "completed":
        return finished, warnings
    if finished is not None:
        warnings.append(
            warning(
                f"Reporting check-run {check_id} was still in progress after the detailed update. "
                "Retrying with a conclusion only."
            )
        )
    try:
        finished = api.update_check_run(
            check_id,
            {
                "status": "completed",
                "conclusion": conclusion,
                "completed_at": completed_at,
            },
        )
    except CheckApiError as error:
        warnings.append(warning(f"Could not finish reporting check-run {check_id} ({error})."))
        return None, warnings
    if finished.get("status") != "completed":
        warnings.append(
            warning(
                f"Reporting check-run {check_id} was not completed after the conclusion update."
            )
        )
        return None, warnings
    return finished, warnings


def publish_check(api, *, name, head_sha, conclusion, title, summary, completed_at):
    """Create one completed check run. Return (check or None, warnings)."""
    warnings = []
    request_body = _completed_body(
        name=name,
        head_sha=head_sha,
        conclusion=conclusion,
        title=title,
        summary=summary,
        completed_at=completed_at,
    )
    try:
        created = api.create_check_run(request_body)
    except CheckApiError as error:
        warnings.append(
            warning(
                f"Could not create the reporting check-run ({error}). "
                "Results remain in the job summary and the raw JUnit report."
            )
        )
        return None, warnings
    if created.get("status") == "completed":
        return created, warnings
    check_id = created.get("id")
    if not check_id:
        warnings.append(
            warning(
                "The reporting check-run was not completed and its id was missing. "
                "No further update was attempted."
            )
        )
        return None, warnings
    finished, update_warnings = _finish_open_run(
        api,
        check_id,
        conclusion=conclusion,
        title=title,
        summary=summary,
        completed_at=completed_at,
    )
    warnings.extend(update_warnings)
    if finished is not None and finished.get("status") == "completed":
        return finished, warnings
    # Updates to this id failed, for example with HTTP 404 Not Found. A new
    # create is already completed, so this attempt does not add another
    # in-progress run. The earlier id cannot be closed when its update 404s.
    warnings.append(
        warning(
            f"Reporting check-run {check_id} could not be updated and was not left as the "
            "published result. Creating one completed check run so the name is not left "
            "in progress."
        )
    )
    try:
        replacement = api.create_check_run(request_body)
    except CheckApiError as error:
        warnings.append(
            warning(
                f"Could not create a completed replacement check-run ({error}). "
                "Results remain in the job summary and the raw JUnit report."
            )
        )
        return None, warnings
    if replacement.get("status") != "completed":
        warnings.append(
            warning(
                "The replacement check-run was not completed. "
                "Results remain in the job summary and the raw JUnit report."
            )
        )
        return None, warnings
    return replacement, warnings


def evaluate(pattern, *, root, name, head_sha, api, publish, completed_at):
    report = load_junit(pattern, root)
    tally = counts(report)
    exit_code = 0 if report_ok(report) else 1
    warnings = []
    check = None
    readable = not report["missing"] and not report["malformed"] and bool(report["cases"])
    if publish and readable:
        sha_ok = re.fullmatch(r"[0-9a-f]{40}", head_sha or "") is not None
        if api is None or not sha_ok:
            warnings.append(
                warning(
                    "Check-run publication was skipped because the token, repository, "
                    "or commit SHA was missing. Results remain in the job summary "
                    "and the raw JUnit report."
                )
            )
        else:
            conclusion = "success" if tally["failed"] == 0 else "failure"
            title = (
                f"{tally['passed']} passed, {tally['failed']} failed, {tally['skipped']} skipped"
            )
            check, warnings = publish_check(
                api,
                name=name,
                head_sha=head_sha,
                conclusion=conclusion,
                title=title,
                summary=check_summary(report),
                completed_at=completed_at,
            )
    summary = gate_summary(report, tally)
    if warnings:
        summary += "\nCheck publication:\n" + "\n".join(warnings) + "\n"
    return {
        "exit_code": exit_code,
        "warnings": warnings,
        "check": check,
        "passed": tally["passed"],
        "failed": tally["failed"],
        "skipped": tally["skipped"],
        "summary": summary,
    }


def gate_summary(report, tally):
    if report["malformed"]:
        return "\nJUnit gate: the report could not be read, so the job fails.\n"
    if report["missing"] or not report["cases"]:
        return "\nJUnit gate: no test cases were found, so the job fails.\n"
    return (
        f"\nJUnit gate: passed {tally['passed']}, failed {tally['failed']}, "
        f"skipped {tally['skipped']}.\n"
    )


def append_text(path, text):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail the process on JUnit failures and publish one completed check run."
    )
    parser.add_argument("--name", required=True, help="Check run name, also used in the summary")
    parser.add_argument(
        "--path",
        required=True,
        help="JUnit file, glob, or comma-separated list, relative to --root",
    )
    parser.add_argument("--publish-check", choices=("true", "false"), default="true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--completed-at",
        default=None,
        help="ISO-8601 timestamp stored on the check run",
    )
    args = parser.parse_args(argv)
    completed_at = args.completed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    publish = args.publish_check == "true"
    sha = os.environ.get("HEAD_SHA") or os.environ.get("GITHUB_SHA", "")
    result = evaluate(
        args.path,
        root=args.root,
        name=args.name,
        head_sha=sha,
        api=client_from_env() if publish else None,
        publish=publish,
        completed_at=completed_at,
    )
    for line in result["warnings"]:
        print(line)
    append_text(os.environ.get("GITHUB_STEP_SUMMARY"), result["summary"])
    url = ""
    if result["check"] and result["check"].get("html_url"):
        url = result["check"]["html_url"]
    append_text(
        os.environ.get("GITHUB_OUTPUT"),
        f"passed={result['passed']}\nfailed={result['failed']}\n"
        f"skipped={result['skipped']}\nurl_html={url}\n",
    )
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
