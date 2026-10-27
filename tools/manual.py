# SPDX-License-Identifier: Apache-2.0
"""Render canonical Markdown, installed command help and actual scenario outcomes."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/manual"


def run(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "source"
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(ROOT / "docs", source)
    wheel_dir = OUT / "wheel"
    if wheel_dir.exists():
        shutil.rmtree(wheel_dir)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheel_dir)], check=True)
    (wheel,) = wheel_dir.glob("*.whl")
    product = OUT / "product"
    if product.exists():
        shutil.rmtree(product)
    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--target", str(product), str(wheel)], check=True
    )
    env = dict(os.environ, PYTHONPATH=str(product))
    command = [sys.executable, "-m", "livingproof_sim_mujoco.cli"]
    origin = run(
        sys.executable,
        "-c",
        "import livingproof_sim_mujoco; print(livingproof_sim_mujoco.__file__)",
        cwd=OUT,
        env=env,
    )
    assert Path(origin).is_relative_to(product), origin
    identity = json.loads(run(*command, "--version", cwd=OUT, env=env))
    assert identity["provenance"] == "recorded-build", identity
    help_text = run(*command, "--help", cwd=OUT, env=env)
    # The module invocation uses cli.py as argv[0]; argparse has an explicit prog.
    committed = (source / "command.md").read_text()
    assert help_text in committed, "Run task docs:reference to refresh CLI help"
    (source / "command.md").write_text(committed)
    task_lines = [
        "# Contributor tasks",
        "",
        "Host wrappers enter Docker; prepared-container commands use Taskfile.runtime.yml.",
        "",
    ]
    for taskfile, label in (
        ("Taskfile.yml", "Host"),
        ("Taskfile.runtime.yml", "Prepared container"),
    ):
        listing = json.loads(run("task", "-t", taskfile, "--list", "--json", cwd=ROOT))
        task_lines += ["## " + label, ""]
        for item in listing["tasks"]:
            summary = run("task", "-t", taskfile, item["name"], "--summary", cwd=ROOT)
            task_lines += ["### " + item["name"], "", "```text", summary, "```", ""]
    (source / "tasks.md").write_text("\n".join(task_lines))
    revision = run("git", "rev-parse", "HEAD", cwd=ROOT)
    dirty = bool(run("git", "status", "--porcelain", cwd=ROOT))
    ci_run = os.environ.get("GITHUB_RUN_ID")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        required = ("GITHUB_RUN_ID", "GITHUB_REPOSITORY", "GITHUB_SERVER_URL")
        if any(not os.environ.get(name) for name in required):
            raise SystemExit("Hosted documentation requires complete GitHub run context")
    run_url = (
        f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{ci_run}"
        if ci_run
        else "local; hosted publication unverified"
    )
    provenance = dict(
        revision=revision,
        dirty=dirty,
        identity=identity,
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_hashes={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT / "docs").glob("*.md")
        },
        ci_run=run_url,
    )
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (source / "context.md").write_text(
        "# Build context\n\n"
        f"[Docs: development · {revision[:7]}]"
        f"(https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/tree/{revision})\n\n"
        "Development revision `"
        + revision
        + "`; dirty checkout: `"
        + str(dirty).lower()
        + "`.\n\n"
        "CI run: " + provenance["ci_run"] + "\n\n"
        "Installed wheel SHA256: `"
        + provenance["wheel_sha256"]
        + "`.\n\n```json\n"
        + json.dumps(identity, indent=2)
        + "\n```\n"
    )
    report = ROOT / "artifacts/tests.xml"
    if report.exists():
        context = json.loads((ROOT / "artifacts/tests-context.json").read_text())
        for name, digest in context["sources"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, (
                "Stale CLI results; rerun task test",
                name,
            )
        shutil.copy(report, OUT / "tests.xml")
        shutil.copy(ROOT / "artifacts/tests-context.json", OUT / "tests-context.json")
    cases = ET.parse(report).findall(".//testcase") if report.exists() else []
    if report.exists():
        assert cases, "Empty CLI report"
        assert sorted(c.attrib["name"] for c in cases) == sorted(context["cases"]), (
            "Incomplete CLI report; rerun task test"
        )
    lines = [
        "# Feature scenarios",
        "",
        "Bounded CLI inventory; these are software/engine observations, not robot capability.",
        "No line coverage measured. Unlisted behavior is unspecified by this inventory.",
        "Results below come from `artifacts/tests.xml`; see [build context](context.md).",
        "",
    ]
    for node in ast.parse((ROOT / "tests/test_cli.py").read_text()).body:
        if not isinstance(node, ast.FunctionDef) or not ast.get_docstring(node):
            continue
        lines += [
            "## " + node.name,
            "",
            ast.get_docstring(node),
            "",
            "| Executed case | Outcome |",
            "| --- | --- |",
        ]
        matches = [c for c in cases if c.attrib["name"].split("[")[0] == node.name]
        if not matches:
            lines.append("| No result | UNTESTED |")
        for c in matches:
            status = (
                "FAILED"
                if c.find("failure") is not None or c.find("error") is not None
                else "SKIPPED"
                if c.find("skipped") is not None
                else "PASSED"
            )
            lines.append(f"| `{c.attrib['name']}` | {status} |")
        lines.append("")
    (source / "features.md").write_text("\n".join(lines))
    subprocess.run(["zensical", "build", "--clean", "-f", "mkdocs.yml"], cwd=ROOT, check=True)
    shutil.copy(OUT / "provenance.json", OUT / "html/provenance.json")


if __name__ == "__main__":
    main()
