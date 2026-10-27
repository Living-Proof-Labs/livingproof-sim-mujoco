# SPDX-License-Identifier: Apache-2.0
"""Bind CLI results to the exact source inputs, so stale reports fail manual builds."""

import hashlib
import json
from pathlib import Path


def pytest_sessionfinish(session):
    report = session.config.getoption("xmlpath", default=None)
    if report and Path(report).name == "tests.xml":
        paths = sorted(Path("src").rglob("*.py")) + sorted(Path("src").rglob("*.xml"))
        paths += [Path("tests/test_cli.py"), Path("uv.lock")]
        Path("artifacts/tests-context.json").write_text(
            json.dumps(
                {
                    "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    "cases": [item.name for item in session.items],
                },
                indent=2,
            )
            + "\n"
        )
