# SPDX-License-Identifier: Apache-2.0
"""Parity tests clone into a per-run temporary directory, never a shared path in /tmp.

Two concurrent test runs on one host used to race on ``rm -rf`` and ``git clone`` in
``/tmp/livingproof-world-parity`` (#117).
"""

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent
CLONING = ("test_world_parity.py", "test_compose.py", "test_lock_cli_review.py")
SHARED = re.compile(
    r"""["']/tmp/livingproof-world-parity|Path\(["']/tmp["']\)\s*/\s*["']livingproof"""
)
FACTORY = re.compile(r"tmp_path_factory\.mktemp\(")


def test_no_test_writes_clones_to_a_shared_tmp_path():
    shared = [p.name for p in sorted(TESTS.glob("test_*.py")) if SHARED.search(p.read_text())]
    assert shared == []


def test_cloning_tests_take_their_cache_from_tmp_path_factory():
    missing = [name for name in CLONING if not FACTORY.search((TESTS / name).read_text())]
    assert missing == []
