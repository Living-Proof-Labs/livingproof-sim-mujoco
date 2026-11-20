# SPDX-License-Identifier: Apache-2.0
"""Verification (Vera) of what a publish hook may return (#107): every non-iterable is a ValueError.

The PR's own test covers a bare Message only. These cover the other values the docs say are refused
(``str``, ``bytes``, ``dict``) and plain non-iterables (``int``, ``float``, ``bool``, ``object``),
and that a tuple or generator of Messages is accepted, on both trial paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_robot_topics_verification import PATHS, _msg, _run

from livingproof_sim_mujoco import robot_topics


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "value", ["/x", b"/x", {"topic": "/x"}, 5, 1.5, True, object()],
    ids=["str", "bytes", "dict", "int", "float", "bool", "object"],
)  # fmt: skip
def test_a_hook_returning_anything_but_an_iterable_of_messages_is_a_value_error(
    path, value, tmp_path, monkeypatch
):
    with pytest.raises(Exception) as caught:
        _run(path, tmp_path, monkeypatch, lambda n, m, s: value)
    assert caught.type is ValueError, f"{caught.type.__name__}: {caught.value}"
    assert not list(tmp_path.rglob("*.mcap"))


@pytest.mark.parametrize("value", [None, [], (), iter(())], ids=["none", "list", "tuple", "iter"])
def test_nothing_is_accepted_as_no_messages(value):
    assert robot_topics.items(value) == []


def test_a_tuple_or_generator_of_messages_is_accepted():
    m = _msg()
    assert robot_topics.items((m, m)) == [m, m]
    assert robot_topics.items(x for x in (m,)) == [m]
