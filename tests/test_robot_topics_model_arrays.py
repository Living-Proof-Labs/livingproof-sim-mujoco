# SPDX-License-Identifier: Apache-2.0
"""Verification (Vera) of the publish hook's model copy (#107, decision (b)): arrays too.

``copy.copy`` is a shallow copy for most Python objects. These tests show the hook's copy of the
MuJoCo model shares no array memory with the model the trial runs on, so writing a numpy array in
place through the copy (masses, geometry, friction, damping, options) cannot reach the live physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_robot_topics_verification import PATHS, _msg, _run

from livingproof_sim_mujoco import robot_topics

XML = """<mujoco><worldbody><body name="b"><freejoint/><geom size=".1" mass="1"/></body></worldbody></mujoco>"""
ARRAYS = ["body_mass", "geom_size", "geom_friction", "dof_damping", "geom_pos"]


class _Robot:
    seen = None

    def publish(self, name, model, step):
        _Robot.seen = model
        return []


@pytest.mark.parametrize("field", ARRAYS)
def test_the_hooks_copy_shares_no_array_memory_with_the_live_model(field):
    live = mujoco.MjModel.from_xml_string(XML)
    robot_topics.publisher(_Robot(), "x", live)(None)
    mine = _Robot.seen
    assert mine is not live
    assert not np.shares_memory(getattr(mine, field), getattr(live, field))
    before = getattr(live, field).copy()
    getattr(mine, field)[...] = 7.0
    mine.opt.gravity[2] = -1.0
    assert np.array_equal(getattr(live, field), before)
    assert live.opt.gravity[2] == pytest.approx(-9.81)


def _outcome(doc, root):
    trial = doc["trials"][0]
    return trial["observation_sha256"], (root / "m" / trial["mcap"]["path"]).read_bytes()


@pytest.mark.parametrize("path", PATHS)
def test_writing_model_arrays_in_place_inside_publish_does_not_change_the_trial(
    path, tmp_path, monkeypatch
):
    def meddling(name, model, step):
        model.body_mass[:] *= 10
        model.geom_friction[:] = 0.0
        model.dof_damping[:] += 50
        model.geom_size[:] *= 2
        model.opt.gravity[:] = 0
        return [_msg(payload={"t": step.time})]

    def polite(name, model, step):
        return [_msg(payload={"t": step.time})]

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    meddled = _run(path, tmp_path / "a", monkeypatch, meddling)
    plain = _run(path, tmp_path / "b", monkeypatch, polite)
    assert _outcome(meddled, tmp_path / "a") == _outcome(plain, tmp_path / "b")
