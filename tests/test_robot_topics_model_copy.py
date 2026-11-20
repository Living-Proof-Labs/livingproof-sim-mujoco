# SPDX-License-Identifier: Apache-2.0
"""The publish hook gets a copy of the model, made once per trial (#107, Rivet's decision).

So an adapter that changes the model inside ``publish`` cannot change the trial: its topics, the
observation hash and the media bytes are those of a run whose hook left the model alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_robot_topics_verification import PATHS, _msg, _run


def _outcome(doc, root):
    trial = doc["trials"][0]
    return trial["observation_sha256"], (root / "m" / trial["mcap"]["path"]).read_bytes()


@pytest.mark.parametrize("path", PATHS)
def test_changing_the_model_inside_publish_does_not_change_the_trial(path, tmp_path, monkeypatch):
    def meddling(name, model, step):
        model.opt.gravity[2] = -1.0
        model.opt.timestep = model.opt.timestep * 2
        return [_msg(payload={"t": step.time})]

    def polite(name, model, step):
        return [_msg(payload={"t": step.time})]

    (tmp_path / "meddled").mkdir()
    (tmp_path / "plain").mkdir()
    meddled = _run(path, tmp_path / "meddled", monkeypatch, meddling)  # must not raise
    plain = _run(path, tmp_path / "plain", monkeypatch, polite)
    assert _outcome(meddled, tmp_path / "meddled") == _outcome(plain, tmp_path / "plain")


@pytest.mark.parametrize("path", PATHS)
def test_one_copy_of_the_model_per_trial_not_per_step(path, tmp_path, monkeypatch):
    seen = []

    def hook(name, model, step):
        seen.append(id(model))
        return []

    doc = _run(path, tmp_path, monkeypatch, hook)
    assert len(seen) > len(doc["trials"])  # many steps...
    assert len(set(seen)) == len(doc["trials"])  # ...one model object per trial
