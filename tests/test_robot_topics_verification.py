# SPDX-License-Identifier: Apache-2.0
"""Verification (Vera) of the adapter publish hook (#107): paths the PR's own tests leave open.

Each case runs the real media re-run on both trial paths (walk-hallway constant command and
walk-corner drive())."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import test_robot_topics as base
from test_robot_topics import PubCorner, PubToy, _by_topic, _read

from livingproof_sim_mujoco import media, robot_topics

PATHS = ["hallway", "corner"]


def _run(path, tmp_path, monkeypatch, hook):
    """Media re-run of one trial on `path` with an adapter whose publish is `hook`."""
    if path == "hallway":

        class A(PubToy):
            publish = staticmethod(hook)

        monkeypatch.setattr(base, "PUB", A())
        return base._hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")

    class C(PubCorner):
        publish = staticmethod(hook)

    monkeypatch.setattr(base, "PUBCORNER", C())
    return base._corner(tmp_path, monkeypatch, "test_robot_topics:PUBCORNER", C)


def _msg(topic="/x", name="s.S", schema=None, payload=None):
    return robot_topics.Message(topic, name, schema or {"type": "object"}, payload or {})


@pytest.mark.parametrize("path", PATHS)
def test_samples_are_exactly_50_hz_in_sim_time_and_sorted(path, tmp_path, monkeypatch):
    doc = _run(path, tmp_path, monkeypatch, lambda n, m, s: [_msg(payload={"t": s.time})])
    messages = _read(tmp_path / "m" / doc["trials"][0]["mcap"]["path"])
    topics = _by_topic(messages)
    stamps = [m.log_time for _, m in topics["/x"]]
    assert stamps == [m.log_time for _, m in topics["/base_pose"]]
    assert set(np.diff(stamps)) == {20_000_000}  # 50 Hz, no gaps or duplicates
    for _, m in topics["/x"]:
        body = json.loads(m.data)
        assert body["t"] == pytest.approx(m.log_time / 1e9, abs=1e-9)
        assert body["timestamp"] == {"sec": m.log_time // 10**9, "nsec": m.log_time % 10**9}
    times = [m.log_time for _, _, m in messages]
    assert times == sorted(times)


@pytest.mark.parametrize("path", PATHS)
def test_several_messages_in_one_step_keep_the_order_the_hook_gave(path, tmp_path, monkeypatch):
    doc = _run(
        path,
        tmp_path,
        monkeypatch,
        lambda n, m, s: [_msg("/a", payload={"k": 1}), _msg("/b", payload={"k": 2}),
                         _msg("/a", payload={"k": 3})],
    )  # fmt: skip
    messages = _read(tmp_path / "m" / doc["trials"][0]["mcap"]["path"])
    seq = [(c.topic, json.loads(m.data).get("k")) for _, c, m in messages if c.topic in "/a/b"]
    assert seq[:3] == [("/a", 1), ("/b", 2), ("/a", 3)]


@pytest.mark.parametrize("path", PATHS)
def test_a_hook_that_writes_to_the_snapshot_cannot_change_the_trial(path, tmp_path, monkeypatch):
    refused = []

    def hook(name, model, step):
        for array in (step.xpos, step.xquat, step.qpos, step.qvel):
            with pytest.raises(ValueError):
                array[...] = 1e6
            array.flags.writeable = True  # even re-enabling writes touches only the copy
            array[...] = 1e6
        with pytest.raises(AttributeError):
            step.time = 1e6  # frozen
        refused.append(1)
        return []

    doc = _run(path, tmp_path, monkeypatch, hook)  # media() raises if the hash changed
    assert refused and doc["trials"][0]["observation_sha256"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    ("returned", "error"),
    [
        (lambda: [{"topic": "/x"}], ValueError),
        (lambda: ["/x"], ValueError),
        (lambda: [_msg("/base_pose")], ValueError),
        (lambda: [_msg("/joint_states")], ValueError),
        (lambda: [_msg("/wall_contacts")], ValueError),
        (lambda: [_msg("x")], ValueError),
        (lambda: [_msg("/")], ValueError),
        (lambda: [_msg("")], ValueError),
        (lambda: [_msg(payload={"v": float("inf")})], ValueError),
    ],
)
def test_a_bad_item_stops_the_run_and_leaves_no_recording(
    path, returned, error, tmp_path, monkeypatch
):
    with pytest.raises(error):
        _run(path, tmp_path, monkeypatch, lambda n, m, s: returned())
    assert not list(tmp_path.rglob("*.mcap"))
    assert not list((tmp_path / "m").rglob("*")) if (tmp_path / "m").exists() else True


@pytest.mark.parametrize("path", PATHS)
def test_a_hook_that_raises_stops_the_run_and_leaves_no_recording(path, tmp_path, monkeypatch):
    def hook(name, model, step):
        if step.time > 0.1:
            raise RuntimeError("adapter bug")
        return []

    with pytest.raises(RuntimeError, match="adapter bug"):
        _run(path, tmp_path, monkeypatch, hook)
    assert not list(tmp_path.rglob("*.mcap"))
    assert not list(tmp_path.rglob("media.json"))


@pytest.mark.parametrize("path", PATHS)
def test_a_hook_that_returns_a_single_message_is_refused_with_a_clear_error(
    path, tmp_path, monkeypatch
):
    with pytest.raises((ValueError, TypeError)) as caught:
        _run(path, tmp_path, monkeypatch, lambda n, m, s: _msg())
    assert caught.type is ValueError, f"{caught.type.__name__}: {caught.value}"


@pytest.mark.parametrize("path", PATHS)
def test_two_identical_runs_give_identical_bytes_with_a_hook(path, tmp_path, monkeypatch):
    out = []
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        doc = _run(path, tmp_path / run, monkeypatch, lambda n, m, s: [_msg(payload={"t": s.time})])
        out.append((tmp_path / run / "m" / doc["trials"][0]["mcap"]["path"]).read_bytes())
    assert out[0] == out[1]


@pytest.mark.parametrize("path", PATHS)
def test_the_hooked_recording_passes_the_mcap_cli_doctor(path, tmp_path, monkeypatch):
    cli = shutil.which("mcap")
    if cli is None:
        # The development image carries the pinned CLI (Dockerfile), so there it must be found.
        assert not os.environ.get("LIVINGPROOF_TOOLING"), "the mcap CLI is missing from the image"
        pytest.skip("the mcap CLI is not installed")
    doc = _run(path, tmp_path, monkeypatch, lambda n, m, s: [_msg(payload={"t": s.time})])
    mcap = tmp_path / "m" / doc["trials"][0]["mcap"]["path"]
    done = subprocess.run([cli, "doctor", str(mcap)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_robot_topics_and_wall_contacts_interleave_in_log_time_order(tmp_path, monkeypatch):
    """Contacts are queued after the run; the hook's messages must still sort among them."""
    import test_media_binding as touch
    from mcap.reader import make_reader

    class Touch(touch.ToyTouch):
        def publish(self, name, model, step):
            return [_msg(payload={"t": step.time})]

    repo = tmp_path / "repo"
    repo.mkdir()
    touch._repo(repo)
    monkeypatch.setattr(Touch, "root", repo)
    monkeypatch.setattr(touch, "TOUCH", Touch())
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    hooked = tmp_path / "hooked.mcap"
    world = json.loads(
        (Path(media.ev.hallway.WORLDS) / "hallway-open" / "0.1.0" / "world.json").read_text()
    )
    p = {"spawn": "spawn-entrance", "time_limit": 3.0, "exit_x": 0.5, "command_speed": 0.5}
    obs = media.ev.run_trial(
        "test_media_binding:TOUCH",
        "stock",
        {},
        p,
        world,
        on_step=lambda m, b: media.Recorder(
            m, b, mcap_path=hooked, publish=robot_topics.publisher(touch.TOUCH, "stock", m)
        ),
    )
    assert obs["wall_contacts"]
    with hooked.open("rb") as f:
        found = list(make_reader(f).iter_messages(log_time_order=False))
    times = [m.log_time for _, _, m in found]
    assert {"/x", "/wall_contacts"} <= {c.topic for _, c, _ in found}
    assert times == sorted(times)
