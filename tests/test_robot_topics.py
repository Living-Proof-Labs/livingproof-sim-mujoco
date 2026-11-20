# SPDX-License-Identifier: Apache-2.0
"""An adapter publishes its own topics into the trial MCAP through a read-only hook (#107).

The hook is ``publish(name, model, step) -> messages`` on the adapter, called by the recorder at
its sample steps on every trial path (the constant command of walk-hallway and the ``drive()`` of
walk-corner). The adapter never writes the MCAP and is handed a read-only snapshot, not ``data``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from mcap.reader import make_reader

sys.path.insert(0, str(Path(__file__).parent))

import test_evidence  # noqa: E402
import test_walk_corner  # noqa: E402
from test_evidence import Toy, _lock, _repo  # noqa: E402
from test_walk_corner import Corner  # noqa: E402

from livingproof_sim_mujoco import evidence, media, robot_topics  # noqa: E402

POSE = {
    "type": "object",
    "title": "foxglove.PoseInFrame",
    "properties": {
        "timestamp": {"type": "object"},
        "frame_id": {"type": "string"},
        "pose": {"type": "object"},
    },
    "required": ["timestamp", "frame_id", "pose"],
}
SIM = {"/base_pose", "/joint_states", "/wall_contacts"}


def _estimate(name, model, step):
    x, y, z = (float(v) for v in step.xpos[model.body(name).id])
    w, qx, qy, qz = (float(v) for v in step.xquat[model.body(name).id])
    return [
        robot_topics.Message(
            topic="/estimated_pose",
            schema_name="foxglove.PoseInFrame",
            schema=POSE,
            payload={
                "frame_id": "world",
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {"x": qx, "y": qy, "z": qz, "w": w},
                },
            },
        )
    ]


class PubToy(Toy):
    id = "pub-toy"

    def publish(self, name, model, step):
        return _estimate("base", model, step)


class PubCorner(Corner):
    id = "pub-corner"

    def publish(self, name, model, step):
        return _estimate("base", model, step)


PUB, PUBCORNER = PubToy(), PubCorner()


def _verification(tmp_path):
    v = tmp_path / "verification.json"
    v.write_text(json.dumps({"trials": [{"index": 1, "status": "PASS"}]}))
    return v


def _hallway(tmp_path, monkeypatch, ref):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    evidence.evidence(ref, "stock", tmp_path / "ev", trials=1, lock=_lock(repo))
    return media.media(ref, "stock", tmp_path / "ev", _verification(tmp_path), tmp_path / "m")


def _corner(tmp_path, monkeypatch, ref, cls):
    repo = test_walk_corner._repo(tmp_path, monkeypatch, cls=cls)
    evidence.evidence(ref, "stock", tmp_path / "ev", trials=1, lock=repo / "caller.lock.json")
    return media.media(ref, "stock", tmp_path / "ev", _verification(tmp_path), tmp_path / "m")


def _read(path: Path):
    with path.open("rb") as f:
        reader = make_reader(f)
        return [(s, c, m) for s, c, m in reader.iter_messages(log_time_order=False)]


def _by_topic(messages):
    out: dict[str, list] = {}
    for schema, channel, message in messages:
        out.setdefault(channel.topic, []).append((schema, message))
    return out


@pytest.fixture(params=["hallway", "corner"])
def published(request, tmp_path, monkeypatch):
    if request.param == "hallway":
        doc = _hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")
    else:
        doc = _corner(tmp_path, monkeypatch, "test_robot_topics:PUBCORNER", PubCorner)
    return request.param, tmp_path, doc


def test_the_published_topic_is_recorded_on_every_trial_path(published):
    _, tmp_path, doc = published
    topics = _by_topic(_read(tmp_path / "m" / doc["trials"][0]["mcap"]["path"]))
    assert "/estimated_pose" in topics and SIM <= set(topics) | {"/wall_contacts"}
    assert len(topics["/estimated_pose"]) == len(topics["/base_pose"]) > 10  # the recorder's 50 Hz


def test_each_message_is_stamped_with_simulation_time_in_the_world_frame(published):
    _, tmp_path, doc = published
    topics = _by_topic(_read(tmp_path / "m" / doc["trials"][0]["mcap"]["path"]))
    base, estimated = topics["/base_pose"], topics["/estimated_pose"]
    assert [m.log_time for _, m in estimated] == [m.log_time for _, m in base]
    for (_, message), (_, truth) in zip(estimated, base, strict=True):
        body, reference = json.loads(message.data), json.loads(truth.data)
        assert body["frame_id"] == "world"
        assert body["timestamp"] == reference["timestamp"]
        for axis in "xyz":  # the same pose the sim records, to the recorder's rounding
            got, want = body["pose"]["position"][axis], reference["pose"]["position"][axis]
            assert got == pytest.approx(want, abs=1e-5)


def test_the_channel_declares_the_schema_the_adapter_gave_and_messages_satisfy_it(published):
    _, tmp_path, doc = published
    topics = _by_topic(_read(tmp_path / "m" / doc["trials"][0]["mcap"]["path"]))
    schema, _ = topics["/estimated_pose"][0]
    assert (schema.name, schema.encoding) == ("foxglove.PoseInFrame", "jsonschema")
    assert json.loads(schema.data) == POSE
    validator = Draft202012Validator(json.loads(schema.data))
    for _, message in topics["/estimated_pose"]:
        assert list(validator.iter_errors(json.loads(message.data))) == []


def test_publishing_does_not_change_the_recording_of_the_sim_topics_or_the_hash(
    tmp_path, monkeypatch
):
    """media() refuses a recording whose re-run observation differs from the evidence, so
    reaching the end already proves the hash. The sim's own topics are also byte-identical."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_hook = _hallway(tmp_path / "a", monkeypatch, "test_robot_topics:PUB")
    without = _hallway(tmp_path / "b", monkeypatch, "test_evidence:TOY")
    assert (
        with_hook["trials"][0]["observation_sha256"] == without["trials"][0]["observation_sha256"]
    )
    one = _by_topic(_read(tmp_path / "a" / "m" / with_hook["trials"][0]["mcap"]["path"]))
    two = _by_topic(_read(tmp_path / "b" / "m" / without["trials"][0]["mcap"]["path"]))
    for topic in SIM:
        assert [m.data for _, m in one.get(topic, [])] == [m.data for _, m in two.get(topic, [])]
        assert [m.log_time for _, m in one.get(topic, [])] == [
            m.log_time for _, m in two.get(topic, [])
        ]


def test_an_adapter_with_no_hook_records_only_todays_topics(tmp_path, monkeypatch):
    doc = _hallway(tmp_path, monkeypatch, "test_evidence:TOY")
    topics = _by_topic(_read(tmp_path / "m" / doc["trials"][0]["mcap"]["path"]))
    assert set(topics) <= SIM and {"/base_pose", "/joint_states"} <= set(topics)
    assert not hasattr(test_evidence.TOY, "publish")


def test_the_mcap_is_in_log_time_order_and_mcap_doctor_is_clean(published):
    _, tmp_path, doc = published
    path = tmp_path / "m" / doc["trials"][0]["mcap"]["path"]
    times = [m.log_time for _, _, m in _read(path)]
    assert times == sorted(times)
    cli = shutil.which("mcap")
    if cli is None:
        pytest.skip("the mcap CLI is not installed")
    done = subprocess.run([cli, "doctor", str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_step_the_adapter_receives_is_a_read_only_snapshot(tmp_path, monkeypatch):
    seen = {}

    class Spy(PubToy):
        def publish(self, name, model, step):
            seen["step"] = step
            return []

    monkeypatch.setattr(sys.modules[__name__], "PUB", Spy())
    _hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")
    step = seen["step"]
    assert isinstance(step, robot_topics.StepView)
    for array in (step.xpos, step.xquat, step.qpos, step.qvel):
        assert isinstance(array, np.ndarray) and not array.flags.writeable
    with pytest.raises(ValueError):
        step.xpos[0, 0] = 99.0


@pytest.mark.parametrize(
    ("message", "why"),
    [
        (dict(topic="/base_pose"), "reserved"),
        (dict(topic="/joint_states"), "reserved"),
        (dict(topic="/wall_contacts"), "reserved"),
        (dict(topic="estimated_pose"), "start with"),
        (dict(topic="/x", payload={"n": float("nan")}), "JSON"),
    ],
)
def test_a_message_that_would_corrupt_the_recording_is_refused(tmp_path, message, why):
    fields = dict(topic="/x", schema_name="s.S", schema={"type": "object"}, payload={})
    fields.update(message)
    with pytest.raises(ValueError, match=why):
        robot_topics.check(robot_topics.Message(**fields))


def test_one_topic_cannot_change_its_schema_mid_trial(tmp_path, monkeypatch):
    class Flip(PubToy):
        n = 0

        def publish(self, name, model, step):
            Flip.n += 1
            schema = {"type": "object", "title": "a.A" if Flip.n % 2 else "b.B"}
            return [robot_topics.Message("/x", schema["title"], schema, {})]

    monkeypatch.setattr(sys.modules[__name__], "PUB", Flip())
    with pytest.raises(ValueError, match="schema"):
        _hallway(tmp_path, monkeypatch, "test_robot_topics:PUB")


def test_the_guide_documents_the_hook():
    text = " ".join((Path(__file__).parents[1] / "docs" / "evidence.md").read_text().split())
    for needle in (
        "`publish(name, model, step)`",
        "`robot_topics.Message`",
        "read-only snapshot",
        "no-op by default",
        "`/base_pose`, `/joint_states` and `/wall_contacts` are reserved",
        "not recorded in the binding",
    ):
        assert needle in text, needle
