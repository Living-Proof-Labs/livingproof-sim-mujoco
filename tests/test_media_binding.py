"""The MCAP contacts channel and the producer and per-trial observation bindings (#28).

Built on livingproof-evaluator v1.6.0 (vendored unchanged under
tests/fixtures/livingproof-evaluator-v1.6.0, checked by sha256): its binding schema and its
hallway vocabulary. ``body_contact`` is contact with a named surface; the evaluator's
``wall_contacts`` metric is the list of body_contact events with hallway walls, each
``{wall, start, end, depth}``, and depth is the intrusion past the wall's inner face (the
simulator's own wording: not a penetration depth).
"""

import copy
import hashlib
import json
import os
from pathlib import Path

# Before livingproof_sim_mujoco, which imports MuJoCo and locks the GL platform.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from livingproof_sim_mujoco import binding, evidence, media

pytest.importorskip("mcap")
pytest.importorskip("imageio_ffmpeg")

import test_evidence  # noqa: E402
from test_evidence import CONF, Toy, _lock, _repo  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "livingproof-evaluator-v1.6.0"
PROVENANCE = json.loads((FIXTURES / "PROVENANCE.json").read_text())
EXPECTED_CONTACT = {
    "topic": "/wall_contacts",
    "encoding": "json",
    "schema": "livingproof.WallContact",
    "fields": {"wall": "wall", "start": "start", "end": "end", "depth": "depth"},
    "frame": "world",
}


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _validator():
    schemas = [_load("binding.schema.json"), _load("defs.schema.json")]
    registry = Registry().with_resources([(s["$id"], Resource.from_contents(s)) for s in schemas])
    return Draft202012Validator(schemas[0], registry=registry)


def _valid(document):
    errors = sorted(e.message for e in _validator().iter_errors(document))
    assert errors == []


def test_the_vendored_evaluator_files_are_the_v1_6_0_ones():
    assert PROVENANCE["tag"] == "v1.6.0"
    assert PROVENANCE["commit"] == "ff8c4e77460dae31965335053bd112ad7bde003e"
    for name, entry in PROVENANCE["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == entry["sha256"], name


def test_the_terms_are_the_vocabularys_in_its_order():
    assert binding.TERMS == [t["id"] for t in _load("vocabulary.json")["terms"]]
    assert binding.TERMS == ["base", "base_height", "body_contact"]


def test_the_producer_binding_validates_and_adds_only_the_contacts_channel():
    producer = binding.producer_binding()
    _valid(producer)
    evaluators = _load("livingproof-sim-mujoco.json")
    assert producer["producer"] == evaluators["producer"] == {"id": "livingproof-sim-mujoco"}
    # The evaluator's example producer already names the pose channels; the sim adds contacts.
    for term, entry in evaluators["observables"].items():
        assert producer["observables"][term] == entry, term
    assert producer["observables"]["body_contact"] == EXPECTED_CONTACT
    assert list(producer["observables"]) == binding.TERMS


def test_the_contacts_channel_wording_is_the_vocabularys():
    fields = binding.producer_binding()["observables"]["body_contact"]["fields"]
    assert list(fields) == ["wall", "start", "end", "depth"]
    assert [m["id"] for m in _load("vocabulary.json")["terms"][2]["metrics"]] == ["wall_contacts"]
    schema = media.CONTACT_SCHEMA
    assert schema["title"] == "livingproof.WallContact"
    assert list(schema["properties"]) == ["timestamp", "wall", "start", "end", "depth"]
    depth = schema["properties"]["depth"]["description"]
    assert "intrusion" in depth and "inner face" in depth and "not a penetration depth" in depth
    assert "signed distance" not in depth


class ToyTouch(Toy):
    """A robot that starts inside the left wall's face, so the trial has a contact."""

    def scene(self, name):
        xml = test_evidence.XML.replace('pos="-0.5 0 0.3"', 'pos="-0.5 0.78 0.3"')
        scene = test_evidence.Scene()
        scene.model = mujoco.MjModel.from_xml_string(xml)
        return scene


def _nominal_conf(monkeypatch, with_nominal):
    conf = copy.deepcopy(CONF)
    if with_nominal:
        conf["locomotion"]["nominal_base_height"] = {"value": 0.3, "unit": "m"}
    monkeypatch.setattr(Toy, "configuration", lambda self, name: conf)
    return conf


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return tmp_path, repo


def _media(tmp_path, repo):
    evidence.evidence("test_evidence:TOY", "stock", tmp_path / "ev", trials=2, lock=_lock(repo))
    v = tmp_path / "verification.json"
    v.write_text(
        json.dumps({"trials": [{"index": 1, "status": "PASS"}, {"index": 2, "status": "FAIL"}]})
    )
    return media.media("test_evidence:TOY", "stock", tmp_path / "ev", v, tmp_path / "m")


def _topics(path):
    from mcap.reader import make_reader

    with path.open("rb") as f:
        summary = make_reader(f).get_summary()
        return {c.topic: c.id for c in summary.channels.values()}


def test_every_mcap_declares_the_contacts_channel_even_with_no_contact(recorded):
    tmp_path, repo = recorded
    doc = _media(tmp_path, repo)
    for trial in doc["trials"]:
        topics = _topics(tmp_path / "m" / trial["mcap"]["path"])
        assert set(topics) == {"/base_pose", "/joint_states", "/wall_contacts"}


def test_a_trial_with_a_contact_writes_each_event_as_one_message(tmp_path, monkeypatch):
    from mcap.reader import make_reader

    repo = tmp_path / "repo"
    repo.mkdir()
    _repo(repo)
    monkeypatch.setattr(ToyTouch, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    path = tmp_path / "t.mcap"
    world = json.loads(
        (Path(media.ev.hallway.WORLDS) / "hallway-open" / "0.1.0" / "world.json").read_text()
    )
    p = {
        "spawn": "spawn-entrance",
        "time_limit": 3.0,
        "exit_x": 0.5,
        "command_speed": 0.5,
    }
    obs = media.ev.run_trial(
        "test_media_binding:TOUCH",
        "stock",
        {},
        p,
        world,
        on_step=lambda m, b: media.Recorder(m, b, mcap_path=path),
    )
    assert obs["wall_contacts"], "the toy robot is in the wall"
    with path.open("rb") as f:
        messages = [
            (c, json.loads(m.data))
            for _, c, m in make_reader(f).iter_messages(topics=["/wall_contacts"])
        ]
    assert len(messages) == len(obs["wall_contacts"])
    for (_, message), event in zip(messages, obs["wall_contacts"]):
        assert {k: message[k] for k in ("wall", "start", "end", "depth")} == event
        assert set(message) == {"timestamp", "wall", "start", "end", "depth"}


TOUCH = ToyTouch()


def test_each_trial_gets_a_valid_merged_binding_next_to_its_mcap(recorded, monkeypatch):
    tmp_path, repo = recorded
    _nominal_conf(monkeypatch, with_nominal=True)
    doc = _media(tmp_path, repo)
    for trial in doc["trials"]:
        entry = trial["binding"]
        assert entry["path"] == f"mcap/trial-{trial['index']:02d}.binding.json"
        path = tmp_path / "m" / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        merged = json.loads(path.read_text())
        _valid(merged)
        assert merged["kind"] == "merged" and merged["trial_id"] == f"trial-{trial['index']:02d}"
        assert merged["recording"] == {
            "uri": f"trial-{trial['index']:02d}.mcap",
            "sha256": trial["mcap"]["sha256"],
        }
        assert merged["robot"] == {"id": "toy-robot", "configuration": "stock"}
        assert merged["producer"] == {"id": "livingproof-sim-mujoco"}
        assert merged["bound_observables"] == ["base", "base_height", "body_contact"]
        assert merged["unbound"] == []
        assert merged["observables"]["base"]["body"] == "base"
        assert merged["observables"]["base_height"]["nominal"] == {
            "value": 0.3,
            "unit": "meters",
            "source": "measured",
            "ref": "stock.json",
        }
        assert merged["observables"]["body_contact"] == EXPECTED_CONTACT
        # The recording the binding names has the channels the binding names.
        topics = _topics(path.with_name(f"trial-{trial['index']:02d}.mcap"))
        for term in merged["observables"].values():
            assert term["topic"] in topics


def test_a_robot_with_no_nominal_leaves_base_height_unbound_not_guessed(recorded):
    tmp_path, repo = recorded
    doc = _media(tmp_path, repo)  # the toy configuration has no nominal_base_height
    merged = json.loads((tmp_path / "m" / doc["trials"][0]["binding"]["path"]).read_text())
    _valid(merged)
    assert merged["bound_observables"] == ["base", "body_contact"]
    assert merged["unbound"] == [{"id": "base_height", "reason": "robot-entry-missing"}]
    assert "base_height" not in merged["observables"]


def test_the_merge_follows_the_evaluators_rules():
    robot = {
        "format": binding.FORMAT,
        "kind": "robot",
        "robot": {"id": "r"},
        "configuration": "stock",
        "observables": {"base": {"body": "pelvis"}},
    }
    producer = binding.producer_binding()
    recording = {"uri": "t.mcap", "sha256": "0" * 64}
    merged = binding.merge(robot, producer, "trial-01", recording)
    assert merged["bound_observables"] == ["base", "body_contact"]
    assert merged["unbound"] == [{"id": "base_height", "reason": "robot-entry-missing"}]
    assert list(merged["observables"]) == merged["bound_observables"]
    no_body = copy.deepcopy(robot)
    no_body["observables"] = {}
    again = binding.merge(no_body, producer, "trial-01", recording)
    assert {"id": "base", "reason": "robot-entry-missing"} in again["unbound"]
    thin = copy.deepcopy(producer)
    del thin["observables"]["body_contact"]
    last = binding.merge(robot, thin, "trial-01", recording)
    assert {"id": "body_contact", "reason": "producer-does-not-carry"} in last["unbound"]


def test_the_robot_nominal_is_in_meters_or_not_offered():
    conf = {"locomotion": {"nominal_base_height": {"value": 0.78, "unit": "m"}}}
    nominal = binding.nominal_from(conf, "configurations/stock.json")
    assert nominal == {
        "value": 0.78,
        "unit": "meters",
        "source": "measured",
        "ref": "configurations/stock.json",
    }
    for bad in (
        {},
        {"locomotion": {}},
        {"locomotion": {"nominal_base_height": {"value": 0.78, "unit": "cm"}}},
        {"locomotion": {"nominal_base_height": {"value": -1, "unit": "m"}}},
        {"locomotion": {"nominal_base_height": {"value": "0.78", "unit": "m"}}},
    ):
        assert binding.nominal_from(bad, "x.json") is None, bad
