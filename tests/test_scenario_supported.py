# SPDX-License-Identifier: Apache-2.0
"""scenario_supported is checked before any trial (robot-capability-specs v2.3.0).

Spec: specs/robots/README.md "Scenario support" and the ``scenarios`` description
on livingproof-robot-manifest/0.1. A scenario that lists the precondition is
UNSUPPORTED naming scenario_supported when the robot manifest's ``scenarios``
list does not name it, including when that list is absent. No trial runs, and
the result is never a robot pass or fail. A scenario that does not list the
precondition keeps running. The parameter file says the same thing: with no
``command``, only an adapter whose manifest declares the scenario is run.
"""

import hashlib
import json
from pathlib import Path

import jsonschema
import mujoco
import pytest

from livingproof_scenario_parameters import ScenarioParameterError

from livingproof_sim_mujoco import compose, evidence, observations, scenario

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests/fixtures/robot-capability-specs-v2.3.0"
SCHEMA = json.loads((CATALOG / "robot-manifest.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
MISSING = object()
RECORD = "UNSUPPORTED: precondition scenario_supported unmet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameters(**edits) -> dict:
    document = json.loads((ROOT / "src/livingproof_sim_mujoco/compat/parameters.json").read_text())
    document.update(edits)
    return document


def _require_support(document: dict) -> dict:
    """Keep the velocity command and name the precondition the runner must check."""
    document["preconditions"]["scenario_supported"] = {
        "condition": {"op": "le", "args": [1, 1]},
        "formula": "the robot manifest scenarios list names this scenario",
    }
    return document


def _manifest(scenarios=MISSING) -> dict:
    document = {
        "format": "livingproof-robot-manifest/0.1",
        "robot": "toy-robot",
        "source": {"repository": "Example/toy-robot", "commit": "a" * 40},
        "profile": {"path": "robot.json", "sha256": "b" * 64},
        "adapter": {"entry_point": "test_evidence:TOY", "package": "toy-robot"},
        "configurations": [
            {
                "name": "stock",
                "version": "1.0.0",
                "path": "stock.json",
                "sha256": "c" * 64,
            }
        ],
        "assets": [
            {
                "repository": "Example/toy-robot",
                "commit": "d" * 40,
                "path": "robot.json",
                "sha256": "e" * 64,
                "bytes": 1,
            }
        ],
        "features": {},
    }
    if scenarios is not MISSING:
        document["scenarios"] = scenarios
    return document


def _repo(tmp_path, monkeypatch):
    from test_evidence import Toy, _repo as make_repo

    repo = tmp_path / "repo"
    repo.mkdir()
    make_repo(repo)
    monkeypatch.setattr(Toy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return repo


def _use_parameters(repo: Path, document: dict) -> None:
    path = repo / "parameters.json"
    path.write_text(json.dumps(document))
    lock_path = repo / "caller.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["sources"] = {"parameters": {"path": "parameters.json", "sha256": _sha256(path)}}
    lock_path.write_text(json.dumps(lock))


def _write_manifest(repo: Path, document: dict | None) -> None:
    if document is None:
        return
    VALIDATOR.validate(document)
    (repo / "manifest.json").write_text(json.dumps(document) + "\n")


def _forbid_robot_and_physics(monkeypatch):
    """Robot setup and physics must not run when the precondition is unmet."""
    from test_evidence import Toy

    calls = []

    def spy(label):
        def inner(*args, **kwargs):
            calls.append(label)
            raise AssertionError(f"{label} ran for an unsupported scenario")

        return inner

    monkeypatch.setattr(evidence, "run_trial", spy("run_trial"))
    monkeypatch.setattr(observations, "observe", spy("observe"))
    monkeypatch.setattr(compose, "place", spy("compose.place"))
    monkeypatch.setattr(evidence.mujoco, "MjData", spy("MjData"))
    monkeypatch.setattr(mujoco, "mj_step", spy("mj_step"))
    for name in ("scene", "controller", "policy", "initial_state"):
        monkeypatch.setattr(Toy, name, spy(name))
    return calls


def _support(manifest: dict) -> dict:
    return next(item for item in manifest["preconditions"] if item["name"] == "scenario_supported")


def test_schema_fixture_is_robot_capability_specs_v2_3_0():
    provenance = json.loads((CATALOG / "PROVENANCE.json").read_text())
    assert provenance["tag"] == "v2.3.0"
    assert provenance["commit"] == "57317e617014a6df97bb7fba9d92a890af8d42e5"
    raw = (CATALOG / "robot-manifest.schema.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == provenance["files"]["robot-manifest.schema.json"]["sha256"]
    )
    assert json.loads(raw)["title"] == "livingproof-robot-manifest/0.1"
    assert "scenarios" not in json.loads(raw)["required"]


@pytest.mark.parametrize(
    ("scenarios", "detail"),
    [
        (MISSING, "the robot manifest has no scenarios list"),
        (None, "the robot manifest has no scenarios list"),
        ([], "walk-hallway is not named by the robot manifest scenarios"),
        (["other-scenario"], "walk-hallway is not named by the robot manifest scenarios"),
    ],
)
def test_undeclared_scenario_is_unsupported_with_no_trial_and_no_physics(
    tmp_path, monkeypatch, capsys, scenarios, detail
):
    repo = _repo(tmp_path, monkeypatch)
    _use_parameters(repo, _require_support(_parameters()))
    _write_manifest(repo, None if scenarios is None else _manifest(scenarios))
    calls = _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    recorded = evidence.evidence(
        "test_evidence:TOY", "stock", out, trials=2, lock=repo / "caller.lock.json"
    )
    assert calls == []
    support = _support(recorded)
    assert support == {"name": "scenario_supported", "met": False, "detail": detail}
    assert recorded["trials"]["results"] == []
    assert list((out / "trials").iterdir()) == []
    assert not (out / "run-manifest.json").exists()
    text = (out / "manifest.json").read_text()
    assert "PASS" not in text and "FAIL" not in text
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(tmp_path / "cli"),
            "--lock",
            str(repo / "caller.lock.json"),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == f"{RECORD}: {detail}; no trials run"
    assert printed.startswith(RECORD)
    assert "PASS" not in printed and "FAIL" not in printed


def test_missing_command_is_unsupported_before_a_trial_when_undeclared(
    tmp_path, monkeypatch, capsys
):
    """A parameter file with no command requires the declaration (v2.3.0)."""
    repo = _repo(tmp_path, monkeypatch)
    document = _parameters()
    del document["command"]
    _use_parameters(repo, document)
    calls = _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    recorded = evidence.evidence("test_evidence:TOY", "stock", out, lock=repo / "caller.lock.json")
    assert calls == []
    assert _support(recorded)["met"] is False
    assert recorded["trials"]["results"] == []
    assert list((out / "trials").iterdir()) == []
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(tmp_path / "cli"),
            "--lock",
            str(repo / "caller.lock.json"),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip().startswith(RECORD)


def test_walk_hallway_manifest_leaves_walk_corner_unsupported(tmp_path, monkeypatch, capsys):
    """A manifest.json like the robot packages: scenarios is only walk-hallway.

    walk-corner lists scenario_supported and has no command, so it is
    UNSUPPORTED with no trials. A leftover robot-manifest.json is not read.
    """
    repo = _repo(tmp_path, monkeypatch)
    document = _parameters()
    del document["command"]
    document["scenario"]["id"] = "walk-corner"
    _use_parameters(repo, document)
    lock_path = repo / "caller.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["scenario"] = "walk-corner@0.1.0"
    lock_path.write_text(json.dumps(lock))
    published = _manifest(["walk-hallway"])
    VALIDATOR.validate(published)
    assert published["format"] == "livingproof-robot-manifest/0.1"
    assert published["scenarios"] == ["walk-hallway"]
    _write_manifest(repo, published)
    decoy = _manifest(["walk-corner"])
    (repo / "robot-manifest.json").write_text(json.dumps(decoy) + "\n")
    calls = _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    recorded = evidence.evidence("test_evidence:TOY", "stock", out, lock=repo / "caller.lock.json")
    detail = "walk-corner is not named by the robot manifest scenarios"
    assert calls == []
    assert _support(recorded) == {"name": "scenario_supported", "met": False, "detail": detail}
    assert recorded["trials"]["results"] == []
    assert list((out / "trials").iterdir()) == []
    assert not (out / "run-manifest.json").exists()
    text = (out / "manifest.json").read_text()
    assert "PASS" not in text and "FAIL" not in text
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(tmp_path / "cli"),
            "--lock",
            str(repo / "caller.lock.json"),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == f"{RECORD}: {detail}; no trials run"


def test_real_walk_corner_file_is_unsupported_before_the_world_fixture(
    tmp_path, monkeypatch, capsys
):
    """v2.3.0 walk-corner parameters, and a manifest that only lists walk-hallway.

    hallway-corner is not a vendored fixture. The record is still written, and
    the world fixture is not loaded.
    """
    repo = _repo(tmp_path, monkeypatch)
    raw = (CATALOG / "walk-corner-parameters.json").read_bytes()
    provenance = json.loads((CATALOG / "PROVENANCE.json").read_text())
    assert (
        hashlib.sha256(raw).hexdigest()
        == provenance["files"]["walk-corner-parameters.json"]["sha256"]
    )
    assert "command" not in json.loads(raw)
    (repo / "corner-parameters.json").write_bytes(raw)
    lock_path = repo / "caller.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["scenario"] = "walk-corner@0.1.0"
    lock["world"] = {"id": "hallway-corner", "version": "0.1.0"}
    lock["sources"] = {
        "parameters": {
            "path": "corner-parameters.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    }
    lock_path.write_text(json.dumps(lock))
    _write_manifest(repo, _manifest(["walk-hallway"]))

    def refuse_world(*_args, **_kwargs):
        raise AssertionError("world fixture was loaded for an undeclared scenario")

    monkeypatch.setattr(compose, "vendored_world", refuse_world)
    calls = _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    recorded = evidence.evidence("test_evidence:TOY", "stock", out, trials=1, lock=lock_path)
    detail = "walk-corner is not named by the robot manifest scenarios"
    assert calls == []
    assert _support(recorded) == {"name": "scenario_supported", "met": False, "detail": detail}
    assert recorded["world"] == {"id": "hallway-corner", "version": "0.1.0"}
    assert "clear_width" not in recorded["world"]
    assert recorded["trials"] == {
        "count": 1,
        "results": [],
        "reproducibility": {"compared": False, "identical": None, "finding": None},
    }
    assert list((out / "trials").iterdir()) == []
    assert not (out / "run-manifest.json").exists()
    text = (out / "manifest.json").read_text()
    assert "PASS" not in text and "FAIL" not in text
    assert "unknown fixture" not in text
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(tmp_path / "cli"),
            "--lock",
            str(lock_path),
            "--trials",
            "1",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == f"{RECORD}: {detail}; no trials run"


def test_a_route_the_world_does_not_have_is_an_error():
    """A missing route place is ERROR, not an uncaught KeyError."""
    from test_evidence import CONF, ROBOT, WORLD

    document = json.loads((CATALOG / "walk-corner-parameters.json").read_text())
    with pytest.raises(ScenarioParameterError, match="ERROR") as caught:
        scenario.prepare(ROBOT, CONF, WORLD, document)
    assert "route-centreline" in str(caught.value)
    assert not isinstance(caught.value, KeyError)


def test_declared_scenario_runs_the_trials(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    _use_parameters(repo, _require_support(_parameters()))
    _write_manifest(repo, _manifest(["walk-hallway"]))
    steps = []
    real = mujoco.mj_step

    def counting(model, data):
        steps.append(model)
        return real(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counting)
    out = tmp_path / "out"
    recorded = evidence.evidence(
        "test_evidence:TOY", "stock", out, trials=1, lock=repo / "caller.lock.json"
    )
    assert steps, "a declared scenario steps physics"
    assert _support(recorded) == {
        "name": "scenario_supported",
        "met": True,
        "detail": "walk-hallway",
    }
    assert [item["index"] for item in recorded["trials"]["results"]] == [1]
    assert recorded["trials"]["results"][0]["outcome"] == "crossed_exit"
    assert (out / "trials" / "trial-01.json").is_file()
    assert (out / "run-manifest.json").is_file()
    code = evidence.main(
        [
            "test_evidence:TOY",
            "stock",
            str(tmp_path / "cli"),
            "--lock",
            str(repo / "caller.lock.json"),
            "--trials",
            "1",
        ]
    )
    assert code == 0


def test_declared_scenario_without_a_command_does_not_invent_one(tmp_path, monkeypatch):
    """A declared outcome-only scenario is not driven by an invented velocity."""
    repo = _repo(tmp_path, monkeypatch)
    document = _parameters()
    del document["command"]
    _use_parameters(repo, document)
    _write_manifest(repo, _manifest(["walk-hallway"]))
    calls = _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="no command"):
        evidence.evidence("test_evidence:TOY", "stock", out, lock=repo / "caller.lock.json")
    assert calls == []
    assert not out.exists()


def test_a_scenario_that_does_not_list_the_precondition_still_runs(tmp_path, monkeypatch):
    """walk-hallway keeps its command and does not list scenario_supported."""
    repo = _repo(tmp_path, monkeypatch)
    _use_parameters(repo, _parameters())
    _write_manifest(repo, _manifest(["other-scenario"]))
    recorded = evidence.evidence(
        "test_evidence:TOY", "stock", tmp_path / "out", trials=1, lock=repo / "caller.lock.json"
    )
    assert "scenario_supported" not in {item["name"] for item in recorded["preconditions"]}
    assert [item["index"] for item in recorded["trials"]["results"]] == [1]


def test_invalid_scenarios_is_a_test_system_error_and_writes_nothing(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    _use_parameters(repo, _require_support(_parameters()))
    # The schema rejects this. The runner must too, and must not call it unsupported.
    (repo / "manifest.json").write_text(json.dumps(_manifest(["Walk-Corner"])))
    calls = _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="ERROR"):
        evidence.evidence("test_evidence:TOY", "stock", out, lock=repo / "caller.lock.json")
    assert calls == []
    assert not out.exists()


def test_unsupported_record_matches_the_evidence_shape_the_spec_grades(tmp_path, monkeypatch):
    """The written manifest is livingproof-hallway-evidence/2.0.0 with zero trial results.

    robot-capability-specs v2.3.0 does not publish a second schema for this line.
    The grade is UNSUPPORTED naming scenario_supported, decided before any trial.
    The robot manifest itself validates against livingproof-robot-manifest/0.1.
    """
    repo = _repo(tmp_path, monkeypatch)
    manifest = _manifest()
    VALIDATOR.validate(manifest)
    _write_manifest(repo, manifest)
    _use_parameters(repo, _require_support(_parameters()))
    _forbid_robot_and_physics(monkeypatch)
    out = tmp_path / "out"
    evidence.evidence("test_evidence:TOY", "stock", out, trials=2, lock=repo / "caller.lock.json")
    recorded = json.loads((out / "manifest.json").read_text())
    assert recorded["format"] == evidence.FORMAT == "livingproof-hallway-evidence/2.0.0"
    assert recorded["trials"]["count"] == recorded["plan"]["trials"]["count"] == 2
    assert recorded["trials"]["results"] == []
    support = _support(recorded)
    assert support["name"] == "scenario_supported" and support["met"] is False
    assert isinstance(support["detail"], str) and support["detail"]
    blob = json.dumps(recorded)
    assert "PASS" not in blob and "FAIL" not in blob
