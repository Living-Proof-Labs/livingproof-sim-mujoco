# SPDX-License-Identifier: Apache-2.0
"""Scenario parameters, route geometry, lock sources, and frozen hallway bytes.

The case files are the bytes of robot-capability-specs v2.2.0 (commit
23bec516bda29be7b26555286cbf2e8217ac5c81). See
tests/fixtures/robot-capability-specs-v2.2.0/PROVENANCE.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from livingproof_scenario_parameters import ScenarioParameterError

from livingproof_sim_mujoco import evidence, scenario, sources
from livingproof_sim_mujoco.compat import hallway as frozen
from livingproof_sim_mujoco.observations import consume_route, route_step
from livingproof_sim_mujoco.pins import sha256


def _toy():
    """The reference toy, imported the same way the evidence command loads it."""
    import sys

    tests = str(ROOT / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import test_evidence

    return test_evidence


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests/fixtures/robot-capability-specs-v2.2.0"
PARITY = ROOT / "tests/fixtures/lpv-run-43402628666"
# Production Python outside these paths must not name the scenario or its world.
ALLOW = (
    "src/livingproof_sim_mujoco/compat",
    "src/livingproof_sim_mujoco/hallway.py",
    "tools/hallway.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog():
    provenance = json.loads((CATALOG / "PROVENANCE.json").read_text())
    assert provenance["tag"] == "v2.2.0"
    assert provenance["commit"] == frozen.PARAMETERS_COMMIT
    loaded = {}
    for name, meta in provenance["files"].items():
        path = CATALOG / name
        assert _sha(path) == meta["sha256"], name
        loaded[name] = json.loads(path.read_text())
    vendored = ROOT / "src/livingproof_sim_mujoco/compat/parameters.json"
    assert _sha(vendored) == provenance["files"]["parameters.json"]["sha256"]
    assert _sha(vendored) == frozen.PARAMETERS_SHA256
    return loaded


def _robot(width, cls="humanoid", fall=0.5):
    return {
        "class": cls,
        "dimensions": {"width": {"value": width, "unit": "m"}},
        "simulators": ["mujoco"],
    }


def _conf(fall=0.5):
    return {
        "locomotion": {"fall_base_height": {"value": fall, "unit": "m"}},
        "controller": {"interface": "velocity_command"},
    }


def _straight_world(route_length, clear, start=-2.5):
    end = start + route_length
    return {
        "format": "livingproof-world/0.1",
        "dimensions": {"clear_width": clear},
        "places": [
            {"name": "spawn-entrance", "type": "pose", "position": [start, 0.0, 0.0]},
            {
                "name": "exit",
                "type": "segment",
                "points": [[end, -clear / 2, 0.0], [end, clear / 2, 0.0]],
                "reference": [end, 0.0, 0.0],
            },
            {
                "name": "passage-centreline",
                "type": "polyline",
                "points": [[start, 0.0, 0.0], [end, 0.0, 0.0]],
            },
        ],
    }


def test_parameter_cases_from_the_pinned_tag():
    catalog = _catalog()
    document = catalog["parameters.json"]
    cases = catalog["parameter-cases.json"]["cases"]
    assert len(cases) == 20
    for case in cases:
        expect = case["expect"]
        if expect.get("error") == "ERROR":
            with pytest.raises(ScenarioParameterError) as caught:
                scenario.resolve_limits(document, case["inputs"])
            assert caught.value.status == "ERROR"
            continue
        resolved = scenario.resolve_limits(document, case["inputs"])
        assert resolved["limits"] == expect["limits"], case["name"]
        assert resolved["preconditions"] == expect["preconditions"], case["name"]


def test_recorded_hallway_limits_keep_historical_rounding():
    """Spec rounding is 4 decimals, half even. The frozen record is not that."""
    document = _catalog()["parameters.json"]
    # 1.5 / 2 - 0.6875 / 2 = 0.40625, which the spec rounds to 0.4062.
    world = _straight_world(5.5, 1.5)
    prepared = scenario.prepare(_robot(0.6875), _conf(), world, document)
    assert prepared.resolved["limits"]["lateral_limit"] == 0.4062
    assert prepared.recorded["lateral_limit"] == 0.40625
    assert prepared.recorded["time_limit"] == 27.0
    assert prepared.resolved["limits"]["time_limit"] == 27.0


def test_straight_hallway_command_is_constant_and_other_modes_are_errors():
    document = _catalog()["parameters.json"]
    toy = _toy()
    prepared = scenario.prepare(toy.ROBOT, toy.CONF, toy.WORLD, document)
    assert prepared.command == [0.5, 0.0, 0.0]
    assert prepared.recorded["command_speed"] == 0.5
    other = json.loads(json.dumps(document))
    other["command"]["mode"] = "route-follow"
    with pytest.raises(ScenarioParameterError) as caught:
        scenario.prepare(toy.ROBOT, toy.CONF, toy.WORLD, other)
    assert caught.value.status == "ERROR"
    assert "ERROR" in str(caught.value)
    assert "route-follow" in str(caught.value)


def test_route_geometry_cases_match_the_reference():
    catalog = _catalog()
    cases = catalog["route-geometry-cases.json"]["cases"]
    distances = crossings = mismatches = 0
    for case in cases:
        for item in case["distances"]:
            distances += 1
            got = route_step(case["route"], case["exit"], None, item["point"])["distance"]
            if got != item["distance"]:
                mismatches += 1
        for item in case["crossings"]:
            crossings += 1
            got = route_step(case["route"], case["exit"], item["p0"], item["p1"])["crosses"]
            if got is not item["crosses"]:
                mismatches += 1
    assert distances == 20 and crossings == 30 and mismatches == 0


def test_route_walk_stops_at_the_first_crossing():
    case = _catalog()["route-geometry-cases.json"]["cases"][0]
    crossing = next(item for item in case["crossings"] if item["crosses"])
    later = [4.0, 0.0, 0.0]
    walked = consume_route([crossing["p0"], crossing["p1"], later], case["route"], case["exit"])
    assert walked["crossing_index"] == 1
    # The sample after the first true crossing is not measured.
    assert (
        walked["max_distance"]
        == route_step(case["route"], case["exit"], crossing["p0"], crossing["p1"])["distance"]
    )


def test_evidence_records_the_vendored_world(tmp_path, monkeypatch):
    """The caller lock selects the world. Geometry is the vendored release, not the adapter."""
    toy = _toy()
    repo = tmp_path / "repo"
    repo.mkdir()
    toy._repo(repo)
    monkeypatch.setattr(toy.Toy, "root", repo)
    out = tmp_path / "out"
    manifest = _evidence(repo, out, trials=1)
    world = json.loads((out / "run-manifest.json").read_text())["world"]
    vendored = ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json"
    assert world["id"] == "hallway-open" and world["version"] == "0.1.0"
    assert world["sha256"] == _sha(vendored)
    assert manifest["world"]["clear_width"] == 1.5
    assert manifest["world"]["exit_x"] == 3.0
    assert manifest["trials"]["results"][0]["outcome"] == "crossed_exit"
    assert manifest["environment"]["walls"] == ["wall-left", "wall-right"]


def _write(root: Path, relative: str, payload: dict) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return sha256(path)


def _pins(root: Path, group: str, relative: str, digest: str) -> None:
    document = {
        "groups": {group: {"repository": "o/r", "commit": "c" * 40, "dest": ".cache", "files": {}}}
    }
    pins_path = root / "pins.json"
    if pins_path.is_file():
        document = json.loads(pins_path.read_text())
    document["groups"].setdefault(
        group, {"repository": "o/r", "commit": "c" * 40, "dest": ".cache", "files": {}}
    )
    document["groups"][group]["files"][relative] = {"sha256": digest}
    pins_path.write_text(json.dumps(document))


def test_lock_entry_wins_over_pins_and_a_mismatch_is_an_error(tmp_path):
    root = tmp_path / "robot"
    root.mkdir()
    lock_world = {
        "format": "livingproof-world/0.1",
        "id": "from-lock",
        "dimensions": {"clear_width": 1.1},
    }
    pins_world = {
        "format": "livingproof-world/0.1",
        "id": "from-pins",
        "dimensions": {"clear_width": 9.9},
    }
    lock_parameters = {
        "format": "parameters",
        "command": {"mode": "constant", "forward": 0.3, "sideways": 0, "turn_rate": 0},
    }
    pins_parameters = {
        "format": "parameters",
        "command": {"mode": "constant", "forward": 0.2, "sideways": 0, "turn_rate": 0},
    }
    world_sha = _write(root, "locked/world.json", lock_world)
    param_sha = _write(root, "locked/parameters.json", lock_parameters)
    pins_world_sha = _write(root, ".cache/worlds/from-lock/0.1.0/world.json", pins_world)
    pins_param_sha = _write(
        root, ".cache/specs/scenarios/from-lock/parameters.json", pins_parameters
    )
    _pins(root, "worlds", "worlds/from-lock/0.1.0/world.json", pins_world_sha)
    _pins(root, "specs", "specs/scenarios/from-lock/parameters.json", pins_param_sha)
    lock = {
        "scenario": "from-lock@0.1.0",
        "world": {"id": "from-lock", "version": "0.1.0"},
        "sources": {
            "world": {"path": "locked/world.json", "sha256": world_sha},
            "parameters": {"path": "locked/parameters.json", "sha256": param_sha},
        },
    }
    assert sources.world_document(root, lock)["id"] == "from-lock"
    document, origin = sources.parameters_document_for(root, lock)
    assert origin == "lock" and document["command"]["forward"] == 0.3

    no_entry = {"scenario": "from-lock@0.1.0", "world": {"id": "from-lock", "version": "0.1.0"}}
    assert sources.world_document(root, no_entry)["id"] == "from-pins"
    document, origin = sources.parameters_document_for(root, no_entry)
    assert origin == "pins" and document["command"]["forward"] == 0.2
    assert sources.parameters_document_for(root, None)[1] == "vendored"

    bad = json.loads(json.dumps(lock))
    bad["sources"]["world"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="ERROR") as caught:
        sources.world_document(root, bad)
    assert "mismatch" in str(caught.value)
    # The pins world is a different id, so a mismatch must not fall through.
    assert "from-pins" not in str(caught.value)


def test_l_shaped_route_matches_and_the_adapter_world_is_not_used(tmp_path, monkeypatch):
    catalog = _catalog()
    world = json.loads((ROOT / "tests/fixtures/l-shaped-0.2.json").read_text())
    assert world["format"] == "livingproof-world/0.2"
    route = next(p for p in world["places"] if p["name"] == "route-centreline")["points"]
    exit_place = next(p for p in world["places"] if p["name"] == "exit")
    exit_doc = {"points": exit_place["points"], "reference": exit_place["reference"]}
    case = next(
        item
        for item in catalog["route-geometry-cases.json"]["cases"]
        if item["route"] == route and item["exit"] == exit_doc
    )
    for item in case["distances"]:
        assert route_step(route, exit_doc, None, item["point"])["distance"] == item["distance"]
    for item in case["crossings"]:
        assert route_step(route, exit_doc, item["p0"], item["p1"])["crosses"] is item["crosses"]
    toy = _toy()
    repo = tmp_path / "repo"
    repo.mkdir()
    toy._repo(repo)
    monkeypatch.setattr(toy.Toy, "root", repo)
    monkeypatch.setattr(toy.Toy, "world", lambda self: world)
    out = tmp_path / "out"
    manifest = _evidence(repo, out, trials=1)
    # The adapter's world is not the simulated one, including a 0.2 document.
    assert manifest["world"]["clear_width"] == 1.5
    assert "livingproof-world/0.2" in frozen.unsupported_world(world)


def test_verifier_traces_reproduce_exit_and_lateral():
    """livingproof-verification run 43402628666. Digests are in PROVENANCE.json."""
    provenance = json.loads((PARITY / "PROVENANCE.json").read_text())
    assert provenance["run_id"] == 43402628666
    route = [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    exit_place = {
        "points": [[3.0, -0.75, 0.0], [3.0, 0.75, 0.0]],
        "reference": [3.0, 0.0, 0.0],
    }
    for item in provenance["files"]:
        path = PARITY / item["file"]
        assert _sha(path) == item["file_sha256"]
        obs = json.loads(path.read_text())
        previous = None
        best = -1.0
        best_time = None
        exit_time = None
        for sample in obs["trace"]["samples"]:
            point = [sample[1], sample[2], sample[3]]
            step = route_step(route, exit_place, previous, point)
            previous = point
            if step["distance"] > best:
                best = step["distance"]
                best_time = sample[0]
            if step["crosses"]:
                exit_time = sample[0]
                break
        assert round(best, 4) == obs["max_abs_lateral"]["value"]
        assert best_time == obs["max_abs_lateral"]["time"]
        assert exit_time == obs["exit_time"]


def test_packaging_keeps_the_git_subdirectory(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("package_tool", ROOT / "tools/package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "mujoco==3.17.0 \\\n"
        "    --hash=sha256:" + "ab" * 32 + "\n"
        "livingproof-scenario-parameters @ git+https://example.test/specs@v2.2.0#subdirectory=python\n"
        "    # via livingproof-sim-mujoco\n"
    )
    urls = module.separate_vcs(requirements)
    assert urls == ["git+https://example.test/specs@v2.2.0#subdirectory=python"]
    text = requirements.read_text()
    assert "git+" not in text
    assert "mujoco==3.17.0" in text
    assert "# via livingproof-sim-mujoco" in text


def test_scenario_names_stay_in_the_compatibility_modules():
    bad = []
    for base in ("src", "tools"):
        for path in (ROOT / base).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if any(relative == item or relative.startswith(item + "/") for item in ALLOW):
                continue
            text = path.read_text()
            for name in ("walk-hallway", "hallway-open"):
                if name in text:
                    bad.append(f"{relative}: {name}")
    assert bad == []


# --- Verification checks for #40 step 3 (Vera). Each was a hand check first. ---

HALLWAY_ROUTE = [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
HALLWAY_EXIT = {
    "points": [[3.0, -0.75, 0.0], [3.0, 0.75, 0.0]],
    "reference": [3.0, 0.0, 0.0],
}


@pytest.mark.parametrize(
    ("p0", "p1", "crosses"),
    [
        # Lateral test is at p1, not where the step meets the line.
        ((2.9, 0.7, 0.0), (3.5, 0.95, 0.0), False),
        ((2.8, 0.9, 0.0), (3.2, 0.7, 0.0), True),
        ((2.996, 0.7525, 0.0), (3.006, 0.7495, 0.0), True),
        ((2.996, 0.7475, 0.0), (3.006, 0.7505, 0.0), False),
        # Landing on the line counts; a start on the line does not cross again.
        ((2.9, 0.0, 0.0), (3.0, 0.0, 0.0), True),
        ((3.0, 0.0, 0.0), (3.1, 0.0, 0.0), False),
        # The segment endpoints are inside (|offset| <= half, inclusive).
        ((2.9, 0.75, 0.0), (3.0, 0.75, 0.0), True),
        ((2.9, -0.75, 0.0), (3.0, -0.75, 0.0), True),
    ],
)
def test_exit_rule_counter_examples_match_the_reference(p0, p1, crosses):
    from livingproof_scenario_parameters import crosses_exit

    got = route_step(HALLWAY_ROUTE, HALLWAY_EXIT, p0, p1)
    assert got["crosses"] is crosses
    assert crosses_exit(p0, p1, HALLWAY_ROUTE, HALLWAY_EXIT) is crosses
    # Any crossing also reaches the line; reaching the line alone is not a crossing.
    assert got["reached_line"] or not got["crosses"]


def test_first_sample_past_the_line_is_not_a_step():
    got = route_step(HALLWAY_ROUTE, HALLWAY_EXIT, None, [3.5, 0.0, 0.0])
    assert got["crosses"] is False and got["reached_line"] is False


def test_sub_picometre_clamp_only_removes_float_residue():
    from livingproof_scenario_parameters import distance_to_route

    on_route = [0.123, 0.0, 0.0]
    # The reference returns a few ulps here, not 0. The clamp records 0.
    assert 0.0 <= distance_to_route(HALLWAY_ROUTE, on_route) < 1e-12
    assert route_step(HALLWAY_ROUTE, HALLWAY_EXIT, None, on_route)["distance"] == 0.0
    # A real offset far below the recorded 1e-4 m resolution is kept as is.
    off = [0.123, 1e-9, 0.0]
    assert route_step(HALLWAY_ROUTE, HALLWAY_EXIT, None, off)["distance"] == pytest.approx(
        1e-9, rel=1e-6
    )
    # The clamp never touches the crossing decision.
    assert route_step(HALLWAY_ROUTE, HALLWAY_EXIT, [2.9, 0.0, 0.0], [3.0, 0.0, 0.0])["crosses"]


def test_vendored_parameter_file_is_checked_byte_for_byte(tmp_path, monkeypatch):
    import inspect

    spec = CATALOG / "parameters.json"
    assert _sha(ROOT / "src/livingproof_sim_mujoco/compat/parameters.json") == _sha(spec)
    tampered = tmp_path / "parameters.json"
    tampered.write_bytes(spec.read_bytes() + b" ")
    monkeypatch.setattr(frozen, "_PATH", tampered)
    with pytest.raises(RuntimeError, match="ERROR: vendored parameter file sha256"):
        frozen.parameters_document()
    with pytest.raises(RuntimeError, match="ERROR"):
        sources.parameters_document_for(tmp_path, None)
    # The fallback is temporary and must say so where it is used.
    assert "TODO(#40 step 7)" in inspect.getsource(sources.parameters_document_for)


def _toy_repo(tmp_path, monkeypatch):
    toy = _toy()
    repo = tmp_path / "repo"
    repo.mkdir()
    toy._repo(repo)
    monkeypatch.setattr(toy.Toy, "root", repo)
    return toy, repo


def _edit_lock(repo: Path, **changes) -> None:
    path = repo / "caller.lock.json"
    lock = json.loads(path.read_text())
    lock.update(changes)
    path.write_text(json.dumps(lock))


def _evidence(repo: Path, out: Path, **kwargs):
    return evidence.evidence(
        "test_evidence:TOY", "stock", out, lock=repo / "caller.lock.json", **kwargs
    )


def _pin_parameters(repo: Path, document: dict) -> None:
    relative = "specs/scenarios/walk-hallway/parameters.json"
    digest = _write(repo, f".cache/{relative}", document)
    _pins(repo, "specs", relative, digest)


def test_lock_parameter_mismatch_is_an_error_and_writes_nothing(tmp_path, monkeypatch):
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    document = _catalog()["parameters.json"]
    _pin_parameters(repo, document)
    _write(repo, "locked/parameters.json", document)
    _edit_lock(repo, sources={"parameters": {"path": "locked/parameters.json", "sha256": "0" * 64}})
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="ERROR: lock sources.parameters sha256 mismatch"):
        _evidence(repo, out)
    assert not out.exists()


def test_pinned_parameters_are_used_without_a_lock_entry(tmp_path, monkeypatch):
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    document = json.loads(json.dumps(_catalog()["parameters.json"]))
    document["parameters"]["wall_clearance"]["values"]["humanoid"] = 0.0625
    _pin_parameters(repo, document)
    out = tmp_path / "out"
    _evidence(repo, out, trials=1)
    run = json.loads((out / "run-manifest.json").read_text())
    assert run["parameters"]["wall_clearance"] == 0.0625
    # And the lock entry wins over that pin when present.
    locked = _catalog()["parameters.json"]
    digest = _write(repo, "locked/parameters.json", locked)
    _edit_lock(repo, sources={"parameters": {"path": "locked/parameters.json", "sha256": digest}})
    _evidence(repo, tmp_path / "out2", trials=1)
    run = json.loads((tmp_path / "out2" / "run-manifest.json").read_text())
    assert run["parameters"]["wall_clearance"] == 0.05


def test_a_robot_world_file_is_not_the_simulated_world(tmp_path, monkeypatch):
    """A 0.2 world file in the robot repository is not loaded. The vendored release is."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    world = json.loads((ROOT / "tests/fixtures/l-shaped-0.2.json").read_text())
    digest = _write(repo, "locked/world.json", world)
    _edit_lock(repo, sources={"world": {"path": "locked/world.json", "sha256": digest}})
    out = tmp_path / "out"
    manifest = _evidence(repo, out, trials=1)
    assert manifest["world"]["clear_width"] == 1.5
    assert "livingproof-world/0.2" in frozen.unsupported_world(world)


def test_run_manifest_names_the_vendored_world_not_a_robot_file(tmp_path, monkeypatch):
    """A sources.world file and a pins file are not the world that was loaded."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    pinned = _pin_world(repo, _named_world(clear=3.0))
    selected = _lock_world(repo, _named_world())
    assert selected != pinned
    out = tmp_path / "out"
    manifest = _evidence(repo, out, trials=1)
    vendored = _sha(ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json")
    run = json.loads((out / "run-manifest.json").read_text())
    assert run["world"] == {"id": "hallway-open", "version": "0.1.0", "sha256": vendored}
    assert manifest["world"]["clear_width"] == 1.5
    assert manifest["trials"]["results"][0]["outcome"] == "crossed_exit"
    lock = json.loads((repo / "caller.lock.json").read_text())
    prepared = evidence.prepare_run(_toy.TOY, "stock", lock)
    assert prepared.world_source.origin == "vendored"
    assert evidence.world_identity(repo, lock, prepared.world_source)["sha256"] == vendored


def test_missing_world_identity_writes_nothing(tmp_path, monkeypatch):
    """No world id on the lock: INCOMPATIBLE, and no output folder. A robot file is not used."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    lock_path = repo / "caller.lock.json"
    lock = json.loads(lock_path.read_text())
    del lock["world"]
    lock_path.write_text(json.dumps(lock))
    _lock_world(repo, _named_world())
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="INCOMPATIBLE"):
        _evidence(repo, out)
    assert not out.exists()


def test_source_world_bytes_are_not_the_simulated_world(tmp_path, monkeypatch):
    """A sources.world sha256 that does not match its file is an error.

    The file is not the simulated world. A mismatch exits 1 and writes nothing,
    the same way a world.sha256 that is not the vendored file does.
    """
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    _write(repo, "locked/world.json", _named_world())
    _edit_lock(
        repo,
        world={"id": "hallway-open", "version": "0.1.0"},
        sources={"world": {"path": "locked/world.json", "sha256": "0" * 64}},
    )
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="sources.world"):
        _evidence(repo, out, trials=1)
    assert not out.exists()


@pytest.mark.parametrize(
    ("world", "shown"),
    [
        ({"id": "hallway-open", "version": "0.1.0", "sha256": "ab" * 32}, "world.sha256"),
        ({"id": "other-world", "version": "0.1.0"}, "UNSUPPORTED"),
        ({"id": "hallway-open", "version": "0.2.0"}, "UNSUPPORTED"),
    ],
)
def test_conflicting_or_unknown_world_writes_nothing(tmp_path, monkeypatch, world, shown):
    """A sha256 that is not the vendored file, or an id/version that is not vendored."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    _edit_lock(repo, world=world)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=shown):
        _evidence(repo, out)
    assert not out.exists()


def test_vendored_digest_is_recorded_and_a_pin_is_not(tmp_path, monkeypatch):
    """The manifest names the vendored world.json, not a file pinned in the robot repository."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    pinned = _pin_world(repo, _named_world())
    vendored = _sha(ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json")
    assert pinned != vendored
    out = tmp_path / "out"
    _evidence(repo, out, trials=1)
    run = json.loads((out / "run-manifest.json").read_text())
    assert run["world"]["sha256"] == vendored


def test_lock_world_sha256_equal_to_the_vendored_file_is_accepted(tmp_path, monkeypatch):
    """A lock world.sha256 that names the vendored file is not a conflict."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    vendored = _sha(ROOT / "src/livingproof_sim_mujoco/worlds/hallway-open/0.1.0/world.json")
    _edit_lock(repo, world={**LEGACY_WORLD, "sha256": vendored})
    out = tmp_path / "out"
    _evidence(repo, out, trials=1)
    run = json.loads((out / "run-manifest.json").read_text())
    assert run["world"]["sha256"] == vendored
    lock = json.loads((repo / "caller.lock.json").read_text())
    assert evidence.prepare_run(_toy.TOY, "stock", lock).world_source.origin == "vendored"


def test_locked_world_is_read_once_and_named_by_those_bytes(tmp_path, monkeypatch):
    """The bytes checked against sources.world are the bytes parsed and attested."""
    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    _edit_lock(repo, world=LEGACY_WORLD)
    digest = _lock_world(repo, _named_world())
    locked = (repo / "locked/world.json").resolve()
    reads = []
    real = Path.read_bytes

    def counting(self):
        data = real(self)
        if self.resolve() == locked:
            reads.append(hashlib.sha256(data).hexdigest())
            if len(reads) > 1:  # a second read would see different bytes
                return data.replace(b"1.6", b"9.9")
        return data

    monkeypatch.setattr(Path, "read_bytes", counting)
    lock = json.loads((repo / "caller.lock.json").read_text())
    selected = sources.world_source(repo, lock)
    assert reads == [digest]
    assert selected.origin == "lock" and selected.sha256 == digest
    assert selected.document["dimensions"]["clear_width"] == 1.6


def test_media_replays_the_recorded_lock_not_a_robot_file(tmp_path, monkeypatch):
    """Media uses the lock stored in the evidence manifest. A robot world file is not read."""
    pytest.importorskip("mcap")
    pytest.importorskip("imageio_ffmpeg")
    from livingproof_sim_mujoco import media

    _toy, repo = _toy_repo(tmp_path, monkeypatch)
    _evidence(repo, tmp_path / "ev", trials=1)
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps({"trials": []}))
    calls = []
    monkeypatch.setattr(sources, "world_source", lambda *a: calls.append(a))
    doc = media.media("test_evidence:TOY", "stock", tmp_path / "ev", verification, tmp_path / "m")
    assert calls == [] and len(doc["trials"]) == 1
    # Changing a robot-repo world file does not change the replay.
    (repo / "locked").mkdir(exist_ok=True)
    (repo / "locked/world.json").write_text("{}\n")
    again = media.media(
        "test_evidence:TOY", "stock", tmp_path / "ev", verification, tmp_path / "m2"
    )
    assert [t["observation_sha256"] for t in again["trials"]] == [
        t["observation_sha256"] for t in doc["trials"]
    ]


def test_scenario_names_appear_only_in_compat_hallway():
    """Stricter than the module check above: package source only, one file allowed."""
    allowed = "src/livingproof_sim_mujoco/compat/hallway.py"
    bad = []
    for path in (ROOT / "src").rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative == allowed:
            continue
        text = path.read_text()
        for name in ("walk-hallway", "hallway-open", "walk-corner"):
            if name in text:
                bad.append(f"{relative}: {name}")
    assert bad == []


@pytest.mark.parametrize(
    ("edit", "shown"),
    [
        (lambda d: d["parameters"]["command_speed"].__setitem__("value", 0.6), "0.6"),
        (lambda d: d["command"].__setitem__("sideways", 0.1), "0.1"),
        (lambda d: d["command"].__setitem__("turn_rate", 0.2), "0.2"),
    ],
)
def test_only_the_constant_half_metre_forward_command_is_accepted(edit, shown):
    """#40 step 3 item 5: constant [0.5, 0, 0] only; any other vector is an ERROR."""
    document = json.loads(json.dumps(_catalog()["parameters.json"]))
    toy = _toy()
    assert scenario.prepare(toy.ROBOT, toy.CONF, toy.WORLD, document).command == [0.5, 0.0, 0.0]
    edit(document)
    with pytest.raises(ScenarioParameterError) as caught:
        scenario.prepare(toy.ROBOT, toy.CONF, toy.WORLD, document)
    assert caught.value.status == "ERROR"
    assert "ERROR" in str(caught.value) and shown in str(caught.value)


# Issue #39: the manifests name the world file that was loaded, not a lock or pins digest.
WORLD_PIN = "worlds/hallway-open/0.1.0/world.json"


def _named_world(clear: float = 1.6) -> dict:
    """The toy world as a world file with its own identity."""
    world = json.loads(json.dumps(_toy().WORLD))
    world["dimensions"]["clear_width"] = clear
    return {"id": "hallway-open", "content_version": "0.1.0", **world}


def _lock_world(repo: Path, world: dict) -> str:
    digest = _write(repo, "locked/world.json", world)
    _edit_lock(repo, sources={"world": {"path": "locked/world.json", "sha256": digest}})
    return digest


def _pin_world(repo: Path, world: dict) -> str:
    digest = _write(repo, f".cache/{WORLD_PIN}", world)
    _pins(repo, "worlds", WORLD_PIN, digest)
    return digest


LEGACY_WORLD = {"id": "hallway-open", "version": "0.1.0"}
