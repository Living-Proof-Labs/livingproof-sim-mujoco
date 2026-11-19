# SPDX-License-Identifier: Apache-2.0
"""Outcome-based hallway and corner tests with configurable route control and Pose API.

Validates:
- Straight hallway outcome verification via ``observe_route`` + MCAP ``/base_pose``.
- Configurable corner left/right/auto and custom-waypoint routes on format 0.3 worlds.
- Closed-loop ``PoseProvider`` injection (interchangeable providers give identical actions;
  missing/stale poses raise explicitly) vs ``open_loop`` deterministic velocity schedules
  that never read pose feedback.
- Compatibility with ``evidence.run_trial`` while leaving historical ``walk-hallway`` intact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from livingproof_scenario_parameters import crosses_exit

from livingproof_sim_mujoco import compose, evidence, hallway, observations, pins, pose, scenario
from livingproof_sim_mujoco.media import Recorder, read_mcap_poses
from livingproof_sim_mujoco.pose import MujocoPoseProvider, PoseSample, PoseUnavailableError

ROOT = Path(__file__).resolve().parents[1]
WORLD_0_3 = (
    ROOT
    / "tests"
    / "fixtures"
    / "livingproof-worlds-v1.4.0"
    / "hallway-corner"
    / "0.2.0"
    / "world.json"
)
XML = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="floor" type="plane" size="10 10 0.1"/>
<body name="base" pos="0 0 0.3"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
</worldbody></mujoco>"""

ROBOT_PROFILE = {
    "class": "humanoid",
    "dimensions": {"width": {"value": 0.4, "unit": "m"}},
    "simulators": ["mujoco"],
}
CONF_PROFILE = {
    "locomotion": {"fall_base_height": {"value": 0.1, "unit": "m"}},
    "controller": {"interface": "velocity_command"},
}


class _Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)


class _HolonomicYawPolicy:
    """Body-frame velocity command ``[vx, vy, wz]`` driving a toy floating sphere."""

    def __init__(self, model):
        self.model = model
        self.commands: list[list[float]] = []

    def step(self, data, command):
        vx, vy, wz = (float(v) for v in command)
        self.commands.append([vx, vy, wz])
        sample = pose.get_pose(self.model, data, "base")
        wx = math.cos(sample.yaw) * vx - math.sin(sample.yaw) * vy
        wy = math.sin(sample.yaw) * vx + math.cos(sample.yaw) * vy
        data.qvel[0] = wx
        data.qvel[1] = wy
        data.qvel[2] = 0.0
        data.qvel[3:6] = [0.0, 0.0, wz]
        mujoco.mj_step(self.model, data)


def _pure_pursuit_command(waypoints: list[list[float]], sample: PoseSample, speed: float = 0.8):
    xy = np.array([p[:2] for p in waypoints], dtype=float)
    last = xy[-1] + (xy[-1] - xy[-2])
    xy = np.vstack([xy, last])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    p = np.array([sample.x, sample.y], dtype=float)

    best, best_s, nearest = math.inf, 0.0, xy[0]
    for i in range(len(xy) - 1):
        a, b = xy[i], xy[i + 1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(q - p))
        if d < best:
            best, best_s, nearest = d, arc[i] + t * float(np.linalg.norm(ab)), q

    s_target = min(best_s + 0.5, float(arc[-1]))
    idx = min(int(np.searchsorted(arc, s_target, side="right") - 1), len(xy) - 2)
    frac = (s_target - arc[idx]) / max(arc[idx + 1] - arc[idx], 1e-12)
    target = xy[idx] + frac * (xy[idx + 1] - xy[idx])

    heading = math.atan2(target[1] - sample.y, target[0] - sample.x)
    error = math.atan2(math.sin(heading - sample.yaw), math.cos(heading - sample.yaw))
    wz = float(np.clip(3.0 * error, -2.5, 2.5))
    vx = speed * max(0.0, math.cos(min(abs(error) / 0.6, 1.0) * math.pi / 2.0))
    ox, oy = nearest - p
    left = -math.sin(sample.yaw) * ox + math.cos(sample.yaw) * oy
    vy = float(np.clip(1.5 * left, -0.4, 0.4))
    return [vx, vy, wz]


class ConfigurableRouteToy:
    """Toy robot implementing Pose API + configurable route_control (feedback/open_loop)."""

    id = "configurable-route-toy"
    root = None
    width_decimals = 2

    def base_body(self, name):
        return "base"

    def scene(self, name):
        return _Scene()

    def controller(self, name, scene):
        return None

    def policy(self, model, ctrl):
        return _HolonomicYawPolicy(model)

    def initial_state(self, model, data, ctrl, spawn, *, start_offset=None):
        mujoco.mj_resetData(model, data)
        pose.set_pose_at_place(model, data, "base", spawn, z=0.3, offset=start_offset)

    def runner_settings(self):
        return {"sample_hz": 50.0}

    def robot(self):
        return ROBOT_PROFILE

    def configuration(self, name):
        return CONF_PROFILE

    def environment(self, name):
        return {"model": "toy"}

    def drive(self, name, model, ctrl, policy, places, *, route_control=None, pose_provider=None):
        del name, ctrl
        cfg = dict(route_control or {})
        mode = cfg.get("mode", "feedback")
        turn_direction = cfg.get("turn_direction", "auto")
        waypoints = cfg.get("waypoints") or places["route"]["points"]
        speed = float(cfg.get("speed", 0.8))
        if mode == "open_loop":
            schedule = list(
                cfg.get("schedule")
                or [
                    (3.125, [0.8, 0.0, 0.0]),
                    (0.7853981633974483, [0.0, 0.0, 2.0 if turn_direction != "right" else -2.0]),
                    (4.5, [0.8, 0.0, 0.0]),
                ]
            )
            elapsed = {"t": 0.0}
            dt = float(model.opt.timestep)

            def open_loop_step(data):
                t = elapsed["t"]
                acc = 0.0
                cmd = schedule[-1][1]
                for dur, seg_cmd in schedule:
                    acc += float(dur)
                    if t < acc - 1e-12:
                        cmd = seg_cmd
                        break
                elapsed["t"] = t + dt
                policy.step(data, cmd)

            return open_loop_step

        if mode != "feedback":
            raise ValueError(f"Unsupported route control mode {mode!r}")

        provider = pose_provider or MujocoPoseProvider(model, "base")
        max_age_s = cfg.get("max_age_s")

        def feedback_step(data):
            sample = pose.validate_pose_sample(
                provider.read(data),
                expected_frame="world",
                max_age_s=max_age_s,
                now_s=float(data.time),
            )
            policy.step(data, _pure_pursuit_command(waypoints, sample, speed=speed))

        return feedback_step


CONFIG_TOY = ConfigurableRouteToy()


def _world_0_3() -> dict:
    return json.loads(WORLD_0_3.read_text())


_PARAMETER_TEMPLATE = json.loads(
    (
        ROOT
        / "src"
        / "livingproof_sim_mujoco"
        / "scenarios"
        / "walk-corner"
        / "0.1.0"
        / "parameters.json"
    ).read_text()
)


def _corner_parameters(direction: str) -> dict:
    traversal = "forward" if direction == "left" else "reverse"
    start = "spawn-west" if direction == "left" else "spawn-north"
    exit_name = "end-north" if direction == "left" else "end-west"
    route_name = f"route-{traversal}"
    doc = json.loads(json.dumps(_PARAMETER_TEMPLATE))
    doc["scenario"] = {"id": f"walk-corner-{direction}", "revision": "0.1.0"}
    doc["world"] = {"id": "hallway-corner", "content_version": "0.2.0"}
    doc["places"] = {
        "start": start,
        "exit": exit_name,
        "route": route_name,
        "through": ["corner"],
    }
    return doc


def _straight_parameters() -> dict:
    doc = json.loads(json.dumps(_PARAMETER_TEMPLATE))
    doc["scenario"] = {"id": "walk-straight-outcome", "revision": "0.1.0"}
    doc["world"] = {"id": "hallway-open", "content_version": "0.1.0"}
    doc["places"] = {
        "start": "spawn-entrance",
        "exit": "exit",
        "route": "passage-centreline",
        "through": [],
    }
    return doc


def _assert_mcap_matches_outcomes(mcap_path: Path, places: dict, fall_height: float):
    msgs = read_mcap_poses(mcap_path)
    assert len(msgs) >= 10
    xyz = []
    for msg in msgs:
        pos = msg["pose"]["position"]
        xyz.append([pos["x"], pos["y"], pos["z"]])
        assert pos["z"] >= fall_height
        assert msg["frame_id"] == "world"
    route_pts = places["route"]["points"]
    crossed = any(crosses_exit(a, b, route_pts, places["exit"]) for a, b in zip(xyz[:-1], xyz[1:]))
    assert crossed, "MCAP /base_pose trace did not cross the exit segment"
    return msgs


def test_outcome_straight_hallway_pass_and_wall_intrusion_failure(tmp_path):
    world = hallway.load_fixture("hallway-open", "0.1.0")["world"]
    params = _straight_parameters()
    places = scenario.scenario_places(world, params)
    prepared = scenario.prepare(ROBOT_PROFILE, CONF_PROFILE, world, params).recorded

    model, data, ctrl, walls = compose.place(CONFIG_TOY, "stock", world, prepared["spawn"])
    policy = CONFIG_TOY.policy(model, ctrl)
    mcap_pass = tmp_path / "straight-pass.mcap"
    recorder = Recorder(model, "base", mcap_path=mcap_pass, video_path=None)
    step = CONFIG_TOY.drive(
        "stock",
        model,
        ctrl,
        policy,
        places,
        route_control={"mode": "feedback"},
        pose_provider=MujocoPoseProvider(model, "base"),
    )
    obs = observations.observe_route(
        model,
        data,
        step,
        base_body="base",
        walls=walls,
        seconds=prepared["time_limit"],
        route=places["route"],
        exit_place=places["exit"],
        through=places["through"],
        on_step=recorder,
    )
    recorder.close()

    assert obs["format"] == observations.ROUTE_FORMAT
    assert obs["end"] == "crossed_exit"
    assert obs["left_world"] is None
    assert obs["wall_contacts"] == []
    assert obs["max_route_distance"]["value"] <= prepared["lateral_limit"]
    assert obs["min_base_height"]["value"] >= prepared["fall_height"]
    _assert_mcap_matches_outcomes(mcap_pass, places, prepared["fall_height"])

    # Negative outcome check: drifting sideways violates wall clearance & exits wall.
    model2, data2, ctrl2, walls2 = compose.place(CONFIG_TOY, "stock", world, prepared["spawn"])
    policy2 = CONFIG_TOY.policy(model2, ctrl2)
    drift_obs = observations.observe_route(
        model2,
        data2,
        lambda d: policy2.step(d, [0.4, 0.9, 0.0]),
        base_body="base",
        walls=walls2,
        seconds=prepared["time_limit"],
        route=places["route"],
        exit_place=places["exit"],
        through=places["through"],
    )
    assert drift_obs["end"] == "left_world"
    assert drift_obs["wall_contacts"], "wall intrusion must be recorded on sideways collision"


@pytest.mark.parametrize("direction", ["left", "right"])
def test_outcome_corner_routes_via_evidence_run_trial_and_mcap(tmp_path, monkeypatch, direction):
    world = _world_0_3()
    params = _corner_parameters(direction)
    places = scenario.scenario_places(world, params)
    prepared = scenario.prepare(ROBOT_PROFILE, CONF_PROFILE, world, params).recorded

    monkeypatch.setattr(ConfigurableRouteToy, "root", tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    mcap_path = tmp_path / f"corner-{direction}.mcap"
    recorders: list[Recorder] = []

    def hook(model, base_body):
        rec = Recorder(model, base_body, mcap_path=mcap_path, video_path=None)
        recorders.append(rec)
        return rec

    obs = evidence.run_trial(
        "test_outcome_routes:CONFIG_TOY",
        "stock",
        {"trials": 1, "route_control": {"mode": "feedback", "turn_direction": direction}},
        prepared,
        world,
        on_step=hook,
        places=places,
    )
    assert obs["end"] == "crossed_exit"
    assert obs["left_world"] is None
    assert obs["places_entered"]["corner"] is not None
    assert obs["places_entered"]["corner"] < obs["exit_time"]
    assert obs["wall_contacts"] == []
    assert obs["max_route_distance"]["value"] <= prepared["lateral_limit"]
    assert obs["min_base_height"]["value"] >= prepared["fall_height"]
    _assert_mcap_matches_outcomes(mcap_path, places, prepared["fall_height"])


def test_interchangeable_fake_pose_providers_produce_identical_commands():
    world = _world_0_3()
    params = _corner_parameters("left")
    places = scenario.scenario_places(world, params)
    model, data, ctrl, _ = compose.place(CONFIG_TOY, "stock", world, places["start"]["name"])

    samples = [
        PoseSample.from_yaw(-2.5, 0.0, 0.3, 0.0, timestamp_s=0.0, source="provider-a"),
        PoseSample.from_yaw(-0.1, 0.05, 0.3, 0.15, timestamp_s=1.0, source="provider-a"),
        PoseSample.from_yaw(0.0, 2.2, 0.3, math.pi / 2, timestamp_s=2.0, source="provider-a"),
    ]

    class DictQueueProvider:
        frame_id = "world"
        source = "fake-dict-stream"

        def __init__(self, items: list[PoseSample]):
            self._items = list(items)
            self._i = 0

        def read(self, _data=None) -> PoseSample:
            s = self._items[self._i]
            self._i += 1
            return PoseSample.from_yaw(
                s.x,
                s.y,
                s.z,
                s.yaw,
                frame_id=s.frame_id,
                timestamp_s=s.timestamp_s,
                source=self.source,
            )

    class ReplayProvider:
        frame_id = "world"
        source = "fake-replay"

        def __init__(self, items: list[PoseSample]):
            self._items = iter(items)

        def read(self, _data=None) -> PoseSample:
            return next(self._items)

    policy_a = CONFIG_TOY.policy(model, ctrl)
    policy_b = CONFIG_TOY.policy(model, ctrl)
    step_a = CONFIG_TOY.drive(
        "stock", model, ctrl, policy_a, places, pose_provider=DictQueueProvider(samples)
    )
    step_b = CONFIG_TOY.drive(
        "stock", model, ctrl, policy_b, places, pose_provider=ReplayProvider(samples)
    )
    for sample in samples:
        data.time = sample.timestamp_s
        step_a(data)
        step_b(data)

    assert np.allclose(policy_a.commands, policy_b.commands)


def test_open_loop_never_queries_pose_provider_and_feedback_rejects_unavailable_pose():
    world = _world_0_3()
    params = _corner_parameters("left")
    places = scenario.scenario_places(world, params)
    prepared = scenario.prepare(ROBOT_PROFILE, CONF_PROFILE, world, params).recorded

    class FailingProvider:
        frame_id = "world"
        source = "broken"
        calls = 0

        def read(self, _data=None):
            self.calls += 1
            raise PoseUnavailableError("sensor offline")

    broken = FailingProvider()
    model, data, ctrl, walls = compose.place(CONFIG_TOY, "stock", world, places["start"]["name"])
    policy = CONFIG_TOY.policy(model, ctrl)
    open_step = CONFIG_TOY.drive(
        "stock",
        model,
        ctrl,
        policy,
        places,
        route_control={"mode": "open_loop", "turn_direction": "left"},
        pose_provider=broken,
    )
    obs = observations.observe_route(
        model,
        data,
        open_step,
        base_body="base",
        walls=walls,
        seconds=prepared["time_limit"],
        route=places["route"],
        exit_place=places["exit"],
        through=places["through"],
    )
    assert broken.calls == 0, "open_loop must not read pose feedback"
    assert obs["end"] == "crossed_exit"
    assert obs["wall_contacts"] == []
    assert obs["places_entered"]["corner"] is not None

    fb_step = CONFIG_TOY.drive(
        "stock",
        model,
        ctrl,
        policy,
        places,
        route_control={"mode": "feedback"},
        pose_provider=broken,
    )
    with pytest.raises(PoseUnavailableError, match="sensor offline"):
        fb_step(data)
    assert broken.calls == 1


def test_custom_waypoint_route_control_and_legacy_adapter_refusal(tmp_path, monkeypatch):
    world = _world_0_3()
    params = _corner_parameters("left")
    places = scenario.scenario_places(world, params)
    prepared = scenario.prepare(ROBOT_PROFILE, CONF_PROFILE, world, params).recorded

    model, data, ctrl, walls = compose.place(CONFIG_TOY, "stock", world, places["start"]["name"])
    policy = CONFIG_TOY.policy(model, ctrl)
    custom_waypoints = [[-2.5, 0.0], [-0.4, 0.0], [0.0, 0.4], [0.0, 3.0]]
    step = CONFIG_TOY.drive(
        "stock",
        model,
        ctrl,
        policy,
        places,
        route_control={"mode": "feedback", "waypoints": custom_waypoints, "speed": 0.9},
    )
    obs = observations.observe_route(
        model,
        data,
        step,
        base_body="base",
        walls=walls,
        seconds=prepared["time_limit"],
        route=places["route"],
        exit_place=places["exit"],
        through=places["through"],
    )
    assert obs["end"] == "crossed_exit"
    assert obs["wall_contacts"] == []

    # Legacy 5-arg adapter refuses explicit route_control rather than silently ignoring it.
    class Legacy5Arg(ConfigurableRouteToy):
        id = "legacy-5-arg"

        def drive(self, name, model, ctrl, policy, places):
            return lambda d: policy.step(d, [0.5, 0.0, 0.0])

    monkeypatch.setattr(Legacy5Arg, "root", tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    legacy = Legacy5Arg()
    globals()["LEGACY_5"] = legacy
    with pytest.raises(SystemExit, match="does not accept route_control"):
        evidence.run_trial(
            "test_outcome_routes:LEGACY_5",
            "stock",
            {"trials": 1, "route_control": {"mode": "open_loop"}},
            prepared,
            world,
            places=places,
        )


def test_evidence_plan_carries_route_control_through_manifest_and_trial(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pins.json").write_text(json.dumps({"groups": {}}))
    (repo / "robot.json").write_text(json.dumps(ROBOT_PROFILE))
    (repo / "stock.json").write_text(json.dumps(CONF_PROFILE))
    (repo / "manifest.json").write_text(
        json.dumps(
            {
                "format": "livingproof-robot-manifest/0.1",
                "robot": "configurable-route-toy",
                "source": {"repository": "Example/toy", "commit": "a" * 40},
                "profile": {"path": "robot.json", "sha256": "b" * 64},
                "adapter": {
                    "entry_point": "test_outcome_routes:CONFIG_TOY",
                    "package": "configurable-route-toy",
                },
                "configurations": [
                    {
                        "name": "stock",
                        "version": "1.0.0",
                        "path": "stock.json",
                        "sha256": pins.sha256(repo / "stock.json"),
                    }
                ],
                "assets": [
                    {
                        "repository": "Example/toy",
                        "commit": "d" * 40,
                        "path": "robot.json",
                        "sha256": pins.sha256(repo / "robot.json"),
                        "bytes": 1,
                    }
                ],
                "features": {},
                "scenarios": ["walk-corner"],
            }
        )
    )
    world_path = Path(hallway.WORLDS / "hallway-corner" / "0.1.0" / "world.json")
    params_path = (
        Path(__file__).resolve().parents[1]
        / "src/livingproof_sim_mujoco/scenarios/walk-corner/0.1.0/parameters.json"
    )
    (repo / "parameters.json").write_bytes(params_path.read_bytes())
    lock = {
        "plan": {
            "trials": 1,
            "route_control": {"mode": "open_loop", "turn_direction": "left"},
        },
        "scenario": "walk-corner@0.1.0",
        "world": {
            "id": "hallway-corner",
            "version": "0.1.0",
            "sha256": pins.sha256(world_path),
        },
        "configurations": [
            {
                "name": "stock",
                "path": "stock.json",
                "version": "1.0.0",
                "sha256": pins.sha256(repo / "stock.json"),
            }
        ],
        "releases": {
            "runner": {"tag": "v3.2.0", "commit": "a" * 40},
            "specs": {
                "tag": "v2.3.0",
                "commit": "57317e617014a6df97bb7fba9d92a890af8d42e5",
            },
        },
        "sources": {
            "parameters": {
                "path": "parameters.json",
                "sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
            }
        },
    }
    lock_path = repo / "caller.lock.json"
    lock_path.write_text(json.dumps(lock))
    import subprocess

    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "t"], check=True)
    subprocess.run([*git, "remote", "add", "origin", "https://example.test/toy.git"], check=True)
    monkeypatch.setattr(ConfigurableRouteToy, "root", repo)
    monkeypatch.syspath_prepend(str(Path(__file__).parent))

    out = tmp_path / "ev"
    manifest = evidence.evidence("test_outcome_routes:CONFIG_TOY", "stock", out, lock=lock_path)
    assert manifest["plan"]["route_control"] == {"mode": "open_loop", "turn_direction": "left"}
    assert manifest["trials"]["results"][0]["outcome"] == "crossed_exit"
