"""MCAP recordings of every evidence trial and screencasts of two representative ones,
for linking from the capability report. Never grades.

Generic over robots (the same ``module:NAME`` adapter as ``livingproof_sim_mujoco.evidence``).
It reads the verifier's ``verification.json`` only to pick the first PASS trial and the first
FAIL trial for video. Every trial is re-run in sequence, exactly as the evidence job ran it,
with a read-only ``on_step`` recorder, and the re-run observation must hash to that trial's
evidence sha256 or nothing is kept: each recording is provably of the graded trial.

    python -m livingproof_sim_mujoco.media <module:adapter> <configuration> \\
        <evidence dir> <verification.json> <out>

Writes ``<out>/mcap/trial-NN.mcap`` (every trial), ``<out>/video/trial-NN-{pass,fail}.mp4``
(at most two) and ``<out>/media.json``. Needs the ``media`` extra (mcap, imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def default_mujoco_gl(environ: dict | None = None, platform: str | None = None) -> None:
    """Set ``MUJOCO_GL=osmesa`` for headless Linux, without replacing a user's value.

    MuJoCo reads ``MUJOCO_GL`` when it is imported. The default applies only when
    the platform is Linux, ``DISPLAY`` is unset or empty, and ``MUJOCO_GL`` is
    not already set. macOS, Windows, and a Linux session with ``DISPLAY`` set
    keep MuJoCo's own choice.
    """
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if platform != "linux" or environ.get("DISPLAY") or "MUJOCO_GL" in environ:
        return
    environ["MUJOCO_GL"] = "osmesa"


# MuJoCo reads MUJOCO_GL at import, so this has to run first.
default_mujoco_gl()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from livingproof_sim_mujoco import binding, robot_topics  # noqa: E402
from livingproof_sim_mujoco import evidence as ev  # noqa: E402
from livingproof_sim_mujoco.pins import fetch, sha256  # noqa: E402

FORMAT = "livingproof-hallway-media/2.1.0"
FPS = 25
SIZE = (360, 640)  # height, width
MCAP_HZ = 50
CAMERA = (3.0, -30.0, 200.0)  # tracking distance, elevation, azimuth; adapter.camera overrides

POSE_SCHEMA = {
    "type": "object",
    "title": "foxglove.PoseInFrame",
    "properties": {
        "timestamp": {"type": "object"},
        "frame_id": {"type": "string"},
        "pose": {"type": "object"},
    },
}
JOINT_SCHEMA = {
    "type": "object",
    "title": "livingproof.JointState",
    "properties": {
        "timestamp": {"type": "object"},
        "name": {"type": "array", "items": {"type": "string"}},
        "position": {"type": "array", "items": {"type": "number"}},
        "velocity": {"type": "array", "items": {"type": "number"}},
        "effort": {"type": "array", "items": {"type": "number"}},
    },
}


CONTACT_SCHEMA = {
    "type": "object",
    "title": binding.CONTACT_SCHEMA_NAME,
    "description": (
        "One body_contact event with a hallway wall: the wall, the start and end time in "
        "seconds from the start of the trial, and the depth."
    ),
    "properties": {
        "timestamp": {"type": "object"},
        "wall": {"type": "string"},
        "start": {"type": "number"},
        "end": {"type": "number"},
        "depth": {
            "type": "number",
            "description": (
                "In meters: the intrusion past the wall's inner face, the largest while the "
                "contact lasts, measured on the robot's collision geometry. It is an "
                "intrusion, not a penetration depth."
            ),
        },
    },
}


def pick(verification: dict) -> dict[int, str]:
    """{index: status} for the first PASS and the first FAIL trial, as the verifier decided."""
    trials = sorted(verification["trials"], key=lambda t: t["index"])
    out = {}
    for status in ("PASS", "FAIL"):
        t = next((t for t in trials if t["status"] == status), None)
        if t is not None:
            out[t["index"]] = status
    return out


class Recorder:
    """Read-only ``on_step`` hook: base pose and joint state at MCAP_HZ, frames at FPS."""

    def __init__(
        self, model, base_body, *, mcap_path=None, video_path=None, camera=CAMERA, publish=None
    ):
        self.model, self.mcap_path, self.video_path = model, mcap_path, video_path
        self.publish = publish  # the adapter's hook (robot_topics.publisher), or None
        self.robot_channels: dict[str, tuple] = {}  # topic -> (channel id, schema name, schema)
        self.base = model.body(base_body).id
        self.every_frame = max(1, int(round(1 / (FPS * model.opt.timestep))))
        self.every_sample = max(1, int(round(1 / (MCAP_HZ * model.opt.timestep))))
        self.frames, self.renderer, self.writer = [], None, None
        if video_path is not None:
            self.renderer = mujoco.Renderer(model, *SIZE)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.base
            self.camera.distance, self.camera.elevation, self.camera.azimuth = camera
        if mcap_path is not None:
            from mcap.writer import CompressionType, Writer

            self.stream = Path(mcap_path).open("wb")
            self.writer = Writer(self.stream, compression=CompressionType.ZSTD)
            self.writer.start(profile="", library="livingproof_sim_mujoco.media")
            pose = self.writer.register_schema(
                "foxglove.PoseInFrame", "jsonschema", json.dumps(POSE_SCHEMA).encode()
            )
            joint = self.writer.register_schema(
                "livingproof.JointState", "jsonschema", json.dumps(JOINT_SCHEMA).encode()
            )
            contact = self.writer.register_schema(
                binding.CONTACT_SCHEMA_NAME, "jsonschema", json.dumps(CONTACT_SCHEMA).encode()
            )
            self.pose = self.writer.register_channel(binding.POSE_TOPIC, "json", pose)
            self.joint = self.writer.register_channel("/joint_states", "json", joint)
            self.contact = self.writer.register_channel(binding.CONTACT_TOPIC, "json", contact)
            free = int(mujoco.mjtJoint.mjJNT_FREE)
            self.joints = [j for j in range(model.njnt) if int(model.jnt_type[j]) != free]
            self.names = [model.joint(j).name for j in self.joints]
            self.qadr = model.jnt_qposadr[self.joints]
            self.vadr = model.jnt_dofadr[self.joints]
            self.seq = 0
            self.pending = []  # (log_time, channel, sequence, data), written in time order

    def __call__(self, i: int, data) -> None:
        if self.renderer is not None and i % self.every_frame == 0:
            self.renderer.update_scene(data, self.camera)
            self.frames.append(self.renderer.render())
        if self.writer is not None and i % self.every_sample == 0:
            ns = int(round(float(data.time) * 1e9))
            stamp = {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}
            x, y, z = (round(float(v), 5) for v in data.xpos[self.base])
            w, qx, qy, qz = (round(float(v), 6) for v in data.xquat[self.base])
            messages = (
                (
                    self.pose,
                    {
                        "timestamp": stamp,
                        "frame_id": "world",
                        "pose": {
                            "position": {"x": x, "y": y, "z": z},
                            "orientation": {"x": qx, "y": qy, "z": qz, "w": w},
                        },
                    },
                ),
                (
                    self.joint,
                    {
                        "timestamp": stamp,
                        "name": self.names,
                        "position": np.round(data.qpos[self.qadr], 5).tolist(),
                        "velocity": np.round(data.qvel[self.vadr], 4).tolist(),
                        "effort": np.round(data.actuator_force, 3).tolist(),
                    },
                ),
            )
            for channel, msg in messages:
                self.pending.append(
                    (ns, channel, self.seq, json.dumps(msg, separators=(",", ":")).encode())
                )
            self.seq += 1
            if self.publish is not None:
                self._publish(data, ns, stamp)

    def _publish(self, data, ns: int, stamp: dict) -> None:
        """Ask the adapter for this step's robot topics and queue them at the step's time.

        The adapter gets a read-only snapshot, not ``data``, and does not write the file."""
        for item in self.publish(robot_topics.snapshot(data)) or ():
            message = robot_topics.check(item)
            known = self.robot_channels.get(message.topic)
            if known is None:
                schema = self.writer.register_schema(
                    message.schema_name, "jsonschema", json.dumps(message.schema).encode()
                )
                channel = self.writer.register_channel(message.topic, "json", schema)
                known = self.robot_channels[message.topic] = (
                    channel,
                    message.schema_name,
                    message.schema,
                )
            elif known[1:] != (message.schema_name, message.schema):
                raise ValueError(
                    f"topic {message.topic} changed its schema mid-trial; one topic has one schema"
                )
            payload = {"timestamp": stamp, **message.payload}
            self.pending.append(
                (ns, known[0], self.seq, json.dumps(payload, separators=(",", ":")).encode())
            )
            self.seq += 1

    def finish(self, obs: dict) -> None:
        """Write the trial's wall contacts, one message per event, after the run.

        The contacts are in the observation, which exists only once the trial ends. They are
        queued with the rest and written in time order on close, as observed: nothing here changes the observation or its hash."""
        if self.writer is None:
            return
        for event in obs.get("wall_contacts") or []:
            ns = int(round(float(event["start"]) * 1e9))
            message = {
                "timestamp": {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000},
                "wall": event["wall"],
                "start": event["start"],
                "end": event["end"],
                "depth": event["depth"],
            }
            data = json.dumps(message, separators=(",", ":")).encode()
            self.pending.append((ns, self.contact, self.seq, data))
            self.seq += 1

    def _flush(self) -> None:
        """Write every message in log_time order (stable: equal times keep their arrival order).

        The contacts are known only after the run, and a contact's time is its start, so they go
        into the stream among the poses rather than after them."""
        for ns, channel, sequence, data in sorted(self.pending, key=lambda m: m[0]):
            self.writer.add_message(
                channel, log_time=ns, publish_time=ns, sequence=sequence, data=data
            )
        self.pending = []

    def close(self) -> None:
        if self.renderer is not None:
            import imageio.v2 as imageio

            imageio.mimwrite(
                self.video_path, self.frames, fps=FPS, codec="libx264", quality=6,
                macro_block_size=8,
            )  # fmt: skip
            self.renderer.close()
        if self.writer is not None:
            self._flush()
            self.writer.finish()
            self.stream.close()


def _file(path: Path, out: Path) -> dict:
    return {
        "path": path.relative_to(out).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _write_binding(robot, name, index, mcap: Path, config_ref: str, out: Path) -> dict:
    """The per-trial merged binding, next to the recording it describes.

    The robot half is what the adapter knows: its id, the configuration, the base body and the
    configuration's nominal base height (absent, so unbound, when the configuration has none).
    The recording is named relative to the binding, with its real sha256.
    """
    robot_half = binding.robot_binding(
        robot.id,
        name,
        robot.base_body(name),
        binding.nominal_from(robot.configuration(name), config_ref),
    )
    recording = {"uri": mcap.name, "sha256": sha256(mcap)}
    merged = binding.merge(robot_half, binding.producer_binding(), f"trial-{index:02d}", recording)
    path = mcap.with_name(f"trial-{index:02d}.binding.json")
    path.write_text(json.dumps(merged, indent=2) + "\n")
    return _file(path, out)


def _record(args):
    (
        ref,
        name,
        index,
        status,
        plan,
        p,
        out,
        expected,
        start_offset,
        command,
        route,
        exit_place,
        world,
        places,
        config_ref,
    ) = args
    robot = ev.adapter(ref)
    mcap = out / "mcap" / f"trial-{index:02d}.mcap"
    video = out / "video" / f"trial-{index:02d}-{status.lower()}.mp4" if status else None
    camera = ev.camera(robot, CAMERA)

    def hook(model, base_body):
        return Recorder(
            model,
            base_body,
            mcap_path=mcap,
            video_path=video,
            camera=camera,
            publish=robot_topics.publisher(robot, name, model),
        )

    obs = ev.run_trial(
        ref,
        name,
        plan,
        p,
        world,
        on_step=hook,
        start_offset=start_offset,
        command=command,
        route=route,
        exit_place=exit_place,
        places=places,
        log_fallback=False,
    )
    got = hashlib.sha256(ev.observation_bytes(obs)).hexdigest()
    if got != expected:
        for path in (mcap, video):
            if path is not None:
                path.unlink(missing_ok=True)
        raise RuntimeError(
            f"trial {index}: re-run observation {got[:12]} differs from the evidence "
            f"{expected[:12]}; no recording kept for a different run"
        )
    trial = {"index": index, "observation_sha256": got, "mcap": _file(mcap, out)}
    trial["binding"] = _write_binding(robot, name, index, mcap, config_ref, out)
    if video is not None:
        trial.update(status=status, video=_file(video, out))
    return trial


def media(
    ref: str,
    name: str,
    evidence_dir: Path,
    verification: Path,
    out: Path,
    *,
    lock: Path | None = None,
) -> dict:
    robot = ev.adapter(ref)
    # Pinned upstream files first: adapters load their profile and model from them, so a
    # fresh checkout (no pins fetched yet) would otherwise fail with FileNotFoundError.
    fetch(Path(robot.root))
    manifest = json.loads((Path(evidence_dir) / "manifest.json").read_text())
    if manifest["configuration"]["name"] != name:
        raise SystemExit(f"evidence is for {manifest['configuration']['name']}, not {name}")
    recorded = manifest.get("lock", {}).get("content")
    if not isinstance(recorded, dict):
        raise SystemExit(
            "INCOMPATIBLE: the evidence manifest has no lock content to replay. "
            "Record evidence with --lock, then point media at that directory."
        )
    if lock is not None:
        supplied = ev.read_lock(lock)
        if supplied != recorded:
            raise SystemExit(
                "INCOMPATIBLE: --lock does not match the lock recorded in the evidence "
                "manifest. Media replays that recorded lock. Pass the same file, or omit --lock."
            )
    plan, prepared = ev.recording(robot, recorded, name)
    ev.world_identity(Path(robot.root), recorded, prepared.world_source)
    p = prepared.recorded
    world = prepared.world
    geometry = ev.sc.route_geometry(world, prepared.document)
    route, exit_place = geometry if geometry else (None, None)
    # An outcome-only scenario is driven by the adapter, as in the evidence run.
    places = ev.sc.scenario_places(world, prepared.document) if prepared.command is None else None
    # The nominal's ref names the configuration file at the robot's pinned commit, never a branch.
    # (the evidence run's robot checkout head, which the lock pins).
    commit = manifest["environment"]["source_commit"]
    config_ref = f"{ev.robot_repo(Path(robot.root), recorded)}@{commit}:" + next(
        c["path"] for c in recorded["configurations"] if c["name"] == name
    )
    picked = pick(json.loads(Path(verification).read_text()))
    offset = ev.declared_offset(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Record into a staging folder: a mismatch or crash leaves no partial files, and a
    # re-run into an existing output folder replaces only the media it wrote before.
    stage = out / ".media-partial"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "mcap").mkdir(parents=True)
    (stage / "video").mkdir()
    work = [
        (
            ref,
            name,
            t["index"],
            picked.get(t["index"]),
            plan,
            p,
            stage,
            t["sha256"],
            offset,
            prepared.command,
            route,
            exit_place,
            prepared.world,
            places,
            config_ref,
        )
        for t in manifest["trials"]["results"]
    ]
    try:  # one process, in order: the same nominal trial as the evidence run
        trials = [_record(w) for w in work]
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    doc = {
        "format": FORMAT,
        "configuration": name,
        "run_id": manifest["run_id"],
        "trials": sorted(trials, key=lambda t: t["index"]),
    }
    for part in ("mcap", "video"):
        shutil.rmtree(out / part, ignore_errors=True)
        (stage / part).rename(out / part)
    (out / "media.json").write_text(json.dumps(doc, indent=2) + "\n")
    stage.rmdir()
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", help="robot adapter, module:NAME (e.g. lpl_g1.adapter:ROBOT)")
    ap.add_argument("configuration")
    ap.add_argument("evidence", type=Path, help="evidence directory (manifest.json)")
    ap.add_argument("verification", type=Path, help="the verifier's verification.json")
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--lock",
        type=Path,
        default=None,
        help="optional check that this file matches the lock recorded in the evidence manifest",
    )
    a = ap.parse_args(argv)
    doc = media(a.adapter, a.configuration, a.evidence, a.verification, a.out, lock=a.lock)
    mcap = sum(t["mcap"]["bytes"] for t in doc["trials"])
    print(f"{len(doc['trials'])} MCAP recordings ({mcap} B), all matching their evidence")
    for t in doc["trials"]:
        if "video" in t:
            print(
                f"trial {t['index']:2d} {t['status']}: {t['video']['path']} {t['video']['bytes']} B"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
