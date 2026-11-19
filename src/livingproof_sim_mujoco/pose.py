# SPDX-License-Identifier: Apache-2.0
"""Pose API and MuJoCo pose provider for headless evaluation and route control.

Provides explicit pose readout (``get_pose``, ``MujocoPoseProvider.read``),
explicit pose placement/teleportation (``set_pose``, ``set_pose_at_place``),
a typed ``PoseSample`` value with provenance metadata, and explicit failure
errors (``PoseUnavailableError``) for missing, stale, wrong-frame, or
non-finite poses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from livingproof_sim_mujoco.scene import PLACE

DEFAULT_FRAME_ID = "world"
DEFAULT_SOURCE = "simulator-ideal"
DEFAULT_CLOCK = "simulation"

__all__ = [
    "DEFAULT_CLOCK",
    "DEFAULT_FRAME_ID",
    "DEFAULT_SOURCE",
    "MujocoPoseProvider",
    "PoseProvider",
    "PoseSample",
    "PoseUnavailableError",
    "get_pose",
    "normalize_quat",
    "quat_to_yaw",
    "set_pose",
    "set_pose_at_place",
    "validate_pose_sample",
    "yaw_to_quat",
]


class PoseUnavailableError(RuntimeError):
    """Raised when a required pose is missing, stale, wrong-frame, or non-finite."""


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """World yaw (rad, about +z) from quaternion ``(w, x, y, z)``."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Normalized upright ``(w, x, y, z)`` quaternion for heading ``yaw`` (rad)."""
    if not math.isfinite(yaw):
        raise PoseUnavailableError(f"Requested yaw is non-finite ({yaw!r}).")
    half = 0.5 * float(yaw)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def normalize_quat(
    quat: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    """Normalize ``(w, x, y, z)`` quaternion; raise ``PoseUnavailableError`` if invalid."""
    if len(quat) != 4:
        raise PoseUnavailableError(
            f"Quaternion must have 4 components (w, x, y, z), got length {len(quat)}."
        )
    vals = tuple(float(v) for v in quat)
    if not all(math.isfinite(v) for v in vals):
        raise PoseUnavailableError(f"Quaternion contains non-finite values: {vals!r}.")
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 0.0:
        raise PoseUnavailableError(f"Quaternion has zero norm: {vals!r}.")
    return (vals[0] / norm, vals[1] / norm, vals[2] / norm, vals[3] / norm)


@dataclass(frozen=True)
class PoseSample:
    """Typed world-frame pose with orientation and explicit provenance metadata."""

    x: float
    y: float
    z: float
    yaw: float
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    frame_id: str = DEFAULT_FRAME_ID
    timestamp_s: float = 0.0
    source: str = DEFAULT_SOURCE
    clock: str = DEFAULT_CLOCK

    def __post_init__(self) -> None:
        numeric = (
            self.x,
            self.y,
            self.z,
            self.yaw,
            self.qw,
            self.qx,
            self.qy,
            self.qz,
            self.timestamp_s,
        )
        if not all(math.isfinite(float(v)) for v in numeric):
            raise PoseUnavailableError(
                f"PoseSample contains non-finite numeric fields from source {self.source!r}."
            )
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise PoseUnavailableError("PoseSample.frame_id must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source:
            raise PoseUnavailableError("PoseSample.source must be a non-empty string.")
        if not isinstance(self.clock, str) or not self.clock:
            raise PoseUnavailableError("PoseSample.clock must be a non-empty string.")

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def xy_yaw(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    @property
    def orientation(self) -> tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)

    @classmethod
    def from_quaternion(
        cls,
        x: float,
        y: float,
        z: float,
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        *,
        frame_id: str = DEFAULT_FRAME_ID,
        timestamp_s: float = 0.0,
        source: str = DEFAULT_SOURCE,
        clock: str = DEFAULT_CLOCK,
    ) -> PoseSample:
        w, vx, vy, vz = normalize_quat((qw, qx, qy, qz))
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            yaw=quat_to_yaw(w, vx, vy, vz),
            qw=w,
            qx=vx,
            qy=vy,
            qz=vz,
            frame_id=frame_id,
            timestamp_s=float(timestamp_s),
            source=source,
            clock=clock,
        )

    @classmethod
    def from_yaw(
        cls,
        x: float,
        y: float,
        z: float = 0.0,
        yaw: float = 0.0,
        *,
        frame_id: str = DEFAULT_FRAME_ID,
        timestamp_s: float = 0.0,
        source: str = DEFAULT_SOURCE,
        clock: str = DEFAULT_CLOCK,
    ) -> PoseSample:
        w, vx, vy, vz = yaw_to_quat(yaw)
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            yaw=float(yaw),
            qw=w,
            qx=vx,
            qy=vy,
            qz=vz,
            frame_id=frame_id,
            timestamp_s=float(timestamp_s),
            source=source,
            clock=clock,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "orientation": {"w": self.qw, "x": self.qx, "y": self.qy, "z": self.qz},
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "source": self.source,
            "clock": self.clock,
        }


@runtime_checkable
class PoseProvider(Protocol):
    """Protocol for an engine-backed or synthetic pose source."""

    source: str
    frame_id: str

    def read(self, data: Any = None) -> PoseSample: ...


def validate_pose_sample(
    sample: PoseSample | None,
    *,
    expected_frame: str = DEFAULT_FRAME_ID,
    max_age_s: float | None = None,
    now_s: float | None = None,
) -> PoseSample:
    """Validate frame, finiteness, and freshness of a ``PoseSample``."""
    if sample is None:
        raise PoseUnavailableError("Pose provider returned None; no pose sample is available.")
    if sample.frame_id != expected_frame:
        raise PoseUnavailableError(
            f"Pose sample frame_id {sample.frame_id!r} does not match expected frame "
            f"{expected_frame!r} (source={sample.source!r})."
        )
    if max_age_s is not None and now_s is not None:
        age = float(now_s) - float(sample.timestamp_s)
        if age > float(max_age_s):
            raise PoseUnavailableError(
                f"Pose sample from {sample.source!r} is stale (age {age:.4f}s > "
                f"max_age_s={float(max_age_s):.4f}s)."
            )
    return sample


def _resolve_body_id(model: mujoco.MjModel, body: str | int) -> int:
    if isinstance(body, (int, np.integer)):
        body_id = int(body)
        if 0 <= body_id < model.nbody:
            return body_id
        raise PoseUnavailableError(f"Body id {body_id} is out of range [0, {model.nbody}).")
    if isinstance(body, str):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id >= 0:
            return int(body_id)
        raise PoseUnavailableError(f"Body {body!r} does not exist in compiled MuJoCo model.")
    raise PoseUnavailableError(f"Body must be a name or id, got {type(body).__name__}.")


def _freejoint_addresses(model: mujoco.MjModel, body_id: int) -> tuple[int, int]:
    jnt_start = int(model.body_jntadr[body_id])
    jnt_count = int(model.body_jntnum[body_id])
    for offset in range(jnt_count):
        jnt_id = jnt_start + offset
        if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jnt_id]), int(model.jnt_dofadr[jnt_id])
    body_name = model.body(body_id).name
    raise PoseUnavailableError(
        f"Body {body_name!r} has no free joint; world pose cannot be set via qpos."
    )


def get_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: str | int,
    *,
    frame_id: str = DEFAULT_FRAME_ID,
    source: str = DEFAULT_SOURCE,
) -> PoseSample:
    """Query the current world-frame pose ``(x, y, z, yaw, quat)`` of ``body``."""
    if data is None:
        raise PoseUnavailableError("MuJoCo data is None; cannot query body pose.")
    body_id = _resolve_body_id(model, body)
    pos = np.asarray(data.xpos[body_id], dtype=float)
    quat = np.asarray(data.xquat[body_id], dtype=float)
    t = float(data.time)
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat)) and math.isfinite(t)):
        body_name = model.body(body_id).name
        raise PoseUnavailableError(
            f"Non-finite state while reading pose for body {body_name!r} at t={t!r}."
        )
    return PoseSample.from_quaternion(
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
        frame_id=frame_id,
        timestamp_s=t,
        source=source,
    )


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: str | int,
    *,
    x: float,
    y: float,
    z: float | None = None,
    yaw: float = 0.0,
    quat: tuple[float, float, float, float] | list[float] | None = None,
    zero_vel: bool = True,
    forward: bool = True,
    frame_id: str = DEFAULT_FRAME_ID,
    source: str = DEFAULT_SOURCE,
) -> PoseSample:
    """Teleport ``body`` (via its free joint) to ``(x, y, z, yaw/quat)`` and return the pose."""
    if data is None:
        raise PoseUnavailableError("MuJoCo data is None; cannot set body pose.")
    body_id = _resolve_body_id(model, body)
    qadr, dofadr = _freejoint_addresses(model, body_id)

    xf, yf = float(x), float(y)
    if not (math.isfinite(xf) and math.isfinite(yf)):
        raise PoseUnavailableError(f"Requested pose position (x={x!r}, y={y!r}) is non-finite.")
    zf = float(data.qpos[qadr + 2]) if z is None else float(z)
    if not math.isfinite(zf):
        raise PoseUnavailableError(f"Pose height z={zf!r} is non-finite.")

    w, vx, vy, vz = normalize_quat(quat) if quat is not None else yaw_to_quat(float(yaw))

    data.qpos[qadr : qadr + 3] = [xf, yf, zf]
    data.qpos[qadr + 3 : qadr + 7] = [w, vx, vy, vz]
    if zero_vel:
        data.qvel[dofadr : dofadr + 6] = 0.0
    if forward:
        mujoco.mj_forward(model, data)
        return get_pose(model, data, body_id, frame_id=frame_id, source=source)
    return PoseSample.from_quaternion(
        xf,
        yf,
        zf,
        w,
        vx,
        vy,
        vz,
        frame_id=frame_id,
        timestamp_s=float(data.time),
        source=source,
    )


def set_pose_at_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body: str | int,
    place_name: str,
    *,
    z: float | None = None,
    offset: dict[str, Any] | None = None,
    zero_vel: bool = True,
    forward: bool = True,
    frame_id: str = DEFAULT_FRAME_ID,
    source: str = DEFAULT_SOURCE,
) -> PoseSample:
    """Set ``body`` pose at a compiled world place site ``place:<place_name>``."""
    try:
        site = model.site(PLACE + place_name)
    except KeyError:
        raise PoseUnavailableError(
            f"Place site {PLACE + place_name!r} does not exist on the compiled model."
        ) from None
    pos = np.asarray(model.site_pos[site.id], dtype=float)
    w, _, _, sz = (float(v) for v in model.site_quat[site.id])
    yaw = 2.0 * math.atan2(sz, w)
    dx, dy, dyaw = 0.0, 0.0, 0.0
    if offset:
        if "base_xy_m" in offset:
            raw_xy = offset["base_xy_m"]
            dx, dy = float(raw_xy[0]), float(raw_xy[1])
        if "base_yaw_rad" in offset:
            dyaw = float(offset["base_yaw_rad"])
    return set_pose(
        model,
        data,
        body,
        x=float(pos[0]) + dx,
        y=float(pos[1]) + dy,
        z=z,
        yaw=yaw + dyaw,
        zero_vel=zero_vel,
        forward=forward,
        frame_id=frame_id,
        source=source,
    )


class MujocoPoseProvider:
    """MuJoCo-backed pose provider with explicit provenance and get/set helpers."""

    def __init__(
        self,
        model: mujoco.MjModel,
        base_body: str | int,
        *,
        source: str = DEFAULT_SOURCE,
        frame_id: str = DEFAULT_FRAME_ID,
        max_age_s: float | None = None,
    ) -> None:
        self.model = model
        self.body_id = _resolve_body_id(model, base_body)
        self.base_body = model.body(self.body_id).name
        self.source = source
        self.frame_id = frame_id
        self.max_age_s = max_age_s

    def read(self, data: mujoco.MjData) -> PoseSample:
        sample = self.get_pose(data)
        return validate_pose_sample(
            sample,
            expected_frame=self.frame_id,
            max_age_s=self.max_age_s,
            now_s=float(data.time),
        )

    def get_pose(self, data: mujoco.MjData) -> PoseSample:
        return get_pose(
            self.model,
            data,
            self.body_id,
            frame_id=self.frame_id,
            source=self.source,
        )

    def set_pose(
        self,
        data: mujoco.MjData,
        *,
        x: float,
        y: float,
        z: float | None = None,
        yaw: float = 0.0,
        quat: tuple[float, float, float, float] | list[float] | None = None,
        zero_vel: bool = True,
        forward: bool = True,
    ) -> PoseSample:
        return set_pose(
            self.model,
            data,
            self.body_id,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            quat=quat,
            zero_vel=zero_vel,
            forward=forward,
            frame_id=self.frame_id,
            source=self.source,
        )

    def set_pose_at_place(
        self,
        data: mujoco.MjData,
        place_name: str,
        *,
        z: float | None = None,
        offset: dict[str, Any] | None = None,
        zero_vel: bool = True,
        forward: bool = True,
    ) -> PoseSample:
        return set_pose_at_place(
            self.model,
            data,
            self.body_id,
            place_name,
            z=z,
            offset=offset,
            zero_vel=zero_vel,
            forward=forward,
            frame_id=self.frame_id,
            source=self.source,
        )
