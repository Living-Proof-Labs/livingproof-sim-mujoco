# SPDX-License-Identifier: Apache-2.0
"""Shared hallway result contract (livingproof-world-result/0.2), vendored from robot-capability-specs.

The schema, required quantity names and probe parameters are copied unchanged from
the specs commit in contract/pins.json and checked by SHA-256 before use. This module
is identical in livingproof-sim-mujoco and livingproof-sim-gazebo apart from PACKAGE.
"""

from __future__ import annotations

from functools import cache
import hashlib
import json
from pathlib import Path
import re
import subprocess

PACKAGE = "livingproof_sim_mujoco"
CONTRACT = Path(__file__).parent / "contract"
RESULT_FORMAT = "livingproof-world-result/0.2"
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
SHA = re.compile(r"^[0-9a-f]{40}$")
RESET_PREFIX = "reset/"
RESET = {"kind": "reset", "unit": "m", "comparison": "within"}


class ContractError(Exception):
    """The vendored contract files are missing, altered, or a record does not conform."""


@cache
def pin() -> dict:
    return json.loads((CONTRACT / "pins.json").read_text())


@cache
def load(name: str) -> dict:
    """Return one vendored contract file after checking its pinned SHA-256."""
    p = pin()
    expected = p["files"].get(name)
    if expected is None:
        raise ContractError(f"{name} is not pinned in contract/pins.json")
    path = CONTRACT / name
    if not path.is_file():
        raise ContractError(f"missing vendored contract file {name}")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise ContractError(
            f"checksum mismatch for {name}: expected {expected} "
            f"({p['repository']} {p['commit']}), got {got}"
        )
    return json.loads(path.read_text())


def schema() -> dict:
    return load("hallway-result.schema.json")


def probe() -> dict:
    return load("hallway-probe.json")


def names() -> dict[str, dict]:
    q = load("hallway-quantities.json")
    return {e["name"]: e for e in q["required"] + q["optional"]}


def required_names() -> set[str]:
    return {e["name"] for e in load("hallway-quantities.json")["required"]}


def passes(comparison: str, canonical: float, measured, tolerance: float) -> bool:
    """Exactly the rule in hallways.md and compare-results.mjs; no hidden epsilon."""
    if measured is None:
        return False
    if comparison == "at_most":
        return measured <= canonical + tolerance
    return abs(measured - canonical) <= tolerance


def quantity(name, canonical, measured, tolerance, method, **extra) -> dict:
    """One 0.2 quantity; kind, unit and comparison come from hallway-quantities.json."""
    spec = names().get(name)
    if spec is None:
        if not name.startswith(RESET_PREFIX):
            raise ContractError(f"quantity {name!r} is not in hallway-quantities.json")
        spec = RESET
    canonical, tolerance = float(canonical), float(tolerance)
    measured = None if measured is None else float(measured)
    return {
        "name": name,
        "kind": spec["kind"],
        "unit": spec["unit"],
        "comparison": spec["comparison"],
        "canonical": canonical,
        "measured": measured,
        "deviation": None if measured is None else measured - canonical,
        "tolerance": tolerance,
        "pass": passes(spec["comparison"], canonical, measured, tolerance),
        "method": method,
        **extra,
    }


def validate(record: dict) -> None:
    """Raise ContractError unless the record matches the shared schema and required names."""
    import jsonschema

    errors = sorted(
        jsonschema.Draft202012Validator(schema()).iter_errors(record), key=lambda e: list(e.path)
    )
    if errors:
        raise ContractError(
            "; ".join(f"/{'/'.join(map(str, e.path))}: {e.message}" for e in errors)
        )
    if record["status"] == "UNSUPPORTED":
        return
    got = [q["name"] for q in record["quantities"]]
    missing = sorted(required_names() - set(got))
    if missing:
        raise ContractError(f"missing required quantities: {', '.join(missing)}")
    if len(got) != len(set(got)):
        raise ContractError("duplicate quantity names")


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except OSError:
        return None
    return out if SHA.match(out) else None


def adapter(name: str, package_dir: Path, display: str, source: str, **extra) -> dict:
    """adapter block: semver version, full commit or null, and provenance git/package."""
    if not SEMVER.match(display):
        raise ContractError(f"adapter version {display!r} is not a semantic version")
    record = {"name": name, "version": display, **extra}
    commit, provenance = None, None
    if source == "checkout":
        commit, provenance = _git_head(package_dir.parent.parent), "git"
    elif source == "recorded-build":
        saved = json.loads((package_dir / "_build_provenance.json").read_text())
        commit = saved.get("identity", {}).get("commit")
        commit = commit if isinstance(commit, str) and SHA.match(commit) else None
        provenance = "package"
    record["commit"] = commit  # null only when neither git nor build provenance knows it
    if provenance:
        record["provenance"] = provenance
    return record


def sampling(physics_step: float) -> dict:
    """physics_step, sample_rate and stride: record every stride-th step at sampling_hz.

    The shared sampling_hz must be an integer number of physics steps; otherwise the
    adapter cannot sample at the contract rate and raises ContractError.
    """
    hz = probe()["sampling_hz"]
    ratio = 1.0 / (hz * physics_step)
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > 1e-9:
        raise ContractError(
            f"physics step {physics_step!r} s does not divide the {hz} Hz sample period"
        )
    return {"physics_step": physics_step, "sample_rate": hz, "sample_stride": stride}
