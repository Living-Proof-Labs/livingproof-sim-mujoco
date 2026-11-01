# SPDX-License-Identifier: Apache-2.0
"""Re-pin the vendored hallway contract to a robot-capability-specs commit.

Usage: python tools/pin_contract.py [--check] /path/to/robot-capability-specs [REF]
REF is a release tag (preferred, e.g. v1.4.0) or a commit; default HEAD. Copies the three
contract files exactly as committed at REF and rewrites src/livingproof_sim_mujoco/contract/pins.json,
the single place holding the tag, commit and SHA-256s. With --check nothing is written;
it exits 1 unless the vendored files and pins.json match REF. Re-run task test afterwards.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

FILES = ["hallway-result.schema.json", "hallway-quantities.json", "hallway-probe.json"]
DEST = Path(__file__).resolve().parents[1] / "src/livingproof_sim_mujoco/contract"


def main(argv):
    check = argv[:1] == ["--check"]
    argv = argv[1:] if check else argv
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    specs, ref = Path(argv[0]), argv[1] if len(argv) == 2 else "HEAD"

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(specs), *args], capture_output=True, check=True
        ).stdout

    commit = git("rev-parse", f"{ref}^{{commit}}").decode().strip()
    tags = git("tag", "--list", ref).decode().split()
    old = json.loads((DEST / "pins.json").read_text())
    pin = {"repository": old["repository"]}
    pin.update({"tag": ref} if tags else {"branch": old.get("branch")})
    pin.update(commit=commit, path=old["path"], files={})
    blobs = {}
    for name in FILES:
        blobs[name] = git("show", f"{commit}:{pin['path']}/{name}")
        pin["files"][name] = hashlib.sha256(blobs[name]).hexdigest()
    if check:
        bad = [n for n in FILES if (DEST / n).read_bytes() != blobs[n]]
        if old != pin:
            bad.append("pins.json")
        print(
            f"{'MISMATCH ' + ', '.join(bad) if bad else 'OK'}: vendored contract vs {ref} ({commit})"
        )
        return 1 if bad else 0
    for name in FILES:
        (DEST / name).write_bytes(blobs[name])
    (DEST / "pins.json").write_text(json.dumps(pin, indent=2) + "\n")
    print(f"pinned {pin['repository']} {ref} {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
