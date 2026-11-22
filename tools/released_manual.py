# SPDX-License-Identifier: Apache-2.0
"""Read-only release and rendered Pages checks; never create or move tags."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys


def validate_release(record, version, tag, revision, root=Path(".")):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) or tag != f"v{version}":
        raise ValueError("Expected a stable release version and its exact v-prefixed tag")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected the full verified commit")
    if (
        record.get("tag_name") != tag
        or record.get("draft") is not False
        or record.get("prerelease") is not False
        or record.get("immutable") is not True
        or not record.get("published_at")
    ):
        raise ValueError("Expected an existing published immutable stable release")
    for ref in ("HEAD", f"refs/tags/{tag}^{{commit}}"):
        actual = subprocess.check_output(["git", "rev-parse", ref], cwd=root, text=True).strip()
        if actual != revision:
            raise ValueError("Release tag and checkout must equal the verified commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Released manual requires a clean checkout")


def validate_site(site, version, revision):
    record = json.loads((site / "provenance.json").read_text())
    if (
        record.get("version") != version
        or record.get("revision") != revision
        or record.get("channel") != "stable"
        or record.get("dirty") is not False
        or record.get("identity", {}).get("package_version") != version
    ):
        raise ValueError("Pages provenance does not match the confirmed release")
    pages = list(site.glob("*.html"))
    if not pages or not (site / "context.html").is_file():
        raise ValueError("Missing rendered manual")
    for page in pages:
        text = page.read_text()
        if f"Docs: {version}" not in text or revision not in text:
            raise ValueError(f"Missing visible release identity: {page}")


if __name__ == "__main__":
    version, tag, revision = (os.environ[key] for key in ("VERSION", "RELEASE_TAG", "GITHUB_SHA"))
    validate_release(json.loads(Path("artifacts/release.json").read_text()), version, tag, revision)
    if "--site" in sys.argv:
        validate_site(Path("artifacts/manual/html"), version, revision)
