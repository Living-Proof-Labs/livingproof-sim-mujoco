# SPDX-License-Identifier: Apache-2.0
"""Run the pinned hallway fixture checks in MuJoCo and write a reviewer evidence bundle.

Exit 0: every check passed; 1: a check failed; 2: fixture/pin/checksum error;
3: UNSUPPORTED feature. Screenshots use the same CPU (OSMesa) renderer as task render.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

ROOT = Path(__file__).resolve().parents[1]
ORTHO_HEIGHT_M = 5.0  # 960x720 shows 6.67 m x 5.0 m: the whole 6 m hallway


def screenshots(xml, directory):
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.6
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    option = mujoco.MjvOption()
    option.sitegroup[4] = 1  # show named places
    views = {
        # Overhead: orthographic, straight down on the floor centre, +x to the right.
        "overhead.png": dict(
            lookat=[0, 0, 0], distance=9.0, azimuth=90, elevation=-90, orthographic=1
        ),
        # Interior: eye near spawn-entrance at about head height, looking toward exit (+x).
        "entrance-view.png": dict(lookat=[3.0, 0, 0.8], distance=5.6, azimuth=0, elevation=-8),
    }
    written = {}
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for name, view in views.items():
            camera = mujoco.MjvCamera()
            camera.lookat[:] = view["lookat"]
            # MuJoCo 3.17 takes the free camera's projection from vis.global_ (the
            # MjvCamera.orthographic field is ignored for free cameras); in orthographic
            # mode fovy is the visible height in metres.
            ortho = bool(view.get("orthographic"))
            model.vis.global_.orthographic = int(ortho)
            model.vis.global_.fovy = ORTHO_HEIGHT_M if ortho else 45.0
            camera.distance, camera.azimuth, camera.elevation = (
                view["distance"],
                view["azimuth"],
                view["elevation"],
            )
            renderer.update_scene(data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(directory / name)
            written[name] = {
                **view,
                "width": 960,
                "height": 720,
                **(
                    {"projection": "orthographic", "visible_height_m": ORTHO_HEIGHT_M}
                    if view.get("orthographic")
                    else {"projection": "perspective", "fovy_deg": 45.0}
                ),
                "renderer": "MuJoCo offscreen, MUJOCO_GL=" + os.environ["MUJOCO_GL"],
            }
    return written


def main(argv=None):
    from livingproof_sim_mujoco import hallway

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", default="hallway-open")
    parser.add_argument("--content-version", default="0.1.0")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/hallway")
    parser.add_argument("--no-screenshots", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        result = hallway.run_checks(args.fixture, args.content_version)
    except hallway.FixtureError as exc:
        print(f"Fixture error: {exc}", file=sys.stderr)
        return 2
    if result["status"] == "UNSUPPORTED":
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"UNSUPPORTED: {result['fixture']} {result['content_version']}: {result['feature']}",
            file=sys.stderr,
        )
        return 3
    loaded = hallway.load_fixture(args.fixture, args.content_version)
    xml = hallway.to_mjcf(loaded["world"], loaded["extension"])
    (args.out / "model.xml").write_text(xml)
    if args.no_screenshots:
        result["screenshots"] = {"status": "NOT-RUN"}
    else:
        try:
            result["screenshots"] = {"status": "PASS", "files": screenshots(xml, args.out)}
        except Exception as exc:  # report, never fake
            result["screenshots"] = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    versions = {
        k: result[k] for k in ("fixture", "contract", "catalog", "worlds", "engine", "adapter")
    }
    (args.out / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    failed = [q["name"] for q in result["quantities"] if not q["pass"]]
    print(
        f"{result['status']}: {len(result['quantities']) - len(failed)}/{len(result['quantities'])} quantities passed; "
        f"screenshots {result['screenshots']['status']}; evidence in {args.out}"
    )
    for name in failed:
        print(f"FAILED {name}", file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
