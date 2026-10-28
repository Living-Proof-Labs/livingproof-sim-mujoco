# SPDX-License-Identifier: Apache-2.0
"""Development-only offscreen playback of the built-in model, not a live viewer."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

# A CPU renderer works on the supported Linux amd64 development image without
# a display, GPU, Docker socket, or changes to the installed CLI's runtime.
os.environ["MUJOCO_GL"] = "osmesa"

ROOT = Path(__file__).resolve().parents[1]
PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MuJoCo falling sphere playback</title>
<style>
body{font:18px system-ui;max-width:760px;margin:2rem auto;padding:0 1rem;background:#f5f6fa;color:#172033}
img{width:100%;max-width:640px;display:block;background:#111}
button,input{margin:1rem .5rem 1rem 0}input{width:60%}output{display:block}
</style>
<h1>MuJoCo falling sphere</h1>
<p>Rendered playback of 200 native physics steps (0.4 simulation seconds).
This is a recorded run, not live interactive simulation or robot capability evidence.</p>
<p>The built-in sphere falls freely under gravity; there is no floor or contact.</p>
<img id="frame" src="/frame-000.png" alt="MuJoCo render of the falling sphere">
<button id="play" type="button" disabled>Play</button>
<label for="position">Frame</label><input id="position" type="range" min="0" max="40" value="0" disabled>
<output id="state" aria-live="polite">Loading recorded physics…</output>
<p>Playback is slowed to one tenth of simulation speed. Drag the frame slider to inspect it.
Run <code>task preview</code> again to regenerate; Ctrl+C stops the server.</p>
<p><a href="/trajectory.json">Recorded positions and engine identity (JSON)</a></p>
<script>
const image=document.querySelector('#frame'), slider=document.querySelector('#position');
const play=document.querySelector('#play'), state=document.querySelector('#state');
let frames=[], timer=null;
function show(i){slider.value=i;image.src=`/frame-${String(i).padStart(3,'0')}.png`;
 state.textContent=`Simulation time: ${frames[i].time.toFixed(3)} s · z: ${frames[i].z.toFixed(6)} m`;}
function stop(){clearInterval(timer);timer=null;play.textContent='Play';}
slider.oninput=()=>{stop();show(Number(slider.value));};
play.onclick=()=>{if(timer){stop();return;}if(Number(slider.value)===frames.length-1)show(0);
 play.textContent='Pause';timer=setInterval(()=>{const next=Number(slider.value)+1;
 if(next>=frames.length){stop();return;}show(next);if(next===frames.length-1)stop();},100);};
fetch('/trajectory.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(data=>{
 frames=data.frames;slider.max=frames.length-1;slider.disabled=false;play.disabled=false;show(0);
}).catch(()=>{state.textContent='Could not load playback. Restart task preview and reload.';});
</script></html>
"""


def render(directory):
    import mujoco
    import numpy as np
    from PIL import Image

    from livingproof_sim_mujoco.cli import identity

    directory.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "src/livingproof_sim_mujoco/models/falling.xml")
    )
    # Visual styling only; keep the built-in model's dynamics unchanged.
    model.geom_rgba[0] = [0.2, 0.7, 1.0, 1.0]
    model.vis.headlight.ambient[:] = 0.4
    model.vis.headlight.diffuse[:] = 0.8
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.6]
    camera.distance = 2.0
    camera.azimuth = 90
    camera.elevation = -10
    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for step in range(201):
            if step:
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                raise ValueError("MuJoCo produced a warning or non-finite position")
            if step % 5 == 0:
                # mj_step leaves derived positions at the prior integration state.
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(directory / f"frame-{len(frames):03}.png")
                frames.append({"step": step, "time": float(data.time), "z": float(data.qpos[2])})
    (directory / "trajectory.json").write_text(
        json.dumps({"model": "builtin:falling", "identity": identity(), "frames": frames}, indent=2)
        + "\n"
    )
    (directory / "index.html").write_text(PAGE)
    return {
        "/": (directory / "index.html", "text/html; charset=utf-8"),
        "/trajectory.json": (directory / "trajectory.json", "application/json"),
        **{
            f"/frame-{i:03}.png": (directory / f"frame-{i:03}.png", "image/png")
            for i in range(len(frames))
        },
    }


def server(assets, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asset = assets.get(urlsplit(self.path).path)
            if asset is None:
                self.send_error(404)
                return
            path, content_type = asset
            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

    # Only generated assets are served, never the checkout, its .git, or caches.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="local HTTP port (default: 8000)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("expected a port from 1 to 65535, for example --port 8000")
    try:
        print("Rendering 0.4 seconds of built-in physics with CPU graphics…", flush=True)
        assets = render(ROOT / "artifacts/preview")
        with server(assets, args.port) as httpd:
            print(f"Playback ready: http://127.0.0.1:{args.port}/", flush=True)
            print(
                "Codespaces: open this port in the Ports tab; keep visibility Private. Ctrl+C stops.",
                flush=True,
            )
            httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(
            f"Preview failed: {exc}. Run task setup; rebuild the development container "
            "if OSMesa is missing. If the port is busy, stop its server or use --port 8001.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
