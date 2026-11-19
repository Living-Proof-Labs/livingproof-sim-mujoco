# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed media extra, with no source checkout or dev lock."""

from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import imageio.v2 as imageio
from mcap.reader import make_reader
import mujoco
import numpy as np
from PIL import Image

from livingproof_sim_mujoco.media import Recorder


assert int(version("pillow").split(".")[0]) >= 13
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    pixels = np.full((32, 32, 3), [10, 80, 160], dtype=np.uint8)
    png = root / "generated.png"
    imageio.imwrite(png, pixels, format="PNG")
    assert np.array_equal(imageio.imread(png), pixels)
    with Image.open(png) as image:
        assert image.format == "PNG"
        assert np.array_equal(np.asarray(image), pixels)

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="ball" pos="0 0 1">'
        '<freejoint/><geom type="sphere" size="0.1"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    recording = root / "trial.mcap"
    video = root / "trial.mp4"
    recorder = Recorder(model, "ball", mcap_path=recording, video_path=video)
    for step in range(100):
        mujoco.mj_step(model, data)
        recorder(step, data)
    recorder.close()
    with recording.open("rb") as stream:
        topics = {channel.topic for _, channel, _ in make_reader(stream).iter_messages()}
    assert topics == {"/base_pose", "/joint_states"}
    with imageio.get_reader(video) as reader:
        frame = reader.get_data(0)
        assert frame.ndim == 3 and frame.shape[2] == 3
    assert video.stat().st_size > 0

print(json.dumps({name: version(name) for name in ("pillow", "imageio", "imageio-ffmpeg", "mcap")}))
print("Installed wheel: PNG round-trip, MuJoCo rendering, MP4 decoding and MCAP topics passed")
