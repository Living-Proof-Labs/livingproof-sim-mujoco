# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the development-only rendered playback."""

import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from playwright.sync_api import expect, sync_playwright

spec = importlib.util.spec_from_file_location("preview", Path("tools/preview.py"))
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


@pytest.fixture(scope="module")
def playback(tmp_path_factory):
    directory = tmp_path_factory.mktemp("playback")
    assets = preview.render(directory)
    with preview.server(assets, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", directory
        finally:
            server.shutdown()
            thread.join()


def test_recorded_physics_and_real_frames(playback):
    """Given builtin:falling, rendering records real steps and visibly different frames."""
    from PIL import Image, ImageChops

    base, directory = playback
    with urlopen(base + "/trajectory.json") as response:
        record = json.load(response)
    frames = record["frames"]
    assert len(frames) == 41
    assert frames[0] == {"step": 0, "time": 0.0, "z": 1.0}
    assert frames[-1]["step"] == 200
    assert frames[-1]["time"] == pytest.approx(0.4)
    assert frames[-1]["z"] == pytest.approx(0.211276)
    assert all(a["z"] > b["z"] for a, b in zip(frames, frames[1:]))
    first, last = (Image.open(directory / f"frame-{i:03}.png") for i in (0, 40))
    assert first.size == last.size == (640, 480)
    assert ImageChops.difference(first, last).getbbox() is not None


def test_server_serves_only_generated_assets(playback):
    """Given a checkout, HTTP exposes only the generated playback assets."""
    base, _ = playback
    for path in ("/.git/config", "/../pyproject.toml", "/%2e%2e/pyproject.toml", "/index.html"):
        with pytest.raises(HTTPError) as error:
            urlopen(base + path)
        assert error.value.code == 404


def test_browser_play_pause_scrub_and_replay(playback):
    """Given rendered physics over HTTP, a browser can play, pause, scrub and replay."""
    base, _ = playback
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base)
        expect(page.get_by_role("heading")).to_have_text("MuJoCo falling sphere")
        expect(page.locator("body")).to_contain_text("not live interactive simulation")
        expect(page.locator("#state")).to_contain_text("z: 1.000000")
        page.wait_for_function("document.querySelector('#frame').naturalWidth === 640")
        page.get_by_role("button", name="Play", exact=True).click()
        page.wait_for_function("Number(document.querySelector('#position').value) >= 2")
        page.get_by_role("button", name="Pause", exact=True).click()
        value = page.locator("#position").input_value()
        page.wait_for_timeout(250)
        assert page.locator("#position").input_value() == value
        page.locator("#position").fill("40")
        expect(page.locator("#state")).to_contain_text("z: 0.211276")
        page.get_by_role("button", name="Play", exact=True).click()
        page.wait_for_function("Number(document.querySelector('#position').value) < 40")
        page.get_by_role("button", name="Pause", exact=True).click()
        page.locator("#position").fill("0")
        Path("artifacts").mkdir(exist_ok=True)
        page.screenshot(path="artifacts/preview-browser.png", full_page=True)
        assert errors == []
        browser.close()


def test_task_preview_first_run_and_shutdown():
    """Given the prepared container, task preview serves the advertised URL and stops on Ctrl+C."""
    import os
    import signal
    import subprocess
    import time
    from urllib.error import URLError

    with Path("artifacts/preview-command.log").open("w") as log:
        process = subprocess.Popen(
            ["task", "preview"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                assert process.poll() is None, Path("artifacts/preview-command.log").read_text()
                try:
                    with urlopen("http://127.0.0.1:8000/", timeout=1) as response:
                        assert b"MuJoCo falling sphere" in response.read()
                    break
                except URLError:
                    assert time.monotonic() < deadline, "task preview did not become ready"
                    time.sleep(0.1)
        finally:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                pytest.fail("task preview did not stop after Ctrl+C")
    assert (
        "Playback ready: http://127.0.0.1:8000/"
        in Path("artifacts/preview-command.log").read_text()
    )
