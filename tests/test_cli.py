# SPDX-License-Identifier: Apache-2.0
"""Given a real MuJoCo model, when stepped, then observable physics advances."""

import json
import os
import signal
import subprocess
import time

import pytest

CLI = os.environ.get("TEST_CLI", "livingproof-sim-mujoco")


def invoke(*args):
    return subprocess.run([CLI, *args], capture_output=True, text=True, timeout=15)


def test_falling_body_advances():
    result = invoke("--steps", "100", "--headless")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["loaded"]["engine"] == "3.17.0"
    assert data["steps"] == 100
    assert data["simulation_time"] == pytest.approx(0.2)
    assert data["final_qpos"][2] == pytest.approx(0.801838, abs=1e-5)
    assert data["final_qpos"][2] < data["initial_qpos"][2]
    assert data["evidence"] == "physics-execution-only"
    assert data["worker_exitcode"] == 0


def test_xml_relative_include(tmp_path):
    (tmp_path / "body.xml").write_text(
        '<mujocoinclude><body><joint type="slide" axis="0 0 1"/><geom size=".1"/></body></mujocoinclude>'
    )
    (tmp_path / "scene.xml").write_text(
        '<mujoco><worldbody><include file="body.xml"/></worldbody></mujoco>'
    )
    result = invoke("--model", str(tmp_path / "scene.xml"), "--steps", "3")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["steps"] == 3


def test_compiled_mjb(tmp_path):
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    path = tmp_path / "model.mjb"
    mujoco.mj_saveModel(model, str(path))
    assert invoke("--model", str(path)).returncode == 0


@pytest.mark.parametrize(
    "args",
    [
        ("--steps", "0"),
        ("--steps", "-1"),
        ("--timeout", "nan"),
        ("--timeout", "inf"),
        ("--timeout", "0"),
        ("--model", "absent.xml"),
        ("--unknown",),
    ],
)
def test_invalid_input_teaches_recovery(args):
    result = invoke(*args)
    assert result.returncode == 2
    assert result.stderr
    assert "Traceback" not in result.stderr


def test_malformed_model(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text("<mujoco><invalid/></mujoco>")
    result = invoke("--model", str(path))
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "error"
    assert "builtin:falling" in json.loads(result.stdout)["hint"]


def test_timeout_during_startup():
    start = time.monotonic()
    result = invoke("--timeout", "0.001")
    assert result.returncode == 124
    assert json.loads(result.stdout)["status"] == "timeout"
    assert time.monotonic() - start < 4


def test_timeout_during_physics():
    result = invoke("--steps", "1000000000", "--timeout", "1")
    assert result.returncode == 124
    data = json.loads(result.stdout)
    assert data["loaded"]
    assert data["steps"] > 0
    assert data["worker_exitcode"] == 0


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_shutdown(signum):
    process = subprocess.Popen(
        [CLI, "--steps", "1000000000", "--timeout", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        import selectors

        with selectors.DefaultSelector() as selector:
            selector.register(process.stderr, selectors.EVENT_READ)
            assert selector.select(timeout=8), "worker did not load"
        assert "loaded" in process.stderr.readline()
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=4)
        assert process.returncode == 128 + signum, stderr
        data = json.loads(stdout)
        assert data["status"] == "signal"
        assert data["worker_exitcode"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_help_and_version():
    assert "capability PASS" in invoke("--help").stdout
    data = json.loads(invoke("--version").stdout)
    assert data["livingproof_common"] == "1.0.0"
    assert data["mujoco"] == "3.17.0"
    if os.environ.get("INSTALLED_PRODUCT"):
        assert data["provenance"] == "recorded-build"


def test_reporting_failure_fixture():
    assert os.environ.get("REPORTING_FAILURE_FIXTURE") != "1", (
        "Deliberate reporting fixture failure"
    )
