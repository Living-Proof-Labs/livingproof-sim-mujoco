# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-native headless execution, with an outer process deadline."""

import argparse
import json
import math
import multiprocessing as mp
import signal
import sys
import time
from importlib.metadata import version
from pathlib import Path

from livingproof_common.versioning import Consumer, current_version


def identity():
    display, package, source = current_version(
        Consumer("livingproof-sim-mujoco", Path(__file__).parent)
    )
    return {
        "version": display,
        "package_version": package,
        "provenance": source,
        "mujoco": version("mujoco"),
        "livingproof_common": version("livingproof-common"),
    }


def positive_int(value):
    try:
        result = int(value)
        if result > 0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected a positive integer, for example --steps 100")


def positive_seconds(value):
    try:
        result = float(value)
        if math.isfinite(result) and result > 0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected finite positive seconds, for example --timeout 10")


def parser():
    p = argparse.ArgumentParser(
        prog="livingproof-sim-mujoco",
        description="Load an MJCF XML or compiled MJB model and advance native MuJoCo physics.",
        epilog="Example: livingproof-sim-mujoco --model builtin:falling --steps 100 --headless. "
        "Outputs one JSON result. Exit 0: requested steps completed; 2: input/engine error; "
        "124: wall timeout; 130/143: SIGINT/SIGTERM. Physics execution is not a robot "
        "capability PASS. Use --version for engine and build provenance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="builtin:falling",
        help="MJCF .xml or MJB .mjb path; includes resolve relative to XML; or builtin:falling",
    )
    p.add_argument(
        "--steps", type=positive_int, default=100, help="maximum number of mj_step calls"
    )
    p.add_argument(
        "--timeout",
        type=positive_seconds,
        default=10.0,
        help="wall seconds including worker startup/model loading; shutdown grace is at most 2 additional seconds",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="run without viewer or graphics context (the only supported mode)",
    )
    p.add_argument(
        "--version",
        action="store_true",
        help="print tool, shared-library and engine versions as JSON, then exit",
    )
    return p


def worker(connection, stop, model_path, steps):
    # The supervisor owns signal semantics, including during native model loading.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import mujoco
        import numpy as np

        path = Path(model_path)
        model = (
            mujoco.MjModel.from_binary_path(str(path))
            if path.suffix.lower() == ".mjb"
            else mujoco.MjModel.from_xml_path(str(path))
        )
        data = mujoco.MjData(model)
        initial = data.qpos[:16].tolist()
        connection.send(
            {
                "event": "loaded",
                "nq": model.nq,
                "nv": model.nv,
                "timestep": model.opt.timestep,
                "engine": mujoco.mj_versionString(),
            }
        )
        completed = 0
        while completed < steps and not stop.is_set():
            before = data.time
            mujoco.mj_step(model, data)
            completed += 1
            if (
                not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
                or not math.isfinite(data.time)
                or data.time <= before
                or any(w.number for w in data.warning)
            ):
                raise ValueError(
                    "MuJoCo reported a warning, non-finite state, or non-advancing time"
                )
        connection.send(
            {
                "event": "result",
                "status": "completed" if completed == steps else "interrupted",
                "steps": completed,
                "simulation_time": data.time,
                "initial_qpos": initial,
                "final_qpos": data.qpos[:16].tolist(),
                "qpos_truncated": model.nq > 16,
            }
        )
    except Exception as exc:
        connection.send(
            {
                "event": "result",
                "status": "error",
                "diagnostic": str(exc)[:2000],
                "hint": "Check the MJCF/MJB model and its assets; try --model builtin:falling --steps 100.",
            }
        )
    finally:
        connection.close()


def run(args):
    path = (
        Path(__file__).parent / "models/falling.xml"
        if args.model == "builtin:falling"
        else Path(args.model)
    )
    if path.suffix.lower() not in {".xml", ".mjb"} or not path.is_file():
        print(
            "Expected an existing .xml/.mjb model or builtin:falling. "
            "Try: livingproof-sim-mujoco --model builtin:falling --steps 100",
            file=sys.stderr,
        )
        return 2
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    stop = ctx.Event()
    process = ctx.Process(target=worker, args=(send, stop, str(path.resolve()), args.steps))
    received_signal = []
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(
            signum, lambda signum, frame: received_signal.append(signum)
        )
    start = time.monotonic()
    result = None
    loaded = None
    reason = None
    try:
        process.start()
        send.close()
        while True:
            if received_signal:
                reason = "signal"
                break
            if time.monotonic() - start >= args.timeout:
                reason = "timeout"
                break
            if receive.poll(0.01):
                try:
                    message = receive.recv()
                except EOFError:
                    break
                if message["event"] == "loaded":
                    loaded = message
                    print("MuJoCo model loaded; stepping headlessly", file=sys.stderr, flush=True)
                else:
                    result = message
                    break
            if not process.is_alive() and not receive.poll():
                break
        stop.set()
        process.join(1)
        if process.is_alive():
            process.terminate()
            process.join(0.5)
        if process.is_alive():
            process.kill()
            process.join(0.5)
        if result is None and receive.poll():
            try:
                message = receive.recv()
                if message["event"] == "result":
                    result = message
            except EOFError:
                pass
        result = result or {
            "status": "error",
            "diagnostic": "Worker exited before returning physics results",
        }
        if reason:
            result["status"] = reason
        result.update(
            model=args.model,
            requested_steps=args.steps,
            loaded=loaded,
            identity=identity(),
            worker_exitcode=process.exitcode,
            wall_seconds=time.monotonic() - start,
            evidence="physics-execution-only",
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        if reason == "signal":
            return 128 + received_signal[0]
        if reason == "timeout":
            return 124
        return 0 if result["status"] == "completed" and process.exitcode == 0 else 2
    finally:
        if process.is_alive():
            process.kill()
            process.join(0.5)
        receive.close()
        send.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    args = parser().parse_args()
    if args.version:
        print(json.dumps(identity(), sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
