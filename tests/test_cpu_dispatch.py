# SPDX-License-Identifier: Apache-2.0
"""Evidence and replay must agree across hosts (unitree-g1 issue 7). The media guard still
refuses a replay that really differs: see test_media."""

import os
import subprocess
import sys

import pytest

from livingproof_sim_mujoco import cpu

pytest.importorskip("torch")

# A closed loop through a TorchScript MLP, as a policy drives a trial: last-digit rounding
# differences between MKL kernels compound, exactly as in the G1 stock trace.
LOOP = """
import hashlib, sys
{first}
import torch
torch.manual_seed(0)
net = torch.jit.script(torch.nn.Sequential(torch.nn.Linear(47, 512), torch.nn.ELU(),
                       torch.nn.Linear(512, 256), torch.nn.ELU(), torch.nn.Linear(256, 47)))
torch.set_num_threads(1)
x = torch.rand(1, 47)
with torch.no_grad():
    for _ in range(400):
        x = torch.tanh(net(x) + 0.9 * x)
print(hashlib.sha256(x.numpy().tobytes()).hexdigest())
"""


def _loop(env_isa: str, *, pinned: bool) -> str:
    first = "import livingproof_sim_mujoco" if pinned else ""
    env = {k: v for k, v in os.environ.items() if k not in cpu.PINNED}
    env["MKL_ENABLE_INSTRUCTIONS"] = env_isa  # what this host would pick on its own
    out = subprocess.run(
        [sys.executable, "-c", LOOP.format(first=first)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_pinned_dispatch_gives_one_result_whatever_the_host_picks():
    assert (
        _loop("AVX512", pinned=True) == _loop("SSE4_2", pinned=True) == _loop("AVX2", pinned=True)
    )


def test_unpinned_dispatch_really_differs_on_this_host():
    # The control: without the pin the same loop differs between kernel paths, so the test
    # above proves the pin, not a lucky host. Hosts without AVX2 have one path only.
    paths = {_loop(isa, pinned=False) for isa in ("AVX512", "AVX2", "SSE4_2")}
    if len(paths) == 1:
        pytest.skip("this CPU offers one MKL kernel path only")
    assert len(paths) > 1


def test_torch_loaded_with_another_dispatch_is_refused():
    code = (
        "import os; os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX512'; import torch; "
        "import livingproof_sim_mujoco"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k not in cpu.PINNED},
    )
    assert r.returncode != 0 and "pinned its CPU dispatch" in r.stderr
