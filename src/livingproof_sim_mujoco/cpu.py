# SPDX-License-Identifier: Apache-2.0
"""One CPU kernel path for torch policies on every host.

torch picks its MKL and ATen kernels from the host CPU at import time. The kernels round
differently, so the same TorchScript policy, inputs and seed give observations that differ
in the last digits between, say, an AVX-512 Intel host and a host where MKL takes its SSE4.2
path; over a closed-loop trial that grows into a different trace (unitree-g1 issue 7: the
evidence and media jobs of one workflow run landed on different CPUs, so the media
consistency guard refused a replay of the same run). Pinning the dispatch before torch loads
makes evidence and replay byte-identical wherever they run. The values reproduce the hosted
evidence already published (SSE4.2 MKL, AVX2 ATen; ATen falls back on hosts without AVX2).
"""

from __future__ import annotations

import os
import sys

PINNED = {"MKL_ENABLE_INSTRUCTIONS": "SSE4_2", "ATEN_CPU_CAPABILITY": "avx2"}


def pin() -> None:
    """Set the pinned dispatch. torch reads it once, at import: if torch is already loaded
    with other settings, refuse rather than produce host-dependent observations."""
    stale = "torch" in sys.modules and any(os.environ.get(k) != v for k, v in PINNED.items())
    if stale:
        raise RuntimeError(
            "torch was imported before livingproof_sim_mujoco pinned its CPU dispatch "
            f"({', '.join(f'{k}={v}' for k, v in PINNED.items())}); import "
            "livingproof_sim_mujoco first or set these in the environment"
        )
    os.environ.update(PINNED)
