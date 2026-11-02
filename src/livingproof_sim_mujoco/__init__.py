# SPDX-License-Identifier: Apache-2.0
from livingproof_sim_mujoco import cpu as _cpu

# Before anything can import torch: one CPU kernel path on every host (see cpu.py).
_cpu.pin()
