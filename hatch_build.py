# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from livingproof_common.versioning import Consumer, record_build


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        consumer = Consumer(
            "livingproof-sim-mujoco", Path(self.root) / "src/livingproof_sim_mujoco"
        )
        build_data.setdefault("artifacts", []).append(
            record_build(self.root, self.metadata.version, consumer)
        )
