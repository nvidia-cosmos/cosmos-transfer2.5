# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cosmos_transfer2.robot_multiview import RobotMultiviewControlAgibotInference
from cosmos_transfer2.robot_multiview_control_agibot_config import (
    RobotMultiviewControlAgibotInferenceArguments,
    RobotMultiviewControlAgibotSetupArguments,
)


class RobotMultiviewAgibotWorker:
    def __init__(
        self,
        num_gpus: int,
        control_type: Literal["depth", "edge", "vis", "seg"],
    ):
        setup_args = RobotMultiviewControlAgibotSetupArguments(
            input_root=Path("/tmp"),
            output_dir=Path("outputs"),
            control_type=control_type,
            context_parallel_size=num_gpus,
        )
        self.pipe = RobotMultiviewControlAgibotInference(setup_args)

    def infer(self, args: dict) -> dict:
        output_dir = Path(args.pop("output_dir", "outputs"))

        inference_args = RobotMultiviewControlAgibotInferenceArguments(**args)
        output_videos = self.pipe.generate([inference_args], output_dir)

        return {
            "videos": output_videos,
        }
