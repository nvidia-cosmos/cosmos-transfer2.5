# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

from cosmos_transfer2.config import (
    InferenceArguments,
    SetupArguments,
)
from cosmos_transfer2.inference import Control2WorldInference


class Control2World_Worker:
    def __init__(self, num_gpus=1, model="edge"):
        setup_args = SetupArguments(
            context_parallel_size=num_gpus,
            output_dir=Path("outputs"),  # dummy parameter, we want to save videos in per inference folders
            model=model,
            keep_going=True,
        )

        self.pipe = Control2WorldInference(setup_args, batch_hint_keys=[model])

    def infer(self, args: dict):
        """
        Adjust inputs from gradio to InferenceArguments.
        Adjust output from Control2WorldInference.generate to gradio.

        Args:
            args (dict): Dictionary containing InferenceArguments attributes

        Returns:
            dict: Dictionary containing:
                - videos: List of generated video paths
        """

        output_dir = args.pop("output_dir", "outputs")

        inference_args = InferenceArguments(**args)
        output_videos = self.pipe.generate([inference_args], Path(output_dir))

        return {
            "videos": output_videos,
        }
