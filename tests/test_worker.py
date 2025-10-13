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

import json
import os

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_server import ModelServer

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import Control2WorldParams, MultiviewParams
from cosmos_transfer2.control2world import Control2WorldInference

global_env = DeploymentEnv()


NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."

asset_dir = os.getenv("ASSET_DIR", "assets/")
sample_params = {
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "negative_prompt": NEGATIVE_PROMPT,
    "output_dir": "outputs/test_worker/",
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "edge": {
        "control_path": os.path.join(asset_dir, "robot_example/edge/robot_edge.mp4"),
    },
}

sample_params_mv = {
    "output_dir": "outputs/multiview_control2world",
    "prompt_path": os.path.join(asset_dir, "multiview_example/prompt.txt"),
    "front_wide": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_0.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_0.mp4",
        ),
    },
    "cross_left": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_1.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_1.mp4",
        ),
    },
    "cross_right": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_2.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_2.mp4",
        ),
    },
    "rear_left": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_3.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_3.mp4",
        ),
    },
    "rear_right": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_4.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_4.mp4",
        ),
    },
    "rear": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_5.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_5.mp4",
        ),
    },
    "front_tele": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_video_6.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/52b3ef06-2b32-4781-aa01-d419a60f141c_10917899711_10937899711_input_world_scenario_6.mp4",
        ),
    },
}


def test_transfer(model_name, params):
    params = Control2WorldParams.create(params)
    log.info(f"params: {json.dumps(params.to_kwargs(), indent=4)}")
    pipeline = Control2WorldInference(num_gpus=1, hint_key=model_name)

    params.output_dir = f"outputs/transfer2/{model_name}"
    pipeline.infer(params.to_kwargs())


def test_transfer_mv():
    params = MultiviewParams.create(sample_params_mv)
    _ = params.input_and_control_paths  # validate paths

    # must run on 8 GPUs
    with ModelServer(
        num_gpus=8, factory_module="cosmos_transfer2.gradio.gradio_bootstrapper", factory_function="create_multiview"
    ) as pipeline:
        params.output_dir = "outputs/transfer2/multiview/"
        pipeline.infer(params.to_kwargs())


# Note that multiview requires 8 GPUs and cannot be tested w/o torchrun
if __name__ == "__main__":
    log.info(f"test_worker current dir={os.getcwd()}")
    log.info(global_env)
    if global_env.model_name == "edge":
        test_transfer(["edge"], sample_params)
    elif global_env.model_name == "multiview":
        test_transfer_mv()
