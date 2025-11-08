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

import os

NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."

asset_dir = os.getenv("ASSET_DIR", "assets/")
sample_request_edge = {
    "name": "robot_edge",
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "guidance": 3,
    "edge": {"control_path": os.path.join(asset_dir, "robot_example/edge/robot_edge.mp4"), "control_weight": 1.0},
}

sample_request_vis = {
    "name": "robot_vis",
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "guidance": 3,
    "vis": {"control_path": os.path.join(asset_dir, "robot_example/vis/robot_vis.mp4"), "control_weight": 1.0},
}

sample_request_depth = {
    "name": "robot_depth",
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "guidance": 3,
    "depth": {"control_path": os.path.join(asset_dir, "robot_example/depth/robot_depth.mp4"), "control_weight": 1.0},
}


sample_request_seg = {
    "name": "robot_seg",
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "guidance": 3,
    "seg": {"control_path": os.path.join(asset_dir, "robot_example/seg/robot_seg.mp4"), "control_weight": 1.0},
}

sample_request_multicontrol = {
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "name": "robot_multicontrol",
    "guidance": 3,
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "depth": {"control_path": os.path.join(asset_dir, "robot_example/depth/robot_depth.mp4"), "control_weight": 1.0},
    "edge": {"control_path": os.path.join(asset_dir, "robot_example/edge/robot_edge.mp4"), "control_weight": 1.0},
    "seg": {"control_path": os.path.join(asset_dir, "robot_example/seg/robot_seg.mp4"), "control_weight": 1.0},
    "vis": {"control_weight": 1.0},
}


sample_request_mv = {
    "name": "multiview_control2world",
    "prompt_path": os.path.join(asset_dir, "multiview_example/prompt.txt"),
    "front_wide": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_front_wide_120fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_front_wide_120fov.mp4",
        ),
    },
    "cross_left": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_cross_left_120fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_cross_left_120fov.mp4",
        ),
    },
    "cross_right": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_cross_right_120fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_cross_right_120fov.mp4",
        ),
    },
    "rear_left": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_rear_left_70fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_rear_left_70fov.mp4",
        ),
    },
    "rear_right": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_rear_right_70fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_rear_right_70fov.mp4",
        ),
    },
    "rear": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_rear_30fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_rear_30fov.mp4",
        ),
    },
    "front_tele": {
        "input_path": os.path.join(
            asset_dir,
            "multiview_example/input_videos/night_front_tele_30fov.mp4",
        ),
        "control_path": os.path.join(
            asset_dir,
            "multiview_example/world_scenario_videos/ws_front_tele_30fov.mp4",
        ),
    },
}
