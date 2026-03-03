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

NEGATIVE_PROMPT_MULTIVIEW_MANY_CAMERA = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    "poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, poorly edited content, "
    "jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
)

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


sample_request_edge_distilled = {
    "name": "robot_edge_distilled",
    "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
    "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
    "guidance": 3,
    "num_steps": 4,
    "seed": 1,
    "edge": {"control_path": os.path.join(asset_dir, "robot_example/edge/robot_edge.mp4"), "control_weight": 1.0},
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

_agibot_asset_dir = os.path.join(asset_dir, "robot_multiview_control-agibot")
_agibot_video_id = "296_656371_chunk0"
sample_request_agibot_edge = {
    "name": "robot_multiview_agibot_edge",
    "head_color": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_head_color_rgb.mp4"),
    },
    "hand_left": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_left_rgb.mp4"),
    },
    "hand_right": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_right_rgb.mp4"),
    },
    "guidance": 7.0,
    "seed": 1,
}

sample_request_agibot_vis = {
    "name": "robot_multiview_agibot_vis",
    "head_color": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_head_color_rgb.mp4"),
    },
    "hand_left": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_left_rgb.mp4"),
    },
    "hand_right": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_right_rgb.mp4"),
    },
    "guidance": 7.0,
    "seed": 1,
}

sample_request_agibot_depth = {
    "name": "robot_multiview_agibot_depth",
    "head_color": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_head_color_rgb.mp4"),
        "control_path": os.path.join(_agibot_asset_dir, "depth", f"{_agibot_video_id}_head_color_depth.mp4"),
    },
    "hand_left": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_left_rgb.mp4"),
        "control_path": os.path.join(_agibot_asset_dir, "depth", f"{_agibot_video_id}_hand_left_depth.mp4"),
    },
    "hand_right": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_right_rgb.mp4"),
        "control_path": os.path.join(_agibot_asset_dir, "depth", f"{_agibot_video_id}_hand_right_depth.mp4"),
    },
    "guidance": 7.0,
    "seed": 1,
}

sample_request_agibot_seg = {
    "name": "robot_multiview_agibot_seg",
    "head_color": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_head_color_rgb.mp4"),
        "control_path": os.path.join(_agibot_asset_dir, "seg", f"{_agibot_video_id}_head_color_seg.mp4"),
    },
    "hand_left": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_left_rgb.mp4"),
        "control_path": os.path.join(_agibot_asset_dir, "seg", f"{_agibot_video_id}_hand_left_seg.mp4"),
    },
    "hand_right": {
        "input_path": os.path.join(_agibot_asset_dir, "videos", f"{_agibot_video_id}_hand_right_rgb.mp4"),
        "control_path": os.path.join(_agibot_asset_dir, "seg", f"{_agibot_video_id}_hand_right_seg.mp4"),
    },
    "guidance": 7.0,
    "seed": 1,
}

sample_request_multiview_many_camera = {
    "name": "multiview_many_camera_motion",
    "input_path": os.path.join(asset_dir, "plenoptic_example/videos/0.mp4"),
    "prompt": (
        "The video takes place in a warehouse with high shelving units on either side, "
        "filled with various items. The floor is marked with yellow caution tape and orange "
        "traffic cones, indicating a restricted or hazardous area. A person wearing a white "
        "shirt and dark pants is seen walking down the aisle, carrying a box. The shelves are "
        "labeled with letters such as 'C', 'D', 'E', and 'F', suggesting an organized system "
        "for inventory management. The lighting is artificial, typical of indoor industrial "
        "settings, and the overall atmosphere is one of routine activity within a storage facility."
    ),
    "negative_prompt": NEGATIVE_PROMPT_MULTIVIEW_MANY_CAMERA,
    "camera_sequence": [
        "static",
        "rot_left",
        "arc_right",
        "azimuth_right",
        "rot_right",
        "arc_left",
        "azimuth_left",
        "tilt_up",
        "translate_down_rot",
        "tilt_down",
        "translate_up_rot",
        "elevation_up_1",
        "zoom_out",
    ],
    "guidance": 7,
    "fps": 30,
    "seed": 1,
}
