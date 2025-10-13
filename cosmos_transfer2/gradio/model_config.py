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
from dataclasses import dataclass, field
from typing import Dict

asset_dir = os.environ.get("ASSET_DIR", "assets/")

default_request_vis = json.dumps(
    {
        "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
        "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
        "control_weight": 1.0,
        "vis": {"control_path": os.path.join(asset_dir, "robot_example/vis/robot_vis.mp4")},
    },
    indent=2,
)

default_request_depth = json.dumps(
    {
        "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
        "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
        "control_weight": 1.0,
        "depth": {"control_path": os.path.join(asset_dir, "robot_example/depth/robot_depth.mp4")},
    },
    indent=2,
)

default_request_edge = json.dumps(
    {
        "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
        "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
        "control_weight": 1.0,
        "edge": {
            "control_path": os.path.join(asset_dir, "robot_example/edge/robot_edge.mp4"),
            "preset_edge_threshold": "medium",
        },
    },
    indent=2,
)

default_request_seg = json.dumps(
    {
        "prompt_path": os.path.join(asset_dir, "robot_example/robot_prompt.txt"),
        "video_path": os.path.join(asset_dir, "robot_example/robot_input.mp4"),
        "control_weight": 1.0,
        "seg": {"control_path": os.path.join(asset_dir, "robot_example/seg/robot_seg.mp4")},
    },
    indent=2,
)


default_request_mv = json.dumps(
    {
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
    },
    indent=2,
)

help_text_vis = """
                    ### Generation Parameters:
                    - `prompt` (string): Text description of desired output
                    - `video_path` (string): Path to input video file
                    - `control_weight` (float): Weight for the control signal (default: 1.0)
                    - `sigma_max` (float | None): Max noise level (0–80). Higher = more variation
                    - `num_conditional_frames` (int): Conditional frames passed from one chunk to the next.

                    """
help_text_depth = help_text_vis
help_text_edge = help_text_vis
help_text_seg = help_text_vis
help_text_mv = help_text_vis


@dataclass
class Config:
    header: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": "Cosmos-Transfer2.5 Blur Transfer",
            "depth": "Cosmos-Transfer2.5 Depth Transfer",
            "edge": "Cosmos-Transfer2.5 Edge Transfer",
            "seg": "Cosmos-Transfer2.5 Segmentation Transfer",
            "multiview": "Cosmos-Transfer2.5 Multiview",
        }
    )

    help_text: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": help_text_vis,
            "depth": help_text_depth,
            "edge": help_text_edge,
            "seg": help_text_seg,
            "multiview": "",
        }
    )

    default_request: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": default_request_vis,
            "depth": default_request_depth,
            "edge": default_request_edge,
            "seg": default_request_seg,
            "multiview": default_request_mv,
        }
    )
