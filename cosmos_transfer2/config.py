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
from dataclasses import dataclass
from typing import Literal

from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid
from cosmos_transfer2._src.imaginaire.utils.validator import Bool, Dict, Float, Int, Path, String
from cosmos_transfer2._src.imaginaire.utils.validator_params import ValidatorParams

DEFAULT_NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."

ModelSize = Literal["2B"]
ModelType = Literal["control2world"]
ModelVariant = Literal["base", "depth", "drive", "edge", "seg", "vis"]


@dataclass(frozen=True, kw_only=True)
class ModelKey:
    size: ModelSize = "2B"
    type: ModelType = "control2world"
    variant: ModelVariant = "base"


MODEL_CHECKPOINTS = {
    ModelKey(variant="base"): get_checkpoint_by_uuid("24a3b7b8-6a3d-432d-b7d1-5d30b9229465"),
    ModelKey(variant="depth"): get_checkpoint_by_uuid("0f214f66-ae98-43cf-ab25-d65d09a7e68f"),
    ModelKey(variant="drive"): get_checkpoint_by_uuid("b5ab002d-a120-4fbf-a7f9-04af8615710b"),
    ModelKey(variant="edge"): get_checkpoint_by_uuid("ecd0ba00-d598-4f94-aa09-e8627899c431"),
    ModelKey(variant="seg"): get_checkpoint_by_uuid("fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab"),
    ModelKey(variant="vis"): get_checkpoint_by_uuid("20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1"),
}


class Control2WorldParams(ValidatorParams):
    """All the required values to generate image from text at a given resolution."""

    video_path = Path(default="")
    image_context_path = String(default=None)
    prompt = String(default=None)
    prompt_path = String(default="")
    negative_prompt = String(default=DEFAULT_NEGATIVE_PROMPT)
    output_dir = String(default="outputs/")

    seed = Int(default=2025)
    resolution = String(default="720")
    guidance = Int(default=3)
    control_weight = Float(default=1.0, min=0.0, max=1.0, step=0.01)
    sigma_max = String(default=None)
    show_control_condition = Bool(default=False)
    show_input = Bool(default=False)
    not_keep_input_resolution = Bool(default=False)
    disable_guardrails = Bool(default=False)
    offload_guardrail_models = Bool(default=True)

    edge = Dict(default={})
    vis = Dict(default={})
    depth = Dict(default={})
    seg = Dict(default={})

    def __init__(self):
        super().__init__()
        self._provided_control_keys = set()

    def from_kwargs(self, kwargs):
        # Track which control modalities were explicitly provided in the JSON
        control_modalities = ["edge", "vis", "depth", "seg"]
        for modality in control_modalities:
            if modality in kwargs:
                self._provided_control_keys.add(modality)

        # Call parent method to set all attributes
        super().from_kwargs(kwargs)

    @property
    def control_modalities(self):
        control_modalities = {
            "edge": self.edge.get("control_path", None),
            "vis": self.vis.get("control_path", None),
            "depth": self.depth.get("control_path", None),
            "seg": self.seg.get("control_path", None),
            "edge_mask": self.edge.get("mask_path", None),
            "vis_mask": self.vis.get("mask_path", None),
            "depth_mask": self.depth.get("mask_path", None),
            "seg_mask": self.seg.get("mask_path", None),
        }

        for modality, path in control_modalities.items():
            if path:
                if not os.path.exists(path):
                    raise ValueError(f"Control input file for {modality} not found: {path}")
        return control_modalities

    @property
    def multicontrol_weight(self):
        # string comma separated
        control_weight_str = str(self.edge.get("control_weight", self.control_weight))
        control_weight_str += "," + str(self.vis.get("control_weight", self.control_weight))
        control_weight_str += "," + str(self.depth.get("control_weight", self.control_weight))
        control_weight_str += "," + str(self.seg.get("control_weight", self.control_weight))
        return control_weight_str

    @property
    def hint_key(self):
        """Extract hint_key as comma-separated string from available control modalities."""
        available_controls = []

        # Check which control modalities were explicitly provided in JSON (control_path is optional)
        for modality in ["edge", "vis", "depth", "seg"]:
            if modality in self._provided_control_keys:
                available_controls.append(modality)

        if not available_controls:
            raise ValueError(
                "No control modalities found. At least one control type (edge, vis, depth, seg) must be specified."
            )

        return available_controls


def get_params_from_json(json_path: str) -> Control2WorldParams:
    with open(json_path, "r") as f:
        try:
            params = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {json_path} - {e}")
    return Control2WorldParams.create(params)


VIEW_INDEX_DICT = {
    "front_wide": 0,
    "cross_left": 1,
    "cross_right": 2,
    "rear_left": 3,
    "rear_right": 4,
    "rear": 5,
    "front_tele": 6,
}


class MultiviewParams(ValidatorParams):
    """All the required values to generate image from text at a given resolution."""

    output_dir = String("outputs/")
    prompt = String(default=None)
    prompt_path = String(default="")

    guidance = Int(3, min=0, max=7)
    seed = Int(0)
    n_views = Int(7, hidden=True)
    num_conditional_frames = Int(1, min=0, max=2)
    control_weight = Float(1.0, min=0.0, max=1.0, step=0.01)

    front_wide = Dict(default={})
    rear = Dict(default={})
    rear_left = Dict(default={})
    rear_right = Dict(default={})
    cross_left = Dict(default={})
    cross_right = Dict(default={})
    front_tele = Dict(default={})
    fps = Int(default=10)

    disable_guardrails = Bool(default=False)
    offload_guardrail_models = Bool(default=True)

    @property
    def input_and_control_paths(self):
        input_and_control_paths = {
            "front_wide_input": self.front_wide.get("input_path", None),
            "rear_input": self.rear.get("input_path", None),
            "rear_left_input": self.rear_left.get("input_path", None),
            "rear_right_input": self.rear_right.get("input_path", None),
            "cross_left_input": self.cross_left.get("input_path", None),
            "cross_right_input": self.cross_right.get("input_path", None),
            "front_tele_input": self.front_tele.get("input_path", None),
            "front_wide_control": self.front_wide.get("control_path", None),
            "rear_control": self.rear.get("control_path", None),
            "rear_left_control": self.rear_left.get("control_path", None),
            "rear_right_control": self.rear_right.get("control_path", None),
            "cross_left_control": self.cross_left.get("control_path", None),
            "cross_right_control": self.cross_right.get("control_path", None),
            "front_tele_control": self.front_tele.get("control_path", None),
        }
        for key, path in input_and_control_paths.items():
            if key:
                if not os.path.exists(path):
                    raise ValueError(f"File {key} not found at path: {path}")
        return input_and_control_paths


def get_multiview_params_from_json(json_path: str) -> MultiviewParams:
    with open(json_path, "r") as f:
        try:
            params = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {json_path} - {e}")
    params = MultiviewParams.create(params)
    params.input_and_control_paths
    return params
