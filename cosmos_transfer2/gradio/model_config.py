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
from dataclasses import dataclass, field
from typing import Dict

from cosmos_transfer2.config import InferenceArguments
from cosmos_transfer2.gradio.sample_data import (
    sample_request_depth,
    sample_request_edge,
    sample_request_multicontrol,
    sample_request_mv,
    sample_request_seg,
    sample_request_vis,
)
from cosmos_transfer2.multiview_config import MultiviewInferenceArguments

help_control2world = f"```json\n{json.dumps(InferenceArguments.model_json_schema(), indent=2)}\n```"
help_text_mv = f"```json\n{json.dumps(MultiviewInferenceArguments.model_json_schema(), indent=2)}\n```"


@dataclass
class Config:
    header: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": "Cosmos-Transfer2.5 Blur Transfer",
            "depth": "Cosmos-Transfer2.5 Depth Transfer",
            "edge": "Cosmos-Transfer2.5 Edge Transfer",
            "seg": "Cosmos-Transfer2.5 Segmentation Transfer",
            "multicontrol": "Cosmos-Transfer2.5 Multi-control Transfer",
            "multiview": "Cosmos-Transfer2.5 Multiview",
        }
    )

    help_text: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": help_control2world,
            "depth": help_control2world,
            "edge": help_control2world,
            "seg": help_control2world,
            "multicontrol": help_control2world,
            "multiview": help_text_mv,
        }
    )

    default_request: Dict[str, str] = field(
        default_factory=lambda: {
            "vis": json.dumps(sample_request_vis, indent=2),
            "depth": json.dumps(sample_request_depth, indent=2),
            "edge": json.dumps(sample_request_edge, indent=2),
            "seg": json.dumps(sample_request_seg, indent=2),
            "multicontrol": json.dumps(sample_request_multicontrol, indent=2),
            "multiview": json.dumps(sample_request_mv, indent=2),
        }
    )
