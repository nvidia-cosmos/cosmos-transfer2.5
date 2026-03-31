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

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.interactive.conditioner.condition_postprocessors import (
    ControlConditionPostprocessor,
    HDMapI2VConditionPostprocessor,
)

control_condition_postprocessor: LazyDict = L(ControlConditionPostprocessor)(hint_keys=["edge"])
hdmap_i2v_condition_postprocessor: LazyDict = L(HDMapI2VConditionPostprocessor)(
    preset_hint_keys=["control_input_hdmap_bbox"],
    hdmap_process_method="vae_encoding",
    hdmap_selection_mode="all",
)


def register_condition_postprocessor():
    cs = ConfigStore.instance()
    cs.store(
        group="condition_postprocessor",
        package="model.config.condition_postprocessor",
        name="control_condition_postprocessor",
        node=control_condition_postprocessor,
    )
    cs.store(
        group="condition_postprocessor",
        package="model.config.condition_postprocessor",
        name="hdmap_i2v_condition_postprocessor",
        node=hdmap_i2v_condition_postprocessor,
    )
