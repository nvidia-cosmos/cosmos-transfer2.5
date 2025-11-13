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
from cosmos_transfer2._src.transfer2_multiview.models.multiview_vid2vid_model_control_vace_rectified_flow import (
    MultiviewControlVideo2WorldModelRectifiedFlow,
    MultiviewControlVideo2WorldRectifiedFlowConfig,
)

MV_FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(MultiviewControlVideo2WorldModelRectifiedFlow)(
        config=MultiviewControlVideo2WorldRectifiedFlowConfig(
            fsdp_shard_size=8,
        ),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_rectified_flow_multiview_control",
        node=MV_FSDP_CONFIG_CONTROL_VACE_RECTIFIED_FLOW,
    )
