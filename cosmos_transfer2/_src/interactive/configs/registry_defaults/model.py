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
from cosmos_transfer2._src.interactive.configs.method_configs.config_dmd2 import DMD2Config
from cosmos_transfer2._src.interactive.methods.diffusion_forcing import (
    DiffusionForcingCLModel,
    DiffusionForcingCLModelConfig,
)
from cosmos_transfer2._src.interactive.methods.distribution_matching.dmd2 import DMD2Model
from cosmos_transfer2._src.interactive.methods.distribution_matching.self_forcing_cl import (
    SelfForcingCLModel,
    SelfForcingCLModelConfig,
)

FSDP_DMD2_TRIGFLOW_MODEL = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(DMD2Model)(
        config=DMD2Config(),
        _recursive_=False,
    ),
)

FSDP_SELF_FORCING_MODEL_CL = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(SelfForcingCLModel)(
        config=SelfForcingCLModelConfig(fsdp_shard_size=8),
        _recursive_=False,
    ),
)

FSDP_DIFFUSION_FORCING_MODEL_CL = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(DiffusionForcingCLModel)(
        config=DiffusionForcingCLModelConfig(fsdp_shard_size=8),
        _recursive_=False,
    ),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_dmd2_model_trigflow",
        node=FSDP_DMD2_TRIGFLOW_MODEL,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_self_forcing_model_cl",
        node=FSDP_SELF_FORCING_MODEL_CL,
    )
    cs.store(
        group="model",
        package="_global_",
        name="fsdp_diffusion_forcing_model_cl",
        node=FSDP_DIFFUSION_FORCING_MODEL_CL,
    )
