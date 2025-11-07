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
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_transfer2._src.predict2_multiview.networks.multiview_cross_dit import MultiViewSACConfig
from cosmos_transfer2._src.transfer2_multiview.networks.multiview_cross_dit_control import MultiViewCrossControlDiT
from cosmos_transfer2._src.transfer2_multiview.networks.multiview_dit_control import MultiViewControlDiT

COSMOS_V2_CONTROL_VACE_2B_MV = L(MultiViewControlDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    n_cameras=3,
    n_cameras_emb=7,
    view_condition_dim=6,
    concat_view_embedding=True,
    use_input_hint_block=True,
    condition_strategy="first_n",
    vace_block_every_n=4,
    num_max_modalities=8,
    sac_config=SACConfig(),
)

COSMOS_V2_CONTROL_VACE_2B_MV_CROSS = L(MultiViewCrossControlDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=MultiViewSACConfig(),
    n_cameras_emb=7,
    view_condition_dim=6,
    concat_view_embedding=False,
    adaln_view_embedding=True,
    enable_cross_view_attn=True,
    use_wan_fp32_strategy=False,
    layer_mask=None,
    timestep_scale=0.001,  # important for rectified flow
    # transfer params
    vace_has_mask=False,
    use_input_hint_block=True,
    condition_strategy="spaced",
    vace_block_every_n=4,
    num_max_modalities=1,
)


def register_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_multiview_control",
        node=COSMOS_V2_CONTROL_VACE_2B_MV,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_multiview_crossview_control",
        node=COSMOS_V2_CONTROL_VACE_2B_MV_CROSS,
    )
