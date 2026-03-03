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

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from einops import rearrange
from hydra.core.config_store import ConfigStore
from torch.distributed import get_process_group_ranks

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.context_parallel import broadcast_split_tensor, find_split
from cosmos_transfer2._src.predict2.conditioner import ReMapkey, Text2WorldCondition, TextAttr
from cosmos_transfer2._src.predict2_multiview.conditioner import MVTextAttr
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import MultiViewCondition
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import (
    _SHARED_CONFIG,
    _SHARED_CONFIG_AV,
    ControlVideo2WorldCondition,
    ControlVideo2WorldConditioner,
)


@dataclass(frozen=True)
class MultiViewControlVideo2WorldCondition(
    MultiViewCondition,  # provides multiview logic (set_video_condition …)
    ControlVideo2WorldCondition,  # provides control-specific fields & helpers (set_control_condition)
):
    """
    Multiview + Control condition.

    • All multiview helpers (`set_video_condition`, `enable_ref_cam_condition`,
      `edit_for_inference`, …) are inherited **unchanged** from
      `MultiViewCondition`.

    • All control helpers (`set_control_condition`, control-field definitions,
      etc.) come from `ControlVideo2WorldCondition`.

    Only the `broadcast` method needs a custom implementation because it has to
    deal with both the multiview tensors **and** the extra
    `latent_control_input`.
    """

    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "MultiViewControlVideo2WorldCondition":
        """
        Broadcasts condition with both multiview and control tensor support.
        Calls parent's broadcast for control-specific logic, then adds multiview broadcasting.
        """
        if self.is_broadcasted:
            return self

        # Handle multiview tensors separately
        gt_frames_B_C_T_H_W = self.gt_frames
        view_indices_B_T = self.view_indices_B_T
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        latent_control_input = self.latent_control_input

        # Temporarily remove multiview tensors for parent broadcasting
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        kwargs["view_indices_B_T"] = None
        kwargs["latent_control_input"] = None

        new_condition = Text2WorldCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames_B_C_T_H_W.shape
        n_views = T // self.state_t
        assert T % self.state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={self.state_t}."

        if process_group is not None and T > 1 and process_group.size() > 1:
            log.debug(f"Broadcasting multiview control tensors {gt_frames_B_C_T_H_W.shape=} to {n_views=} views")
            cp_ranks = get_process_group_ranks(process_group)
            cp_size = len(cp_ranks)
            # Perform spatial split only when it's required, i.e. temporal split is not enough.
            # Refer to "find_split" definition for more details.
            use_spatial_split = cp_size > self.state_t or self.state_t % cp_size != 0
            after_split_shape = (
                find_split(gt_frames_B_C_T_H_W.shape, cp_size, view_factor=n_views) if use_spatial_split else None
            )
            gt_frames_B_C_V_T_H_W = rearrange(gt_frames_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_views)
            condition_video_input_mask_B_C_V_T_H_W = rearrange(
                condition_video_input_mask_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_views
            )
            if use_spatial_split:
                gt_frames_B_C_V_T_H_W = rearrange(gt_frames_B_C_V_T_H_W, "B C V T H W -> B C V (T H W)")
                condition_video_input_mask_B_C_V_T_H_W = rearrange(
                    condition_video_input_mask_B_C_V_T_H_W, "B C V T H W -> B C V (T H W)"
                )
            view_indices_B_V_T = rearrange(view_indices_B_T, "B (V T) -> B V T", V=n_views)
            if latent_control_input is not None:
                if latent_control_input.dim() == 5:  # B, C, T, H, W
                    latent_control_input_B_C_V_T_H_W = rearrange(
                        latent_control_input, "B C (V T) H W -> B C V T H W", V=n_views
                    )
                    if use_spatial_split:
                        latent_control_input_B_C_V_T_H_W = rearrange(
                            latent_control_input_B_C_V_T_H_W, "B C V T H W -> B C V (T H W)"
                        )
                    latent_control_input_B_C_V_T_H_W = broadcast_split_tensor(
                        latent_control_input_B_C_V_T_H_W, seq_dim=3, process_group=process_group
                    )
                    if use_spatial_split:
                        after_split_shape_latent = find_split(latent_control_input.shape, cp_size, view_factor=n_views)
                        latent_control_input_B_C_V_T_H_W = rearrange(
                            latent_control_input_B_C_V_T_H_W,
                            "B C V (T H W) -> B C V T H W",
                            V=n_views,
                            T=after_split_shape_latent[0],
                            H=after_split_shape_latent[1],
                        )
                    latent_control_input = rearrange(
                        latent_control_input_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                    )
            gt_frames_B_C_V_T_H_W = broadcast_split_tensor(
                gt_frames_B_C_V_T_H_W, seq_dim=3, process_group=process_group
            )
            condition_video_input_mask_B_C_V_T_H_W = broadcast_split_tensor(
                condition_video_input_mask_B_C_V_T_H_W, seq_dim=3, process_group=process_group
            )
            if use_spatial_split:
                if cp_size % self.state_t != 0:
                    split_size_t = math.gcd(self.state_t, cp_size)
                    B, V, _ = view_indices_B_V_T.shape
                    view_indices_B_V_T = view_indices_B_V_T.view(B, V, split_size_t, -1)
                    view_indices_B_V_T = view_indices_B_V_T.repeat_interleave(
                        repeats=max(1, cp_size // split_size_t), dim=2
                    )
                    view_indices_B_V_T = view_indices_B_V_T.view(B, V, -1)
                else:
                    view_indices_B_V_T = view_indices_B_V_T.repeat_interleave(
                        repeats=max(1, cp_size // self.state_t), dim=2
                    )
            view_indices_B_V_T = broadcast_split_tensor(view_indices_B_V_T, seq_dim=2, process_group=process_group)

            if use_spatial_split:
                gt_frames_B_C_V_T_H_W = rearrange(
                    gt_frames_B_C_V_T_H_W,
                    "B C V (T H W) -> B C V T H W",
                    V=n_views,
                    T=after_split_shape[0],
                    H=after_split_shape[1],
                )
                condition_video_input_mask_B_C_V_T_H_W = rearrange(
                    condition_video_input_mask_B_C_V_T_H_W,
                    "B C V (T H W) -> B C V T H W",
                    V=n_views,
                    T=after_split_shape[0],
                    H=after_split_shape[1],
                )
            gt_frames_B_C_T_H_W = rearrange(gt_frames_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views)
            condition_video_input_mask_B_C_T_H_W = rearrange(
                condition_video_input_mask_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
            )
            view_indices_B_T = rearrange(view_indices_B_V_T, "B V T -> B (V T)", V=n_views)

        kwargs["gt_frames"] = gt_frames_B_C_T_H_W
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["view_indices_B_T"] = view_indices_B_T
        kwargs["latent_control_input"] = latent_control_input

        return type(self)(**kwargs)


class MultiViewControlVideo2WorldConditioner(ControlVideo2WorldConditioner):
    """
    Conditioner that produces MultiViewControlVideo2WorldCondition objects.
    Inherits from GeneralConditioner and adds multiview control functionality.
    """

    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> MultiViewControlVideo2WorldCondition:
        output = super()._forward(batch, override_dropout_rate)
        return MultiViewControlVideo2WorldCondition(**output)


MultiViewVideoPredictionControlConditioner: LazyDict = L(MultiViewControlVideo2WorldConditioner)(
    **_SHARED_CONFIG_AV,  # Inherits all transfer2 control config
    # Add multiview-specific config
    view_indices_B_T=L(ReMapkey)(
        input_key="latent_view_indices_B_T",
        output_key="view_indices_B_T",
        dropout_rate=0.0,
        dtype=None,
    ),
    ref_cam_view_idx_sample_position=L(ReMapkey)(
        input_key="ref_cam_view_idx_sample_position",
        output_key="ref_cam_view_idx_sample_position",
        dropout_rate=0.0,
        dtype=None,
    ),
)


_SHARED_CONFIG_AV_PER_VIEW_DROPOUT = copy.deepcopy(_SHARED_CONFIG_AV)
_SHARED_CONFIG_AV_PER_VIEW_DROPOUT["text"] = L(MVTextAttr)(
    input_key=["t5_text_embeddings"],
    dropout_rate=0.2,
    use_empty_string=False,
)

MultiViewVideoPredictionControlConditionerPerViewDropout_AV: LazyDict = L(MultiViewControlVideo2WorldConditioner)(
    **_SHARED_CONFIG_AV_PER_VIEW_DROPOUT,
    # Add multiview-specific config
    view_indices_B_T=L(ReMapkey)(
        input_key="latent_view_indices_B_T",
        output_key="view_indices_B_T",
        dropout_rate=0.0,
        dtype=None,
    ),
    ref_cam_view_idx_sample_position=L(ReMapkey)(
        input_key="ref_cam_view_idx_sample_position",
        output_key="ref_cam_view_idx_sample_position",
        dropout_rate=0.0,
        dtype=None,
    ),
)


_SHARED_CONFIG_PER_VIEW_DROPOUT = copy.deepcopy(_SHARED_CONFIG)
_SHARED_CONFIG_PER_VIEW_DROPOUT["text"] = L(MVTextAttr)(
    input_key=["t5_text_embeddings"],
    dropout_rate=0.2,
    use_empty_string=False,
)

MultiViewVideoPredictionControlConditionerPerViewDropout: LazyDict = L(MultiViewControlVideo2WorldConditioner)(
    **_SHARED_CONFIG_PER_VIEW_DROPOUT,
    # Add multiview-specific config
    view_indices_B_T=L(ReMapkey)(
        input_key="latent_view_indices_B_T",
        output_key="view_indices_B_T",
        dropout_rate=0.0,
        dtype=None,
    ),
    ref_cam_view_idx_sample_position=L(ReMapkey)(
        input_key="ref_cam_view_idx_sample_position",
        output_key="ref_cam_view_idx_sample_position",
        dropout_rate=0.0,
        dtype=None,
    ),
)


class TextAttrEmptyStringDropout(TextAttr):
    def __init__(
        self,
        input_key: str,
        pos_input_key: str,
        dropout_input_key: str,
        dropout_rate: Optional[float] = 0.0,
        use_empty_string: bool = False,
        **kwargs,
    ):
        self._input_key = input_key
        self._pos_input_key = pos_input_key
        self._dropout_input_key = dropout_input_key
        self._dropout_rate = dropout_rate
        self._use_empty_string = use_empty_string

        super().__init__(
            input_key,
            dropout_rate,
        )

    def forward(self, tensor: torch.Tensor):
        return {"crossattn_emb": tensor}

    def random_dropout_input(
        self,
        in_tensor_dict: torch.Tensor | Dict[str, torch.Tensor],
        dropout_rate: Optional[float] = None,
        key: Optional[str] = None,
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return in_tensor_dict
        del key
        assert isinstance(in_tensor_dict, dict), f"in_tensor_dict must be a dict. Got {type(in_tensor_dict)}"
        in_tensor = in_tensor_dict[self._pos_input_key]
        B = in_tensor.shape[0]
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        keep_mask = torch.bernoulli((1.0 - dropout_rate) * torch.ones(B)).type_as(in_tensor)
        if self._use_empty_string:
            empty_prompt = in_tensor_dict[self._dropout_input_key]
            if empty_prompt.shape[0] != B:
                empty_prompt = empty_prompt.repeat(B, 1, 1)
        else:
            empty_prompt = torch.zeros_like(in_tensor)

        return keep_mask * in_tensor + (1 - keep_mask) * empty_prompt

    def details(self) -> str:
        return "Output key: [crossattn_emb]"


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_multiview_control_conditioner",
        node=MultiViewVideoPredictionControlConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_multiview_control_conditioner_per_view_dropout_av",
        node=MultiViewVideoPredictionControlConditionerPerViewDropout_AV,
    )
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_multiview_control_conditioner_per_view_dropout",
        node=MultiViewVideoPredictionControlConditionerPerViewDropout,
    )
