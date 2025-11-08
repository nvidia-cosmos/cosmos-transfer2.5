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

import gc
import math
from typing import Any, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from megatron.core import parallel_state
from torch import amp
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torchvision import transforms

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import (
    PatchEmbed,
)
from cosmos_transfer2._src.predict2_multiview.networks.multiview_cross_dit import (
    MultiViewCrossBlock,
    MultiViewCrossDiT,
    MultiViewSACConfig,
)


class MultiViewCrossControlAwareBlock(MultiViewCrossBlock):
    def __init__(self, *args, block_id: Optional[int] = None, **kwargs):
        """
        Initialize the block.
        Args:
            block_id: block id in vace. used to index into the hints tensor.
        """
        super().__init__(*args, **kwargs)
        self.block_id = block_id

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        hints: Optional[torch.Tensor] = None,
        control_context_scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass of the block.
        """
        x_B_T_H_W_D = super().forward(x_B_T_H_W_D, **kwargs)
        if self.block_id is not None and hints is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + hints[self.block_id] * control_context_scale
        return x_B_T_H_W_D


class MultiViewCrossControlEncoderBlock(MultiViewCrossBlock):
    def __init__(
        self,
        *args,
        block_id: Optional[int] = None,
        hint_dim: Optional[int] = None,
        use_after_proj: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_id = block_id
        self.hint_dim = hint_dim
        self.use_after_proj = use_after_proj
        if block_id == 0:
            self.before_proj = nn.Linear(hint_dim if hint_dim else self.x_dim, self.x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(self.x_dim, self.x_dim)

    def init_weights(self):
        super().init_weights()
        if self.use_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)
        if self.block_id == 0:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

    def forward(self, c: torch.Tensor, x_B_T_H_W_D: torch.Tensor, **kwargs):
        """
        stacks the previous block's output with the current block's output,
        so that the final output from the control branch is a stack of all the control block outputs,
        and easy to apply block-wise to base model.

        """
        if self.block_id == 0:
            c = self.before_proj(c) + x_B_T_H_W_D
            all_c = []
        elif self.use_after_proj:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        output = super().forward(c, **kwargs)
        c = output

        if self.use_after_proj:
            c_skip = self.after_proj(c)
            all_c += [c_skip, c]
            c = torch.stack(all_c)
        return c


class MultiViewCrossControlDiT(MultiViewCrossDiT):
    def __init__(
        self,
        *args,
        sac_config: MultiViewSACConfig = MultiViewSACConfig(),
        # transfer params
        vace_has_mask: bool = False,
        vace_block_every_n: int = 2,
        condition_strategy: Literal["spaced", "first_n"] = "spaced",
        num_max_modalities: int = 1,  # limit to 1
        use_input_hint_block: bool = False,
        spatial_compression_factor: int = 8,
        **kwargs,
    ):
        self.num_control_branches = 1
        self.separate_embedders = False

        super().__init__(
            *args,
            sac_config=sac_config,
            **kwargs,
        )

        # if vace_has_mask, the control latent is 16 + 64 (for mask)
        self.vace_has_mask = vace_has_mask
        self.num_max_modalities = num_max_modalities
        in_channels = self.in_channels - 1  # subtract the condition mask, added in MiniTrainDIT
        self.vace_in_channels = (in_channels + spatial_compression_factor**2) if vace_has_mask else in_channels
        self.vace_in_channels *= num_max_modalities
        self.vace_in_channels += 1  # adding the condition mask back
        self.use_input_hint_block = use_input_hint_block

        # for finding corresponding control block with base model block.
        self.condition_strategy = condition_strategy
        if self.condition_strategy == "spaced":
            # base block k uses the 2k'th element in the hint list, {0:0, 2:1, 4:2, ...}, as in VACE paper
            self.control_layers = [i for i in range(0, self.num_blocks, vace_block_every_n)]
            self.vace_block_id_to_base_block_id = {i: n for n, i in enumerate(self.control_layers)}
        elif self.condition_strategy == "first_n":
            # condition first n base model blocks, where n is number of control blocks
            self.control_layers = list(range(0, self.num_blocks // vace_block_every_n))
            self.vace_block_id_to_base_block_id = {i: i for i in range(len(self.control_layers))}
        else:
            raise ValueError(f"Invalid condition strategy: {self.condition_strategy}")
        assert 0 in self.control_layers

        del self.blocks
        self.blocks = nn.ModuleList(
            [
                MultiViewCrossControlAwareBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    state_t=self.state_t,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    cross_view_attn_map=self.cross_view_attn_map,
                    enable_cross_view_attn=self.enable_cross_view_attn,
                    block_id=self.vace_block_id_to_base_block_id[i] if i in self.control_layers else None,
                )
                for i in range(self.num_blocks)
            ]
        )
        nf = kwargs["model_channels"]
        hint_nf = kwargs.pop("hint_nf", [nf, nf, nf, nf, nf, nf, nf, nf])
        if self.use_input_hint_block:
            input_hint_block = []
            nonlinearity = nn.SiLU()
            for i in range(len(hint_nf) - 1):
                input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
            self.input_hint_block = nn.Sequential(*input_hint_block)

        # Replace control blocks with multiview cross-view attention versions
        self.control_blocks = nn.ModuleList(
            [
                MultiViewCrossControlEncoderBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    state_t=self.state_t,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                    cross_view_attn_map=self.cross_view_attn_map,
                    enable_cross_view_attn=self.enable_cross_view_attn,
                    block_id=i,
                    hint_dim=hint_nf[-1] if self.use_input_hint_block else None,
                    use_after_proj=True,
                )
                for i in self.control_layers
            ]
        )
        self.build_patch_embed_vace()

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)
        self.enable_selective_checkpoint(sac_config, self.control_blocks)

        # Log x_embedder
        if hasattr(self, "x_embedder"):
            x_embedder_in_channels = self.x_embedder.proj[1].in_features
            x_embedder_out_channels = self.x_embedder.proj[1].out_features
            log.debug(
                f"X_EMBEDDER - Input channels: {x_embedder_in_channels}, Output channels: {x_embedder_out_channels}"
            )
        # Log control embedder
        if hasattr(self, "control_embedder"):
            control_embedder_in_channels = self.control_embedder.proj[1].in_features
            control_embedder_out_channels = self.control_embedder.proj[1].out_features
            log.debug(
                f"CONTROL_EMBEDDER - Input channels: {control_embedder_in_channels}, Output channels: {control_embedder_out_channels}"
            )
        gc.collect()
        torch.cuda.empty_cache()

    def fully_shard(self, mesh):
        super().fully_shard(mesh)
        for i, block in enumerate(self.control_blocks):
            reshard_after_forward = i < len(self.control_blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.enable_context_parallel(process_group=process_group)

        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        for block in self.control_blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    def disable_context_parallel(self):
        # pos_embedder
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        for block in self.control_blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def init_weights(self):
        super().init_weights()

        # parent class's __init__ also calls init_weights
        if hasattr(self, "control_embedder"):
            self.control_embedder.init_weights()

        if hasattr(self, "input_hint_block"):
            for module in self.input_hint_block.modules():
                if hasattr(module, "weight"):
                    std = 1.0 / math.sqrt(module.weight.shape[0])
                    torch.nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)

        if hasattr(self, "control_blocks"):
            for block in self.control_blocks:
                block.init_weights()

    def build_patch_embed_vace(self):
        """Override to ensure control embedder handles view embedding dimensions."""
        (
            concat_padding_mask,
            base_in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.vace_in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )

        in_channels = base_in_channels
        in_channels += 1 if concat_padding_mask else 0

        if self.concat_view_embedding:
            in_channels += self.view_condition_dim

        self.control_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def prepare_embedded_sequence_for_control_branch(
        self,
        control_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare embedded sequence for control branch with multiview support."""
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(control_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            control_B_C_T_H_W = torch.cat(
                [control_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, control_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        # Determine number of cameras
        if parallel_state.is_initialized():
            process_group = parallel_state.get_context_parallel_group()
            cp_size = len(get_process_group_ranks(process_group))
            n_cameras = (control_B_C_T_H_W.shape[2] * cp_size) // self.state_t
        else:
            n_cameras = control_B_C_T_H_W.shape[2] // self.state_t

        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]

        log.debug(f"control_B_C_T_H_W shape before: {control_B_C_T_H_W.shape}")
        if self.concat_view_embedding:
            if view_indices_B_T is None:
                view_indices = torch.arange(n_cameras).clamp(max=self.n_cameras_emb - 1)
                view_indices = view_indices.to(control_B_C_T_H_W.device)
                view_embedding = self.view_embeddings(view_indices)
                view_embedding = rearrange(view_embedding, "V D -> D V")
                view_embedding = view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
            else:
                view_indices_B_T = view_indices_B_T.clamp(max=self.n_cameras_emb - 1)
                view_indices_B_T = view_indices_B_T.to(control_B_C_T_H_W.device).long()
                view_embedding = self.view_embeddings(view_indices_B_T)
                view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=n_cameras)
                view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)

            control_B_C_V_T_H_W = rearrange(control_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
            view_embedding = view_embedding.expand(
                control_B_C_V_T_H_W.shape[0],
                view_embedding.shape[1],
                view_embedding.shape[2],
                control_B_C_V_T_H_W.shape[3],
                control_B_C_V_T_H_W.shape[4],
                control_B_C_V_T_H_W.shape[5],
            )
            control_B_C_V_T_H_W = torch.cat([control_B_C_V_T_H_W, view_embedding], dim=1)
            control_B_C_T_H_W = rearrange(control_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_cameras)

        control_B_T_H_W_D = self.control_embedder(control_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_embedder = self.extra_pos_embedders_options[f"n_cameras_{n_cameras}"]
            extra_pos_emb = extra_pos_embedder(control_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return control_B_T_H_W_D, pos_embedder(control_B_T_H_W_D, fps=fps), extra_pos_emb

        control_B_T_H_W_D = control_B_T_H_W_D + pos_embedder(control_B_T_H_W_D)
        return control_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        latent_control_input: torch.Tensor,
        img_context_emb: Optional[torch.Tensor] = None,
        control_context_scale: float = 1.0,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        view_indices_B_T: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        # Deletes elements like condition.use_video_condition that are not used in the forward pass
        del kwargs
        if type(control_context_scale) == float:
            B, _, _, _, _ = x_B_C_T_H_W.shape
            control_context_scale_B_1_1_1_1 = (
                torch.ones((B, 1, 1, 1, 1), device=x_B_C_T_H_W.device) * control_context_scale
            )
            control_context_scale_B_1_1_1_1 = control_context_scale_B_1_1_1_1.to(dtype=x_B_C_T_H_W.dtype)
        else:
            control_context_scale_B_1_1_1_1 = control_context_scale.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # Control branch forward
        if latent_control_input is not None:
            # Get the original shape
            B, C, T, H, W = x_B_C_T_H_W.shape
            control_B_C_T_H_W = latent_control_input
            # Pad control input channels to match the maximum number of modalities.
            if control_B_C_T_H_W.shape[1] < self.vace_in_channels - 1:
                pad_C = self.vace_in_channels - 1 - control_B_C_T_H_W.shape[1]
                # log.info(f"Input control has {c} channels, but we need {self.vace_in_channels} channels. Padding with zeros.")
                control_B_C_T_H_W = torch.cat(
                    [
                        control_B_C_T_H_W,
                        torch.zeros(
                            (B, pad_C, T, H, W), dtype=control_B_C_T_H_W.dtype, device=control_B_C_T_H_W.device
                        ),
                    ],
                    dim=1,
                )

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
            control_B_C_T_H_W = torch.cat(
                [control_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(control_B_C_T_H_W)], dim=1
            )
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
            control_B_C_T_H_W = torch.cat(
                [
                    control_B_C_T_H_W,
                    torch.zeros((B, 1, T, H, W), dtype=control_B_C_T_H_W.dtype, device=control_B_C_T_H_W.device),
                ],
                dim=1,
            )

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            view_indices_B_T=view_indices_B_T,
        )

        # NEW code: patch emb for control branch using control patch embedder
        (
            control_B_T_H_W_D,
            rope_emb_L_1_1_D,
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        ) = self.prepare_embedded_sequence_for_control_branch(
            control_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            view_indices_B_T=view_indices_B_T,
        )
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if self.use_input_hint_block:
            control_B_T_H_W_D = self.input_hint_block(control_B_T_H_W_D)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timesteps_B_T = timesteps_B_T * self.timestep_scale

        # Use the same pattern as parent class for mixed precision training
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        if self.adaln_view_embedding:
            num_cameras = torch.unique(view_indices_B_T[0]).shape[0]
            with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                view_indices_B_V_T = rearrange(view_indices_B_T, "b (v t) -> b v t", v=num_cameras)
                view_embedding_B_V = self.adaln_view_embedder(view_indices_B_V_T[..., 0])  # B, V, D
                view_embedding_proj_B_V_9D = self.adaln_view_proj(view_embedding_B_V)  # B, V, 9D
        else:
            view_embedding_proj_B_V_9D = None

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape

        for vace_block_idx, block in enumerate(self.control_blocks):
            control_B_T_H_W_D = block(
                c=control_B_T_H_W_D,
                x_B_T_H_W_D=x_B_T_H_W_D,
                view_indices_B_T=view_indices_B_T,
                emb_B_T_D=t_embedding_B_T_D,
                view_embedding_proj_B_V_9D=view_embedding_proj_B_V_9D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                block_idx=vace_block_idx,
            )

        hints = torch.unbind(control_B_T_H_W_D)[:-1]  # list of layerwise control modulations

        for base_block_idx, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D=x_B_T_H_W_D,
                hints=hints,
                control_context_scale=control_context_scale_B_1_1_1_1,
                view_indices_B_T=view_indices_B_T,
                emb_B_T_D=t_embedding_B_T_D,
                view_embedding_proj_B_V_9D=view_embedding_proj_B_V_9D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                block_idx=base_block_idx,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp
