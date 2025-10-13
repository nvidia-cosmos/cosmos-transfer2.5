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
from collections.abc import Sequence
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
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
from cosmos_transfer2._src.predict2.networks.a2a_cp import NattenA2AAttnOp
from cosmos_transfer2._src.predict2_multiview.networks.multiview_dit import (
    MultiCameraSinCosPosEmbAxis,
    MultiCameraVideoRopePosition3DEmb,
    MultiViewCrossAttention,
)
from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
    ControlAwareDiTBlock,
    ControlEncoderDiTBlock,
    MinimalV4LVGControlVaceDiT,
    PatchEmbed,
    SACConfig,
)


class MultiViewControlAwareDiTBlock(ControlAwareDiTBlock):
    """
    A transformer block that takes n_cameras as input. This block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        state_t: int = None,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        block_id: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            block_id,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.state_t = state_t
        if image_context_dim is None:
            del self.cross_attn
            self.cross_attn = MultiViewCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                state_t=state_t,
            )
        else:
            raise NotImplementedError("image_context_dim is not supported for MultiViewBlock")


# Also do the same thing for MultiViewControlEncoderDiTBlock that inherits from ControlEncoderDiTBlock
class MultiViewControlEncoderDiTBlock(ControlEncoderDiTBlock):
    """
    A transformer block that takes n_cameras as input. This block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        state_t: int = None,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        block_id: Optional[int] = None,
        hint_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            backend,
            image_context_dim,
            block_id,
            hint_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )
        self.state_t = state_t
        if image_context_dim is None:
            del self.cross_attn
            self.cross_attn = MultiViewCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                state_t=state_t,
            )
        else:
            raise NotImplementedError("image_context_dim is not supported for MultiViewBlock")


def blocks_replace_selfattn_op_with_sparse_attn_op(
    blocks: nn.ModuleList, n_dense_blocks: int = 0, gna_parameters: Union[dict, list] = None
):
    """
    Replace the self-attention operator with a sparse self-attention operator.

    Args:
        blocks: nn.ModuleList of blocks
        n_dense_blocks: Number of blocks that will remain dense (not replaced with NeighborhoodAttention)
            If 0, all blocks use NeighborhoodAttention.
            If -1, return model directly without any modifications.
            Otherwise, n_dense_blocks blocks will remain dense, distributed evenly across the network.

    Returns:
        Modified instance
    """
    # Special case: return model directly without modifications
    if n_dense_blocks == -1:
        return

    num_blocks = len(blocks)

    if gna_parameters is None:
        raise ValueError("Please specify gna_parameters when n_dense_blocks > -1.")

    if isinstance(gna_parameters, Sequence) and len(gna_parameters) != num_blocks:
        raise ValueError(
            "List of GNA parameters must be the same length as the number of blocks, "
            f"got {len(gna_parameters)=} != {num_blocks=}."
        )

    if isinstance(gna_parameters, Sequence) and n_dense_blocks > 0:
        log.warning(f"GNA parameters was a list; ignoring {n_dense_blocks=}.")

    if isinstance(gna_parameters, Sequence):
        gna_parameters_list = gna_parameters
    else:
        if n_dense_blocks >= num_blocks:
            raise ValueError(f"n_dense_blocks ({n_dense_blocks}) must be less than the number of blocks ({num_blocks})")

        # Determine which blocks should remain dense
        dense_indices = set()

        if n_dense_blocks > 0:
            # General rule: distribute n_dense_blocks blocks evenly across the network
            if n_dense_blocks == 1:
                # Special case: just the middle block
                dense_indices.add(num_blocks // 2)
            else:
                # For multiple blocks, distribute them evenly from start to end
                indices = np.linspace(0, num_blocks - 1, n_dense_blocks, dtype=int)
                dense_indices.update(indices.tolist())

        gna_parameters_list = [None if i in dense_indices else gna_parameters for i in range(num_blocks)]

    # Replace self-attention with NeighborhoodAttention for non-dense blocks
    for i, block in enumerate(blocks):
        gna_params = gna_parameters_list[i]
        if gna_params is not None:
            gna_parameters_layer = {k: v for k, v in gna_params.items()}
            gna_parameters_layer["layer_id"] = i
            if block.self_attn.backend == "minimal_a2a":
                sparse_attn_op = NattenA2AAttnOp(gna_parameters=gna_parameters_layer)
            else:
                raise NotImplementedError(
                    f"Using sparsity with attention backend {block.self_attn.backend} is not supported."
                )
            log.info(f"Replace self-attention with sparse self-attention for block {i}")
            block.self_attn.register_module("attn_op", sparse_attn_op)
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )


class MultiViewControlDiT(MinimalV4LVGControlVaceDiT):
    def __init__(
        self,
        *args,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        # multiview params
        state_t: int,
        n_cameras_emb: int,
        view_condition_dim: int,
        concat_view_embedding: bool,
        layer_mask: Optional[List[bool]] = None,
        dense_n_blocks: int = -1,
        gna_parameters: Optional[dict] = None,
        # transfer params
        vace_has_mask: bool = False,
        vace_block_every_n: int = 2,
        condition_strategy: Literal["spaced", "first_n"] = "spaced",
        num_max_modalities: int = 8,
        use_input_hint_block: bool = False,
        sac_config: SACConfig = SACConfig(),
        use_wan_fp32_strategy: bool = False,
        **kwargs,
    ):
        self.state_t = state_t
        self.n_cameras_emb = n_cameras_emb
        self.view_condition_dim = view_condition_dim
        self.concat_view_embedding = concat_view_embedding
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += (
            self.view_condition_dim if self.concat_view_embedding else 0
        )  # this avoids overwritting build_patch_embed which still adds padding_mask channel as appropriate
        assert layer_mask is None, "layer_mask is not supported for MultiViewDiT"
        if "n_cameras" in kwargs:
            del kwargs["n_cameras"]
        super().__init__(
            *args,
            crossattn_emb_channels=crossattn_emb_channels,
            mlp_ratio=mlp_ratio,
            vace_has_mask=vace_has_mask,
            vace_block_every_n=vace_block_every_n,
            condition_strategy=condition_strategy,
            num_max_modalities=num_max_modalities,
            use_input_hint_block=use_input_hint_block,
            sac_config=sac_config,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
            **kwargs,
        )

        del self.blocks
        self.blocks = nn.ModuleList(
            [
                MultiViewControlAwareDiTBlock(
                    x_dim=self.model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    state_t=self.state_t,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers else None,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                )
                for i in range(self.num_blocks)
            ]
        )
        nf = kwargs["model_channels"]
        hint_nf = kwargs.pop("hint_nf", [nf, nf, nf, nf, nf, nf, nf, nf])
        if use_input_hint_block:
            input_hint_block = []
            nonlinearity = nn.SiLU()
            for i in range(len(hint_nf) - 1):
                input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nonlinearity]
            self.input_hint_block = nn.Sequential(*input_hint_block)
        # Replace control blocks with multiview versions
        del self.control_blocks
        self.control_blocks = nn.ModuleList(
            [
                MultiViewControlEncoderDiTBlock(
                    x_dim=self.model_channels,
                    context_dim=self.crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    backend=self.atten_backend,
                    image_context_dim=None if self.extra_image_context_dim is None else self.model_channels,
                    block_id=i,
                    hint_dim=hint_nf[-1] if use_input_hint_block else None,
                    state_t=self.state_t,
                    use_wan_fp32_strategy=self.use_wan_fp32_strategy,
                )
                for i in self.control_layers
            ]
        )

        if self.concat_view_embedding:
            self.view_embeddings = nn.Embedding(self.n_cameras_emb, view_condition_dim)

        # Replace self-attention with sparse attention if specified
        if dense_n_blocks != -1:
            log.info(
                f"MultiViewControlDiT: Replace self-attention with sparse attention for {dense_n_blocks} base blocks"
            )
            blocks_replace_selfattn_op_with_sparse_attn_op(self.blocks, dense_n_blocks, gna_parameters=gna_parameters)
        if dense_n_blocks != -1:
            log.info(
                f"MultiViewControlDiT: Replace self-attention with sparse attention for {dense_n_blocks} control blocks"
            )
            blocks_replace_selfattn_op_with_sparse_attn_op(
                self.control_blocks, dense_n_blocks, gna_parameters=gna_parameters
            )

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
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                fully_shard(extra_pos_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False)

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
            block.self_attn.set_context_parallel_group(
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
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def init_weights(self):
        self.x_embedder.init_weights()
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.init_weights()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

        # copied from transfer2
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

    def build_pos_embed(self):
        self.pos_embedder_options = nn.ModuleDict()
        self.extra_pos_embedders_options = nn.ModuleDict()
        for n_cameras in range(1, self.n_cameras_emb + 1):
            pos_embedder, extra_pos_embedder = self.build_pos_embed_for_n_cameras(n_cameras)
            self.pos_embedder_options[f"n_cameras_{n_cameras}"] = pos_embedder
            self.extra_pos_embedders_options[f"n_cameras_{n_cameras}"] = extra_pos_embedder

    def build_pos_embed_for_n_cameras(self, n_cameras: int):
        if self.pos_emb_cls == "rope3d":
            cls_type = MultiCameraVideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")
        pos_embedder, extra_pos_embedder = None, None
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            n_cameras=n_cameras,
        )
        pos_embedder = cls_type(
            **kwargs,
        )
        assert pos_embedder.enable_fps_modulation == self.rope_enable_fps_modulation, (
            "enable_fps_modulation must be the same"
        )

        if self.extra_per_block_abs_pos_emb:
            raise NotImplementedError("extra_per_block_abs_pos_emb is not tested for multi-view DIT")
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            extra_pos_embedder = MultiCameraSinCosPosEmbAxis(
                **kwargs,
            )
        return pos_embedder, extra_pos_embedder

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

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        view_indices_B_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        process_group = parallel_state.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(process_group))
        n_cameras = (x_B_C_T_H_W.shape[2] * cp_size) // self.state_t
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]
        if self.concat_view_embedding:
            if view_indices_B_T is None:
                view_indices = torch.arange(n_cameras).clamp(
                    max=self.n_cameras_emb - 1
                )  # View indices [0, 1, ..., V-1]
                view_indices = view_indices.to(x_B_C_T_H_W.device)
                view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
                view_embedding = rearrange(view_embedding, "V D -> D V")
                view_embedding = (
                    view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
                )  # Shape: [1, D, V, 1, 1, 1]
            else:
                view_indices_B_T = view_indices_B_T.clamp(max=self.n_cameras_emb - 1)
                view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
                view_embedding = self.view_embeddings(view_indices_B_T)  # B, (V T), D
                view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=n_cameras)
                view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, D, V, T, 1, 1]
            x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
            view_embedding = view_embedding.expand(
                x_B_C_V_T_H_W.shape[0],
                view_embedding.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
            x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=n_cameras)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_embedder = self.extra_pos_embedders_options[str(n_cameras)]
            extra_pos_emb = extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            raise NotImplementedError("FPS-aware positional embedding is not supported for multi-view DIT")

        x_B_T_H_W_D = x_B_T_H_W_D + pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

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
        process_group = parallel_state.get_context_parallel_group()
        cp_size = len(get_process_group_ranks(process_group))
        n_cameras = (control_B_C_T_H_W.shape[2] * cp_size) // self.state_t
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

        for block in self.control_blocks:
            control_B_T_H_W_D = block(
                c=control_B_T_H_W_D,
                x_B_T_H_W_D=x_B_T_H_W_D,
                emb_B_T_D=t_embedding_B_T_D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )

        hints = torch.unbind(control_B_T_H_W_D)[:-1]  # list of layerwise control modulations

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D=x_B_T_H_W_D,
                hints=hints,
                control_context_scale=control_context_scale_B_1_1_1_1,
                emb_B_T_D=t_embedding_B_T_D,
                crossattn_emb=context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp
