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

from typing import Optional

import torch
from projects.cosmos.sil.causal_multiview.networks.causal_cosmos_hdmap import (
    CosmosCausalHdmapDiT,  # pyrefly: ignore  # missing-import
)

from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.utils.kv_cache import KVCacheConfig, VideoSeqPos


class HDMapCausalDITwithConditionalMask(CosmosCausalHdmapDiT):
    """Adapter that exposes CosmosCausalHdmapDiT through the interactive forward_seq API.

    SIL counterpart: CosmosCausalHdmapDiT in
    projects/cosmos/sil/causal_multiview/networks/causal_cosmos_hdmap.py.
    The SIL class owns _forward_train and _forward_inference; this class bridges
    the predict2-style forward_seq call signature used by SelfForcingModel /
    denoise_edm_seq into those two SIL paths.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_cp_size(self) -> int:
        """Return the context parallel group size (1 if CP is disabled)."""
        if self._is_context_parallel_enabled and self.cp_group is not None:
            return self.cp_group.size()
        return 1

    def init_kv_cache(
        self,
        max_cache_size: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        token_h: int = 1,
        token_w: int = 1,
    ) -> None:
        """Initialize the SIL-native KV cache before each video rollout.

        SIL counterpart: CosmosCausalDiT.init_kv_cache in
        projects/cosmos/sil/causal_multiview/networks/causal_cosmos.py.
        The SIL method accepts (batch_size, max_seq_len, device, dtype) and
        pre-allocates k/v/global_end_index/local_end_index tensors for all blocks.
        This override adds the frame-count-based (max_cache_size, token_h, token_w)
        API used by generate_streaming_video, converts to token-level max_seq_len,
        then delegates to the parent.
        """
        max_seq_len = max_cache_size * token_h * token_w
        # With context parallelism, each GPU stores only its shard of the KV cache.
        cp_size = self._get_cp_size()
        if cp_size > 1:
            assert max_seq_len % cp_size == 0, f"max_seq_len ({max_seq_len}) must be divisible by cp_size ({cp_size})"
            max_seq_len = max_seq_len // cp_size
        self._kv_cache = super().init_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self._kv_max_cache_size = max_cache_size

    def precapture_cuda_graphs(self, **kwargs) -> None:
        """No-op: CUDA graphs are not supported for the CL causal model.

        SIL counterpart: there is no CUDA-graph capture in CosmosCausalHdmapDiT.
        The action-conditioned interactive CausalDIT (projects/cosmos/interactive/
        networks/dit_causal.py) does support CUDA graphs via precapture_cuda_graphs;
        this no-op maintains API compatibility with that class while doing nothing,
        matching SIL's behavior of not using CUDA graphs for the HDMap model.
        """
        pass

    def forward_seq(
        self,
        x_B_C_T_H_W: torch.Tensor,
        video_pos: VideoSeqPos,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        img_context_emb: Optional[torch.Tensor] = None,
        control_input_hdmap_bbox: Optional[torch.Tensor] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Bridge from denoise_edm_seq's forward_seq API to CosmosCausalHdmapDiT.forward().

        Delegates to the parent forward() which handles condition mask concatenation,
        timestep scaling, and dispatch to _forward_train / _forward_inference.

        For inference (kv_cache_cfg is not None), computes token-level current_start /
        current_end from the frame index in kv_cache_cfg and lets the SIL KV cache be
        updated in-place — matching the approach used in joint_causal_cosmos_model.py's
        generate_samples_from_batch.  During multi-step denoising of the same frame,
        the first forward call advances global_end_index; subsequent calls with the
        same current_start / current_end simply overwrite the same cache positions.
        The caller's commit step (with a clean frame at t=0) then overwrites those
        positions with clean K/V, exactly as the SIL inference loop does.
        """
        if kv_cache_cfg is None:
            # Full-sequence training path
            return self(
                x_B_C_T_H_W=x_B_C_T_H_W,
                timesteps_B_T=timesteps_B_T,
                crossattn_emb=crossattn_emb,
                condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
                fps=fps,
                padding_mask=padding_mask,
                data_type=data_type,
                img_context_emb=img_context_emb,
                control_input_hdmap_bbox=control_input_hdmap_bbox,
                kv_cache=None,
            )

        # Per-frame inference path — compute token-level positions from frame index,
        # matching how SIL's inference_i2v.py computes them before each forward call.
        t_idx = kv_cache_cfg.current_idx
        token_h = video_pos.H
        token_w = video_pos.W
        frame_tokens = token_h * token_w
        # With context parallelism, tokens are split across GPUs, so KV cache
        # positions must use per-GPU token counts.
        cp_size = self._get_cp_size()
        if cp_size > 1:
            assert frame_tokens % cp_size == 0, (
                f"frame_tokens ({frame_tokens}) must be divisible by cp_size ({cp_size})"
            )
            frame_tokens = frame_tokens // cp_size
        current_start = t_idx * frame_tokens
        current_end = (t_idx + 1) * frame_tokens

        return self(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            img_context_emb=img_context_emb,
            control_input_hdmap_bbox=control_input_hdmap_bbox,
            kv_cache=self._kv_cache,
            current_start=current_start,
            current_end=current_end,
            start_frame_for_rope=t_idx,
        )
