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

"""
CosmosCausalDiT: A causal DiT model combining Cosmos architecture with
block-causal attention masking and KV-caching for autoregressive generation.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Final, List, TypeAlias

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.distributed import ProcessGroup

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

RMSNorm = torch.nn.RMSNorm
# TE RMSNorm seems to be faster on single GPU?
# import transformer_engine as te

from cosmos_transfer2._src.av.bidirectional.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_transfer2._src.av.causal.fast_infer.utils.profile import NVTXRangeDecorator
from cosmos_transfer2._src.av.causal.fast_infer.v2.attn import create_attn_op
from cosmos_transfer2._src.av.causal.fast_infer.v2.minimal_v4_dit import (
    FinalLayer,
    GPT2FeedForward,
    PatchEmbed,
    TimestepEmbedding,
    Timesteps,
)

# RMSNorm = te.pytorch.RMSNorm

CameraKeyType: TypeAlias = str

DEFAULT_CAMERAS: Final[tuple[CameraKeyType, ...]] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: Final = dict(zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS))))

DEFAULT_4VIEWS_NAMES: Final[tuple[CameraKeyType, ...]] = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
DEFAULT_4VIEWS_VIEW_INDICES: list[int] = [DEFAULT_CAMERA_VIEW_MAPPING[name] for name in DEFAULT_4VIEWS_NAMES]


class TokenMemoryLayout(Enum):
    """
    Layout of tokens in memory.
    """

    V_HW_T = "V,HW,T"
    V_T_HW = "V,T,HW"


# define token layout globally for this file.
TOKEN_MEMORY_LAYOUT: TokenMemoryLayout = TokenMemoryLayout.V_T_HW


class ContextParallelDim(Enum):
    """
    Dimension to parallelize along.
    """

    V = "V"
    HW = "HW"
    T = "T"


@dataclass
class KVCacheSteadyState:
    """
    KV cache for causal self-attention with two-phase behavior and CUDA-graph support.

    Layout (fixed size): sink_token_size + local_attn_token_size.

    Caller must call before_update(current_start, input_size), then update(k, v), then
      after_update(current_start, input_size). current_start must equal _valid_end
      (contiguous append only).

    **Non-steady phase** (filling): _valid_end < cache length.
    - New k/v are appended at [_valid_end, _valid_end + input_size).

    **Steady phase** (full): _valid_end == cache length.
    - before_update: if advancing to a new chunk, rolls local window left by input_size
      so new tokens are written at the end.
    - update(k, v): writes to the last input_size positions.
    - after_update: _valid_end stays at cache length.

    Use cached_k / cached_v for attention: in non-steady returns [:, :_valid_end];
    in steady returns the full buffer.
    """

    k: Tensor  # [batch_size, sink_token_size + local_attn_token_size, n_heads, head_dim]
    v: Tensor  # [batch_size, sink_token_size + local_attn_token_size, n_heads, head_dim]
    sink_token_size: int
    local_attn_token_size: int

    _last_update_end: int = 0
    _valid_end: int = 0

    def __post_init__(self):
        assert self.k.ndim == 4, "k is expected to be 4D tensor with shape [batch_size, seq_len, n_heads, head_dim]"
        assert self.v.ndim == 4, "v is expected to be 4D tensor with shape [batch_size, seq_len, n_heads, head_dim]"
        assert self.k.shape == self.v.shape, "k and v must have the same shape"
        expected_cache_len = self.sink_token_size + self.local_attn_token_size
        assert self.k.shape[1] == expected_cache_len, (
            f"KV cache seq_len ({self.k.shape[1]}) must equal sink_token_size + local_attn_token_size "
            f"({self.sink_token_size} + {self.local_attn_token_size} = {expected_cache_len})"
        )

    def _roll_steady_inplace(self, input_size: int) -> None:
        """
        Roll the cache to the left by `input_size` tokens. This is only used in steady-state.
        """
        cache_len = self.k.shape[1]
        assert cache_len == self._valid_end, (
            f"We expect the cache to be full, but got {cache_len=} != {self._valid_end=}"
        )
        last_n_to_keep = self.local_attn_token_size - input_size

        # Source: tokens after the ones we're discarding
        src_start = self.sink_token_size + input_size
        src_end = cache_len

        # Dest: right after sink tokens
        dst_start = self.sink_token_size
        dst_end = self.sink_token_size + last_n_to_keep

        self.k[:, dst_start:dst_end] = self.k[:, src_start:src_end].clone()
        self.v[:, dst_start:dst_end] = self.v[:, src_start:src_end].clone()

    def _roll_nonsteady_inplace(self, current_start: int) -> None:
        """
        In non-steady state we only allow contiguous append; no physical roll.
        Asserts that current_start == _valid_end.
        """
        assert current_start == self._valid_end, (
            "In non-steady state we expect contiguous append: "
            f"current_start ({current_start}) must equal _valid_end ({self._valid_end})"
        )

    def _update_steady_inplace(self, k: Tensor, v: Tensor) -> None:
        """
        Update the cache in place in steady-state.
        """
        input_size = k.shape[1]
        cache_len = self.k.shape[1]
        assert cache_len == self._valid_end, (
            f"We expect the cache to be full, but got {cache_len=} != {self._valid_end=}"
        )
        self.k[:, cache_len - input_size : cache_len] = k
        self.v[:, cache_len - input_size : cache_len] = v

    def _update_nonsteady_inplace(self, k: Tensor, v: Tensor) -> None:
        """
        Update the cache in place in non-steady state.
        """
        input_size = k.shape[1]
        current_start = self._valid_end
        self.k[:, current_start : current_start + input_size] = k
        self.v[:, current_start : current_start + input_size] = v

    def is_steady_state(self) -> bool:
        """
        Define the steady state as when cache is full.
        """
        cache_len = self.k.shape[1]
        return cache_len == self._valid_end

    def before_update(self, current_start: int, input_size: int) -> None:
        """
        Call before update(): rolls in steady state when advancing to a new chunk;
        in non-steady state only asserts contiguous append (current_start == _valid_end).
        No-op when re-updating the same chunk (_last_update_end == current_start + input_size).
        """
        end = current_start + input_size
        if self._last_update_end == end:
            # The current update location is the same as the last update location.
            # So we just need to update the cache in place. Nothing need to be done
            # before the update.
            return
        else:
            # The current update location is different from the last update location.
            # We assume the new update is exactly one chunk (input_size) after the last.
            assert end == self._last_update_end + input_size, f"{end=} != {self._last_update_end=} + {input_size=}"
            if self.is_steady_state():
                self._roll_steady_inplace(input_size)
            else:
                self._roll_nonsteady_inplace(current_start)

    def update(self, k: Tensor, v: Tensor) -> None:
        """
        Write k/v into the cache. Steady: last input_size positions; non-steady: append at _valid_end.
        Call after before_update() and before after_update().
        """
        if self.is_steady_state():
            self._update_steady_inplace(k, v)
        else:
            self._update_nonsteady_inplace(k, v)

    def after_update(self, current_start: int, input_size: int) -> None:
        """
        Call after update(): sets _last_update_end and, in non-steady state, _valid_end.
        """
        end = current_start + input_size
        self._last_update_end = end
        if self.is_steady_state():
            # In steady state, the valid end won't be changed.
            pass
        else:
            # In non-steady state, the valid end is the end of the update.
            self._valid_end = end

    def cached_k(self, input_size: int) -> Tensor:
        """Keys to use for attention

        Note this will be called in attention operator, before we can call `after_update`.
        So the valid end is actually self._valid_end + input_size [for eager mode]
        """
        if self.is_steady_state():
            return self.k
        return self.k[:, : self._valid_end + input_size]

    def cached_v(self, input_size: int) -> Tensor:
        """Values to use for attention

        Note this will be called in attention operator, before we can call `after_update`.
        So the valid end is actually self._valid_end + input_size [for eager mode]
        """
        if self.is_steady_state():
            return self.v
        return self.v[:, : self._valid_end + input_size]

    def reset(self) -> None:
        """Reset the cache."""
        self._last_update_end = 0
        self._valid_end = 0


@dataclass
class AttentionBlockKVCache:
    self_attn: KVCacheSteadyState
    cross_attn: KVCacheSteadyState

    def before_update(self, self_attn_seq_start: int, self_attn_seq_len: int) -> None:
        self.self_attn.before_update(self_attn_seq_start, self_attn_seq_len)

    def after_update(self, self_attn_seq_start: int, self_attn_seq_len: int) -> None:
        self.self_attn.after_update(self_attn_seq_start, self_attn_seq_len)


@dataclass
class CosmosCausalDiTNetworkCache:
    block_kv_caches: List[AttentionBlockKVCache]

    def __getitem__(self, index: int) -> AttentionBlockKVCache:
        return self.block_kv_caches[index]

    def before_update(self, self_attn_seq_start: int, self_attn_seq_len: int) -> None:
        for block_cache in self.block_kv_caches:
            block_cache.before_update(self_attn_seq_start, self_attn_seq_len)

    def after_update(self, self_attn_seq_start: int, self_attn_seq_len: int) -> None:
        for block_cache in self.block_kv_caches:
            block_cache.after_update(self_attn_seq_start, self_attn_seq_len)


class SinsoidalRotaryFrequency:
    r"""
    Sinsoidal rotary frequency for 3D video.
    """

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.head_dim = head_dim
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        seq_t = torch.arange(len_t, dtype=torch.float32)
        seq_h = torch.arange(len_h, dtype=torch.float32)
        seq_w = torch.arange(len_w, dtype=torch.float32)

        dim_spatial_range = torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)] / dim_h
        dim_temporal_range = torch.arange(0, dim_t, 2, dtype=torch.float32)[: (dim_t // 2)] / dim_t

        h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**dim_temporal_range)
        self.temporal_freqs = temporal_freqs.to(device)  # [D]

        # align with the patchify pattern
        if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
            freqs_h = repeat(torch.outer(seq_h, h_spatial_freqs), "h d -> (h w t) 1 1 d", t=len_t, w=len_w)
            freqs_w = repeat(torch.outer(seq_w, w_spatial_freqs), "w d -> (h w t) 1 1 d", t=len_t, h=len_h)
            freqs_t = repeat(torch.outer(seq_t, temporal_freqs), "t d -> (h w t) 1 1 d", h=len_h, w=len_w)
        elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
            freqs_t = repeat(torch.outer(seq_t, temporal_freqs), "t d -> (t h w) 1 1 d", h=len_h, w=len_w)
            freqs_h = repeat(torch.outer(seq_h, h_spatial_freqs), "h d -> (t h w) 1 1 d", t=len_t, w=len_w)
            freqs_w = repeat(torch.outer(seq_w, w_spatial_freqs), "w d -> (t h w) 1 1 d", t=len_t, h=len_h)
        else:
            assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"
        self.freqs_h = freqs_h.to(device)
        self.freqs_w = freqs_w.to(device)
        self.freqs_t = freqs_t.to(device)

        self._cp_enabled = False
        self._freqs_t_cp: Tensor | None = None
        self._freqs_h_cp: Tensor | None = None
        self._freqs_w_cp: Tensor | None = None

    def shift_t(self, offset: int) -> Tensor:
        r"""
        Shift the time dimension by the given offset.
        """
        if self._cp_enabled:
            freqs_t = self._freqs_t_cp + offset * self.temporal_freqs
            freqs_h = self._freqs_h_cp
            freqs_w = self._freqs_w_cp
        else:
            freqs_t = self.freqs_t + offset * self.temporal_freqs
            freqs_h = self.freqs_h
            freqs_w = self.freqs_w
        freqs = torch.cat([freqs_t, freqs_h, freqs_w] * 2, dim=-1)
        return freqs  # [L, 1, 1, D]

    def set_context_parallel_group(
        self, process_group: ProcessGroup | None = None, cp_dim: ContextParallelDim | None = None
    ):
        if process_group is None:
            self._cp_enabled = False
            return

        cp_size = process_group.size()

        if cp_dim == ContextParallelDim.T:
            # split the frequency tensors along the T dimension
            if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
                freqs_t = rearrange(self.freqs_t, "(hw t) 1 1 d -> hw t 1 1 d", t=self.len_t)
                freqs_h = rearrange(self.freqs_h, "(hw t) 1 1 d -> hw t 1 1 d", t=self.len_t)
                freqs_w = rearrange(self.freqs_w, "(hw t) 1 1 d -> hw t 1 1 d", t=self.len_t)
                if freqs_t.shape[1] % cp_size != 0:
                    raise ValueError(
                        f"Frequency tensor T dimension {freqs_t.shape[1]} cannot be divisible by cp_size {cp_size}",
                        f"Please consider using TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW instead.",
                    )
                freqs_t_cp = split_inputs_cp(freqs_t, seq_dim=1, cp_group=process_group)
                freqs_h_cp = split_inputs_cp(freqs_h, seq_dim=1, cp_group=process_group)
                freqs_w_cp = split_inputs_cp(freqs_w, seq_dim=1, cp_group=process_group)
                self._freqs_t_cp = rearrange(freqs_t_cp, "hw t 1 1 d -> (hw t) 1 1 d")
                self._freqs_h_cp = rearrange(freqs_h_cp, "hw t 1 1 d -> (hw t) 1 1 d")
                self._freqs_w_cp = rearrange(freqs_w_cp, "hw t 1 1 d -> (hw t) 1 1 d")
            elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
                # in this memory layout, split along T is effectively the same as split along L=T*H*W.
                self._freqs_t_cp = split_inputs_cp(self.freqs_t, seq_dim=0, cp_group=process_group)
                self._freqs_h_cp = split_inputs_cp(self.freqs_h, seq_dim=0, cp_group=process_group)
                self._freqs_w_cp = split_inputs_cp(self.freqs_w, seq_dim=0, cp_group=process_group)
            else:
                assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"
        elif cp_dim == ContextParallelDim.HW:
            # split the frequency tensors along the HW dimension
            if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
                # in this memory layout, split along HW is effectively the same as split along L=T*H*W.
                self._freqs_t_cp = split_inputs_cp(self.freqs_t, seq_dim=0, cp_group=process_group)
                self._freqs_h_cp = split_inputs_cp(self.freqs_h, seq_dim=0, cp_group=process_group)
                self._freqs_w_cp = split_inputs_cp(self.freqs_w, seq_dim=0, cp_group=process_group)
            elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
                freqs_t = rearrange(self.freqs_t, "(t hw) 1 1 d -> t hw 1 1 d", t=self.len_t)
                freqs_h = rearrange(self.freqs_h, "(t hw) 1 1 d -> t hw 1 1 d", t=self.len_t)
                freqs_w = rearrange(self.freqs_w, "(t hw) 1 1 d -> t hw 1 1 d", t=self.len_t)
                if freqs_t.shape[1] % cp_size != 0:
                    raise ValueError(
                        f"Frequency tensor HW dimension {freqs_t.shape[1]} cannot be divisible by cp_size {cp_size}",
                        f"Please consider using TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T instead.",
                    )
                freqs_t_cp = split_inputs_cp(freqs_t, seq_dim=1, cp_group=process_group)
                freqs_h_cp = split_inputs_cp(freqs_h, seq_dim=1, cp_group=process_group)
                freqs_w_cp = split_inputs_cp(freqs_w, seq_dim=1, cp_group=process_group)
                self._freqs_t_cp = rearrange(freqs_t_cp, "t hw 1 1 d -> (t hw) 1 1 d")
                self._freqs_h_cp = rearrange(freqs_h_cp, "t hw 1 1 d -> (t hw) 1 1 d")
                self._freqs_w_cp = rearrange(freqs_w_cp, "t hw 1 1 d -> (t hw) 1 1 d")
        else:
            assert False, f"Invalid context parallel dimension: {cp_dim}"
        self._cp_enabled = True


class SelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        # QKV projections
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

        # QK normalization
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # Attention operator for inference (no mask)
        self.attn_op = create_attn_op(
            num_heads=n_heads,
            head_dim=head_dim,
            choice="Torch",  # options: "TE", "Torch"
        )
        self._cp_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        rope_emb: torch.Tensor,
        kv_cache: KVCacheSteadyState,
    ) -> torch.Tensor:
        """
        Forward pass with block-causal attention or KV-caching.

        Args:
            x: The query tensor of shape [..., L, D]
            rope_emb: RoPE cosine and sine embeddings [L, 1, 1, D]
            kv_cache: KV cache for inference

        Output:
            The output tensor of shape [..., L, D]

        Note:
            In graph mode, this always writes new k/v to the last positions of the cache.
            The roll operation (for new timesteps) should be done OUTSIDE this function
            before calling it, using kv_cache.roll(input_size).
        """
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim

        # Compute Q, K, V
        q = self.q_norm(self.q_proj(x).reshape(batch_size, L, n, d))
        k = self.k_norm(self.k_proj(x).reshape(batch_size, L, n, d))
        v = self.v_proj(x).reshape(batch_size, L, n, d)

        # Inference mode with KV caching
        roped_q = apply_rotary_pos_emb(q, rope_emb, tensor_format="bshd", fused=True)
        roped_k = apply_rotary_pos_emb(k, rope_emb, tensor_format="bshd", fused=True)

        # Note: CUDA graph-compatible:
        # Call before_update(current_start, input_size) before updating the kv cache.
        kv_cache.update(roped_k, v)
        # Call after_update(current_start, input_size) after updating the kv cache.

        cached_k = kv_cache.cached_k(input_size=L)
        cached_v = kv_cache.cached_v(input_size=L)
        out = self.attn_op(roped_q, cached_k, cached_v)
        out = out.reshape(batch_shape + (L, n * d))
        return self.output_proj(out)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None):
        self.attn_op.set_context_parallel_group(cp_group=cp_group)
        self._cp_enabled = cp_group is not None

    def enable_all_backends(self) -> None:
        self.attn_op.enable_all_backends()

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    def prepare_cache(
        self,
        batch_size: int,
        sink_token_size: int,
        local_attn_token_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> KVCacheSteadyState:
        total_token_size = sink_token_size + local_attn_token_size
        k = torch.randn(batch_size, total_token_size, self.n_heads, self.head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, total_token_size, self.n_heads, self.head_dim, device=device, dtype=dtype)
        return KVCacheSteadyState(
            k=k,
            v=v,
            sink_token_size=sink_token_size,
            local_attn_token_size=local_attn_token_size,
        )


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

        self.attn_op = create_attn_op(
            num_heads=n_heads,
            head_dim=head_dim,
            choice="Torch",  # options: "TE", "Torch"
        )
        self._cp_enabled = False
        self._cp_size = 1

    def set_context_parallel_group(self, cp_group: ProcessGroup | None):
        self.attn_op.set_context_parallel_group(cp_group=cp_group)
        self._cp_enabled = cp_group is not None
        self._cp_size = cp_group.size() if cp_group is not None else 1

    def enable_all_backends(self) -> None:
        self.attn_op.enable_all_backends()

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    @property
    def cp_size(self) -> int:
        return self._cp_size

    def prepare_cache(
        self,
        context: torch.Tensor,  # [..., L, D]
    ) -> KVCacheSteadyState:
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = context.shape[-2:]
        context_BV_L_D = context.reshape(batch_size, L, D)
        k = self.k_proj(context_BV_L_D)
        k = k.reshape(batch_size, L, self.n_heads, self.head_dim)
        k = self.k_norm(k)
        v = self.v_proj(context_BV_L_D)
        v = rearrange(v, "... (h d) -> ... h d", h=self.n_heads, d=self.head_dim)
        return KVCacheSteadyState(k=k, v=v, sink_token_size=0, local_attn_token_size=L)

    def forward(
        self,
        x: torch.Tensor,  # [..., L, D]
        kv_cache: KVCacheSteadyState,
    ) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, D = x.shape[-2:]
        n, d = self.n_heads, self.head_dim
        q = self.q_norm(self.q_proj(x).reshape(batch_size, L, n, d))
        out = self.attn_op(q, kv_cache.k, kv_cache.v).reshape(batch_shape + (L, n * d))
        return self.output_proj(out)


class CausalCosmosBlock(nn.Module):
    """
    Transformer block with causal self-attention, cross-attention, and MLP.

    Uses AdaLN modulation for timestep conditioning following the Cosmos architecture.
    API aligned with Block class from minimal_v4_dit.py.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        enable_cross_view_attn: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.enable_cross_view_attn = enable_cross_view_attn

        # Self-attention with causal masking
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(
            query_dim=x_dim,
            context_dim=None,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
        )

        # Cross-attention (using standard Attention from minimal_v4_dit)
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            query_dim=x_dim,
            context_dim=context_dim,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
        )

        # MLP
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        # AdaLN modulation
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        if enable_cross_view_attn:
            # dense cross view attention
            self.cross_view_attn = CrossAttention(
                query_dim=x_dim,
                context_dim=x_dim,
                n_heads=num_heads,
                head_dim=x_dim // num_heads,
            )
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)

    def prepare_cache(
        self, n_views: int, sink_token_size: int, local_attn_token_size: int, context: torch.Tensor
    ) -> AttentionBlockKVCache:
        self_attn_kv_cache = self.self_attn.prepare_cache(
            n_views, sink_token_size, local_attn_token_size, device=context.device, dtype=context.dtype
        )
        cross_attn_kv_cache = self.cross_attn.prepare_cache(context)
        return AttentionBlockKVCache(self_attn=self_attn_kv_cache, cross_attn=cross_attn_kv_cache)

    def refresh_cache(self, cache: AttentionBlockKVCache, context: torch.Tensor) -> None:
        """Refresh the cross-attention and self-attention cache with the new context."""
        # self-attention cache
        cache.self_attn.reset()
        # cross-attention cache
        cross_attn_kv_cache = self.cross_attn.prepare_cache(context)
        cache.cross_attn.k.copy_(cross_attn_kv_cache.k)
        cache.cross_attn.v.copy_(cross_attn_kv_cache.v)

    def set_context_parallel_group(
        self, self_attn_group: ProcessGroup | None, cross_view_attn_group: ProcessGroup | None = None
    ):
        """Set hierarchical CP groups for self-attention and cross-view attention.

        Args:
            self_attn_group: Group for ranks processing same view (for T gathering in self-attention)
            cross_view_attn_group: Group for ranks at same T slice (for V gathering in cross-view)
        """
        # Self-attention uses self_attn_group (for T gathering)
        self.self_attn.set_context_parallel_group(cp_group=self_attn_group)
        # Cross-view attention uses cross_view_attn_group (for V gathering)
        if self.enable_cross_view_attn:
            self.cross_view_attn.set_context_parallel_group(cp_group=cross_view_attn_group)

    def enable_all_attn_backends(self) -> None:
        self.self_attn.enable_all_backends()
        self.cross_attn.enable_all_backends()
        if self.enable_cross_view_attn:
            self.cross_view_attn.enable_all_backends()

    def enable_cudnn_manual_ring(self) -> None:
        self.self_attn.attn_op.enable_cudnn_manual_ring = True
        self.cross_attn.attn_op.enable_cudnn_manual_ring = True
        if self.enable_cross_view_attn:
            self.cross_view_attn.attn_op.enable_cudnn_manual_ring = True

    def enable_cudnn_manual_ulysses(self) -> None:
        self.self_attn.attn_op.enable_cudnn_manual_ulysses = True
        self.cross_attn.attn_op.enable_cudnn_manual_ulysses = True
        if self.enable_cross_view_attn:
            self.cross_view_attn.attn_op.enable_cudnn_manual_ulysses = True

    def forward(
        self,
        x_B_Ellipsis_D: torch.Tensor,  # [B, <TOKEN_MEMORY_LAYOUT>, D]
        emb_B_D: torch.Tensor,
        block_kv_cache: AttentionBlockKVCache,
        rope_emb: torch.Tensor,  # [L, 1, 1, D]
        adaln_lora_B_3D: torch.Tensor | None = None,
        view_embedding_proj_B_V_9D: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x: Input tensor [B, ..., D], If TOKEN_MEMORY_LAYOUT is VHWT, then shape is [B, V, HW, T, D],
                if TOKEN_MEMORY_LAYOUT is VTHW, then shape is [B, V, T, HW, D]
            emb_B_D: Time embedding [B, D]
            rope_emb: RoPE cosine and sine embeddings [L, 1, 1, D]
            adaln_lora_B_3D: AdaLN LoRA embeddings [B, 3D]
            block_kv_cache: AttentionBlockKVCache
            view_embedding_proj_B_V_9D: View embedding projection [B, V, 9D]
        """
        B, *ellipsis_shape, D = x_B_Ellipsis_D.shape
        if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
            assert len(ellipsis_shape) == 3, f"Expected 3 dimensions in ellipsis_shape, but got {ellipsis_shape=}"
            V, HW, T = ellipsis_shape
            ellipsis_shape_str = "v hw t"
        elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
            assert len(ellipsis_shape) == 3, f"Expected 3 dimensions in ellipsis_shape, but got {ellipsis_shape=}"
            V, T, HW = ellipsis_shape
            ellipsis_shape_str = "v t hw"
        else:
            assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"

        # reshape embeddings to be broadcastable with x.
        emb_B_Ellipsis_D = emb_B_D.reshape(B, *([1] * len(ellipsis_shape)), D)

        # Compute AdaLN modulation
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None, "adaln_lora_B_3D is required when use_adaln_lora is True"
            adaln_lora_B_Ellipsis_3D = adaln_lora_B_3D.reshape(B, *([1] * len(ellipsis_shape)), 3 * D)
            shift_self, scale_self, gate_self = (
                self.adaln_modulation_self_attn(emb_B_Ellipsis_D) + adaln_lora_B_Ellipsis_3D
            ).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = (
                self.adaln_modulation_cross_attn(emb_B_Ellipsis_D) + adaln_lora_B_Ellipsis_3D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                self.adaln_modulation_mlp(emb_B_Ellipsis_D) + adaln_lora_B_Ellipsis_3D
            ).chunk(3, dim=-1)
        else:
            shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_B_Ellipsis_D).chunk(3, dim=-1)
            shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_B_Ellipsis_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_Ellipsis_D).chunk(3, dim=-1)

        if self.enable_cross_view_attn:
            assert view_embedding_proj_B_V_9D is not None
            (
                view_shift_self,
                view_scale_self,
                view_gate_self,
                view_shift_cross,
                view_scale_cross,
                view_gate_cross,
                view_shift_mlp,
                view_scale_mlp,
                view_gate_mlp,
            ) = view_embedding_proj_B_V_9D.chunk(9, dim=-1)

            def expand_view_mod(v_mod: torch.Tensor) -> torch.Tensor:
                return v_mod.reshape(B, V, 1, 1, D)  # B_Ellipsis_D

            shift_self = shift_self + expand_view_mod(view_shift_self)
            scale_self = scale_self + expand_view_mod(view_scale_self)
            gate_self = gate_self + expand_view_mod(view_gate_self)

            shift_cross = shift_cross + expand_view_mod(view_shift_cross)
            scale_cross = scale_cross + expand_view_mod(view_scale_cross)
            gate_cross = gate_cross + expand_view_mod(view_gate_cross)

            shift_mlp = shift_mlp + expand_view_mod(view_shift_mlp)
            scale_mlp = scale_mlp + expand_view_mod(view_scale_mlp)
            gate_mlp = gate_mlp + expand_view_mod(view_gate_mlp)

        # Self-attention (API aligned with Attention.forward)
        normed_x = self.layer_norm_self_attn(x_B_Ellipsis_D) * (1 + scale_self) + shift_self
        attn_out = self.self_attn(
            normed_x.reshape(B, V, -1, D),
            rope_emb=rope_emb,
            kv_cache=block_kv_cache.self_attn,
        ).reshape_as(normed_x)
        x_B_Ellipsis_D = x_B_Ellipsis_D + gate_self * attn_out

        # Cross-view attention: dense
        if self.enable_cross_view_attn:
            normed_x_cv = self.layer_norm_cross_view_attn(x_B_Ellipsis_D)
            x = rearrange(normed_x_cv, f"b {ellipsis_shape_str} d -> b t v hw d")
            if self.cross_view_attn.cp_enabled:
                # Note: we cross view attention is CP enabled, we assume multi-view is split across GPU
                # ranks IN ORDER. E.g., for 4 views on 2 GPUs, we assume the groups are [0, 1] views and [2, 3] views.
                if V == 1:
                    # When cross attention is CP enabled, and the CP size is equal to the number of views,
                    # then each gpu processes exactly one view. Since attention will gather
                    # all KV from all gpus, each gpu effectively only need to process KV for its own view.
                    x_context = x
                    # effectively same as the following, but since V=1 it results in the same tensor.
                    # x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
                else:
                    # When CP size is less than the number of views, e.g., for 4 views on 2 GPUs,
                    # each gpu processes multiple views. We can still rely on attention to gather
                    # all KV from all gpus, but in this case KV on each gpu should cover 4/2 = 2 views.
                    x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
            else:
                # When cross attention is not CP enabled, we need to repeat the context
                # to match the number of views. such that the attention will be computed
                # across all views.
                x_context = repeat(x, f"b t v hw d -> b t v2 (v hw) d", v2=V)
            cross_attn_kv_cache = self.cross_view_attn.prepare_cache(x_context)
            cv_out = self.cross_view_attn(x, kv_cache=cross_attn_kv_cache)
            cv_out = rearrange(cv_out, f"b t v hw d -> b {ellipsis_shape_str} d")
            x_B_Ellipsis_D = x_B_Ellipsis_D + cv_out

        # Cross-attention
        normed_x = self.layer_norm_cross_attn(x_B_Ellipsis_D) * (1 + scale_cross) + shift_cross
        cross_out = self.cross_attn(normed_x.reshape(B, V, -1, D), kv_cache=block_kv_cache.cross_attn).reshape_as(
            normed_x
        )
        x_B_Ellipsis_D = x_B_Ellipsis_D + gate_cross * cross_out

        # MLP
        normed_x = self.layer_norm_mlp(x_B_Ellipsis_D) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(normed_x)
        x_B_Ellipsis_D = x_B_Ellipsis_D + gate_mlp * mlp_out

        return x_B_Ellipsis_D


class CosmosCausalDiT(nn.Module):
    """
    Causal DiT model for video generation with block-causal attention and KV-caching.

    Combines the Cosmos DiT architecture with causal attention masking for
    autoregressive video generation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        timestep_scale: float = 1.0,
        additional_concat_ch: int = 0,  # hdmap
        enable_cross_view_attn: bool = False,
        # multiview embedding
        view_condition_dim: int = 16,
        n_cameras_emb: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.timestep_scale = timestep_scale
        # add 1 for the condition mask
        self.in_channels = in_channels + 1
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask

        # Positional embedding settings
        self.additional_concat_ch = additional_concat_ch

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels
        self.enable_cross_view_attn = enable_cross_view_attn
        # Build embeddings
        self._build_patch_embed()

        # Time embeddings
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Transformer blocks (API aligned with Block from minimal_v4_dit)
        self.blocks = nn.ModuleList(
            [
                CausalCosmosBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    enable_cross_view_attn=enable_cross_view_attn,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        if enable_cross_view_attn:
            self.adaln_view_embedder = nn.Embedding(n_cameras_emb, model_channels)
            self.adaln_view_proj = nn.Linear(model_channels, model_channels * 9)
        else:
            self.adaln_view_embedder = None
            self.adaln_view_proj = None

        self._is_shuffle_op_fused = False
        self._is_padding_mask_fused = False
        self._parameters_updated_after_loading_checkpoint = False

    def prepare_cache(
        self, n_views: int, sink_token_size: int, local_attn_token_size: int, context: torch.Tensor
    ) -> CosmosCausalDiTNetworkCache:
        # Context embeddings
        if self.use_crossattn_projection:
            context = self.crossattn_proj(context)

        block_kv_caches: list[AttentionBlockKVCache] = []
        for block in self.blocks:
            block_kv_cache = block.prepare_cache(n_views, sink_token_size, local_attn_token_size, context)
            block_kv_caches.append(block_kv_cache)
        return CosmosCausalDiTNetworkCache(block_kv_caches=block_kv_caches)

    def refresh_cache(self, cache: CosmosCausalDiTNetworkCache, context: torch.Tensor) -> None:
        """Update the cache with the new context."""
        if self.use_crossattn_projection:
            context = self.crossattn_proj(context)

        for block_idx, block in enumerate(self.blocks):
            block.refresh_cache(cache[block_idx], context)

    def _fuse_shuffle_op_into_last_layer(self):
        """
        In the Cosmos model, the patchify operation is
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",

        while the unpatchify operation is
        "b (t h w) (kt kh kw c) -> b c (t kt) (h kh) (w kw)"

        This is likely a bug in the Cosmos model where the last dimension is shuffled after the network.

        To fix this, we could fuse this shuffle op into the last linear layer,
        so that we do not have to do this shuffle op explicitly before returning the result.

        Calling this function to modify the last layer in place, is equivalent to the following code
        after the last layer:
        ```python
        x = rearrange(
            x,
            "... (kt kh kw c) -> ... (c kt kh kw)",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
            c=self.out_channels,
        )
        ```
        """
        if self._is_shuffle_op_fused:
            return

        self.final_layer.linear.weight.data = rearrange(
            self.final_layer.linear.weight,
            "(kt kh kw c) in_dim -> (c kt kh kw) in_dim",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
            c=self.out_channels,
        ).contiguous()
        if self.final_layer.linear.bias is not None:
            self.final_layer.linear.bias.data = rearrange(
                self.final_layer.linear.bias,
                "(kt kh kw c) -> (c kt kh kw)",
                kt=self.patch_temporal,
                kh=self.patch_spatial,
                kw=self.patch_spatial,
                c=self.out_channels,
            ).contiguous()

        self._is_shuffle_op_fused = True
        return

    def _fuse_padding_mask_into_patch_embed(self) -> None:
        """
        Fuse the padding mask into the patch embedder in place.

        If `self.concat_padding_mask` is True, during training we are concatenating a
        padding_mask with shape [B, 1, T, H, W] to the input x_B_C_T_H_W on the C dimension,
        before passing it into the self.x_embedder. This is to work with data with different
        spatial resolutions during training, where `1` indicates padded regions. During
        inference, the padding_mask is always 0. So here we could simply remove the corresponding
        channels in the x_embedder in place

        Calling this function to modify the patch embedder in place, is equivalent to the following code
        before passing the input into the patch embedder:
        ```python
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask], dim=1)
        ```
        """
        if not self.concat_padding_mask:
            return

        if self._is_padding_mask_fused:
            return

        self.x_embedder.in_channels -= 1
        in_channels_to_keep = self.x_embedder.get_linear_in_channels()
        self.x_embedder.proj[1].weight.data = self.x_embedder.proj[1].weight.data[:, :in_channels_to_keep].contiguous()
        if self.x_embedder.proj[1].bias is not None:
            self.x_embedder.proj[1].bias.data = self.x_embedder.proj[1].bias.data[:in_channels_to_keep].contiguous()

        self._is_padding_mask_fused = True
        return

    def update_parameters_after_loading_checkpoint(self) -> None:
        # This function should be called after loading the checkpoint, to fuse some operations in the model
        # weights to reduce computation during inference.
        if self._parameters_updated_after_loading_checkpoint:
            return

        self._fuse_padding_mask_into_patch_embed()
        self._fuse_shuffle_op_into_last_layer()
        self._parameters_updated_after_loading_checkpoint = True

    def _build_patch_embed(self) -> None:
        in_ch = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_ch,
            out_channels=self.model_channels,
        )
        # HDMap conditioning
        if self.additional_concat_ch > 0:
            self.additional_patch_embedding = PatchEmbed(
                spatial_patch_size=self.patch_spatial,
                temporal_patch_size=self.patch_temporal,
                in_channels=self.additional_concat_ch,
                out_channels=self.model_channels,
            )

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,  # [B, V, C, T, H, W]
        process_groups: List[ProcessGroup | None] | None = None,
        cp_dims: List[ContextParallelDim | None] | None = None,
    ) -> Tensor:
        r"""
        Patchify the input tensor and maybe split it along cp_dim if a process group is provided.

        The patchify pattern is:
            "b v c (t kt) (h kh) (w kw) -> b v (h w) t (c kt kh kw)",
        Or:
            "b v c (t kt) (h kh) (w kw) -> b v t (h w) (c kt kh kw)",
        depending on the TOKEN_MEMORY_LAYOUT.

        Returns:
            Tensor: The patched tensor with shape [B, <TOKEN_MEMORY_LAYOUT>, D]
        """
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"

        if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
            token_shape_str = "v (h w) t"
        elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
            token_shape_str = "v t (h w)"
        else:
            assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"
        x = rearrange(
            x,
            f"... v c (t kt) (h kh) (w kw) -> ... {token_shape_str} (c kt kh kw)",
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
        )

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided"
                "and the number of dimensions must match the number of process groups"
            )
            for cp_dim, process_group in zip(cp_dims, process_groups):
                if process_group is not None:
                    assert cp_dim is not None, (
                        "Context parallel dimension must be provided if process group is provided"
                    )
                    if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
                        seq_dim = {
                            ContextParallelDim.V: 1,
                            ContextParallelDim.HW: 2,
                            ContextParallelDim.T: 3,
                        }[cp_dim]
                    elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
                        seq_dim = {
                            ContextParallelDim.V: 1,
                            ContextParallelDim.T: 2,
                            ContextParallelDim.HW: 3,
                        }[cp_dim]
                    else:
                        assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"
                    x = split_inputs_cp(x, seq_dim=seq_dim, cp_group=process_group)
        return x

    def unpatchify_and_maybe_gather_cp(
        self,
        pH: int,
        pW: int,
        x: Tensor,  # [B, <TOKEN_MEMORY_LAYOUT>, D]
        process_groups: List[ProcessGroup | None] | None = None,
        cp_dims: List[ContextParallelDim | None] | None = None,
    ) -> Tensor:
        r"""
        Unpatchify the input tensor and maybe gather it along cp_dim if a process group is provided.

        The unpatchify pattern is:
            "b v (h w) t (c kt kh kw) -> b v c (t kt) (h kh) (w kw)",
        Or:
            "b v t (h w) (c kt kh kw) -> b v c (t kt) (h kh) (w kw)",
        depending on the TOKEN_MEMORY_LAYOUT.

        Returns:
            Tensor: The unpatched tensor with shape [B, V, C, T, H, W]
        """
        assert x.ndim == 5, f"x must be a 5D tensor, but got shape {x.shape}"

        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups), (
                "Context parallel dimensions and process groups must be provided"
                "and the number of dimensions must match the number of process groups"
            )
            for cp_dim, process_group in zip(cp_dims, process_groups):
                if process_group is not None:
                    assert cp_dim is not None, (
                        "Context parallel dimension must be provided if process group is provided"
                    )
                    if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
                        seq_dim = {
                            ContextParallelDim.V: 1,
                            ContextParallelDim.HW: 2,
                            ContextParallelDim.T: 3,
                        }[cp_dim]
                    elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
                        seq_dim = {
                            ContextParallelDim.V: 1,
                            ContextParallelDim.T: 2,
                            ContextParallelDim.HW: 3,
                        }[cp_dim]
                    else:
                        assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"
                    x = cat_outputs_cp(x, seq_dim=seq_dim, cp_group=process_group)

        if TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_HW_T:
            token_shape_str = "v (h w) t"
        elif TOKEN_MEMORY_LAYOUT == TokenMemoryLayout.V_T_HW:
            token_shape_str = "v t (h w)"
        else:
            assert False, f"Invalid token memory layout: {TOKEN_MEMORY_LAYOUT}"
        x = rearrange(
            x,
            f"b {token_shape_str} (c kt kh kw) -> b v c (t kt) (h kh) (w kw)",
            h=pH,
            w=pW,
            kt=self.patch_temporal,
            kh=self.patch_spatial,
            kw=self.patch_spatial,
        )
        return x  # [B, V, C, T, H, W]

    @NVTXRangeDecorator("CosmosCausalDiT::forward")
    def forward(
        self,
        x_B_Ellipsis_D: torch.Tensor,
        timesteps_B: torch.Tensor,
        rope_emb: torch.Tensor,  # [L, 1, 1, D]
        network_cache: CosmosCausalDiTNetworkCache,
        condition_video_input_mask_B_Ellipsis_D: torch.Tensor,
        current_start: int = 0,
        hdmap_condition_B_Ellipsis_D: torch.Tensor | None = None,
        view_indices_B_V: torch.Tensor | None = None,
        eager_mode: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass dispatching to training or inference mode.

        Args:
            x_B_Ellipsis_D: Input video tensor [B, <TOKEN_MEMORY_LAYOUT>, D] after patchify
            condition_video_input_mask_B_Ellipsis_D: Condition video input mask [B, <TOKEN_MEMORY_LAYOUT>, D] after patchify
            timesteps_B: Timesteps [B]
            network_cache: KV cache for inference.
            current_start: Token range for inference (only used in eager mode)
            local_attn_token_size: Local attention window size (only used in eager mode)
            sink_token_size: Number of sink tokens (only used in eager mode)
            hdmap_condition_B_Ellipsis_D: HDMap tensor [B, <TOKEN_MEMORY_LAYOUT>, D]
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() after loading the checkpoint"
        )

        assert timesteps_B.ndim == 1
        timesteps_B = timesteps_B * self.timestep_scale

        # Memory layout: [B, V, T, HW, D] -> L = T * HW
        # Memory layout: [B, V, HW, T, D] -> L = HW * T
        self_attn_seq_len = x_B_Ellipsis_D.shape[-2] * x_B_Ellipsis_D.shape[-3]

        # Patch embedding
        x_B_Ellipsis_D = torch.cat([x_B_Ellipsis_D, condition_video_input_mask_B_Ellipsis_D], dim=-1)
        x_B_Ellipsis_D = self.x_embedder(x_B_Ellipsis_D)

        if self.additional_concat_ch > 0:
            assert hdmap_condition_B_Ellipsis_D is not None, (
                "hdmap is expected to be provided for additional concat channels"
            )
            additional_x_B_Ellipsis_D = self.additional_patch_embedding(hdmap_condition_B_Ellipsis_D)
            x_B_Ellipsis_D = x_B_Ellipsis_D + additional_x_B_Ellipsis_D

        # Time embedding
        t_emb_B_D, adaln_lora_B_3D = self.t_embedder(timesteps_B)
        t_emb_B_D = self.t_embedding_norm(t_emb_B_D)

        # AdaLN view modulation if enabled
        if view_indices_B_V is not None:
            assert self.adaln_view_embedder is not None and self.adaln_view_proj is not None, (
                "adaln_view_embedder and adaln_view_proj must be provided if view_indices_B_V is provided"
            )
            view_emb = self.adaln_view_embedder(view_indices_B_V)  # [B, V, D]
            view_embedding_proj_B_V_9D = self.adaln_view_proj(view_emb)  # [B, V, 9D]
        else:
            view_embedding_proj_B_V_9D = None

        # Note: If not in eager mode, we should call `before_update` and `after_update` MANUALLY outside the network.
        if eager_mode:
            network_cache.before_update(self_attn_seq_start=current_start, self_attn_seq_len=self_attn_seq_len)
        for block_idx, block in enumerate(self.blocks):
            x_B_Ellipsis_D = block(
                x_B_Ellipsis_D=x_B_Ellipsis_D,
                emb_B_D=t_emb_B_D,
                rope_emb=rope_emb,
                adaln_lora_B_3D=adaln_lora_B_3D,
                block_kv_cache=network_cache[block_idx],
                view_embedding_proj_B_V_9D=view_embedding_proj_B_V_9D,
            )
        if eager_mode:
            network_cache.after_update(self_attn_seq_start=current_start, self_attn_seq_len=self_attn_seq_len)

        # Final layer
        x_B_Ellipsis_D = self.final_layer(x_B_Ellipsis_D, t_emb_B_D, adaln_lora_B_3D)
        return x_B_Ellipsis_D

    def set_context_parallel_group(
        self, self_attn_group: ProcessGroup | None, cross_view_attn_group: ProcessGroup | None = None
    ) -> None:
        """Enable hierarchical CP with separate view and temporal groups.

        Hierarchical CP splits along view dimension (V) first, then temporal dimension (T).
        - self_attn_group: Ranks processing same view (for T gathering in self-attention)
        - cross_view_attn_group: Ranks at same T slice (for V gathering in cross-view attention)

        Args:
            self_attn_group: ProcessGroup for ranks with same view ID
            cross_view_attn_group: ProcessGroup for ranks at same temporal slice
        """
        for block in self.blocks:
            block.set_context_parallel_group(self_attn_group, cross_view_attn_group)


class CosmosCausalDiTCUDAGraphWrapper:
    """
    CUDA Graph wrapper for CosmosCausalDiT.

    This wrapper handles the complexity of using CUDA graphs with the network by:
    1. Managing the transition between eager mode (warmup) and graph mode (steady-state)
    2. Converting KV cache to fixed-size KVCacheSteadyState for graph capture
    3. Providing methods for graph capture and replay

    CUDA graphs can only be used when:
        current_start >= sink_token_size + local_attn_token_size
    At this point, the KV cache size is fixed and operations become deterministic.
    """

    def __init__(
        self,
        network: CosmosCausalDiT,
        sink_token_size: int,
        local_attn_token_size: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.network = network
        self.sink_token_size = sink_token_size
        self.local_attn_token_size = local_attn_token_size
        self.device = device
        self.dtype = dtype
        self.reset()

    def refresh_cache(self, context: torch.Tensor) -> None:
        """Inplace update the cache with the new context."""
        assert self._steady_network_cache is not None, "steady network cache is not initialized"
        self.network.refresh_cache(self._steady_network_cache, context)

    def can_use_graph(self, current_start: int) -> bool:
        """Check if CUDA graph can be used at the given position."""
        if self.local_attn_token_size <= 0:
            return False
        return current_start >= self.sink_token_size + self.local_attn_token_size

    def capture_graph(
        self,
        x_B_Ellipsis_D: Tensor,
        timesteps_B: Tensor,
        rope_emb: Tensor,
        network_cache: CosmosCausalDiTNetworkCache,
        condition_video_input_mask_B_Ellipsis_D: Tensor,
        current_start: int,
        hdmap_condition_B_Ellipsis_D: Tensor | None = None,
        view_indices_B_V: Tensor | None = None,
        warmup_iters: int = 3,
    ) -> None:
        """
        Capture the CUDA graph for steady-state inference.

        This should be called when current_start >= sink_token_size + local_attn_token_size.

        Args:
            x_B_Ellipsis_D: Input tensor
            timesteps_B: Timesteps
            rope_emb: RoPE embeddings
            steady_state_cache: Steady-state network cache (from prepare_steady_state_cache)
            condition_video_input_mask_B_Ellipsis_D: Condition mask
            current_start: Current token position
            hdmap_condition_B_Ellipsis_D: Optional hdmap condition
            view_indices_B_V: Optional view indices
            warmup_iters: Number of warmup iterations before capture
        """
        assert self.can_use_graph(current_start), (
            f"Cannot capture graph at current_start={current_start}. "
            f"Need current_start >= {self.sink_token_size + self.local_attn_token_size}"
        )

        # Calculate input size from tensor shape
        # For V_T_HW layout: [B, V, T, HW, D] -> T * HW per view
        # For V_HW_T layout: [B, V, HW, T, D] -> HW * T per view
        self._input_size = x_B_Ellipsis_D.shape[2] * x_B_Ellipsis_D.shape[3]

        # Convert network cache to steady-state
        self._steady_network_cache = network_cache

        # Allocate static input buffers
        self._static_x = x_B_Ellipsis_D.clone()
        self._static_timesteps = timesteps_B.clone()
        self._static_rope_emb = rope_emb.clone()
        self._static_condition_mask = condition_video_input_mask_B_Ellipsis_D.clone()
        if hdmap_condition_B_Ellipsis_D is not None:
            self._static_hdmap = hdmap_condition_B_Ellipsis_D.clone()
        if view_indices_B_V is not None:
            self._static_view_indices = view_indices_B_V.clone()

        # Manually call `before_update` outside the network, since we disable
        # eager mode for the network forward pass.
        self._steady_network_cache.before_update(
            self_attn_seq_start=current_start,
            self_attn_seq_len=self._input_size,
        )

        # Warmup iterations
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                self._static_output = self.network(
                    x_B_Ellipsis_D=self._static_x,
                    timesteps_B=self._static_timesteps,
                    rope_emb=self._static_rope_emb,
                    network_cache=self._steady_network_cache,
                    condition_video_input_mask_B_Ellipsis_D=self._static_condition_mask,
                    hdmap_condition_B_Ellipsis_D=self._static_hdmap,
                    view_indices_B_V=self._static_view_indices,
                    eager_mode=False,
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self.network(
                x_B_Ellipsis_D=self._static_x,
                timesteps_B=self._static_timesteps,
                rope_emb=self._static_rope_emb,
                network_cache=self._steady_network_cache,
                condition_video_input_mask_B_Ellipsis_D=self._static_condition_mask,
                hdmap_condition_B_Ellipsis_D=self._static_hdmap,
                view_indices_B_V=self._static_view_indices,
                eager_mode=False,
            )

        # Manually call `after_update` outside the network, since we disable
        # eager mode for the network forward pass.
        self._steady_network_cache.after_update(
            self_attn_seq_start=current_start,
            self_attn_seq_len=self._input_size,
        )
        self._graph_captured = True

    def forward_graph(
        self,
        x_B_Ellipsis_D: Tensor,
        timesteps_B: Tensor,
        rope_emb: Tensor,
        hdmap_condition_B_Ellipsis_D: Tensor | None = None,
        current_start: int = 0,
    ) -> Tensor:
        """
        Run forward pass using captured CUDA graph.

        Args:
            x_B_Ellipsis_D: Input tensor (will be copied to static buffer)
            timesteps_B: Timesteps (will be copied to static buffer)
            rope_emb: RoPE embeddings (will be copied to static buffer)
            hdmap_condition_B_Ellipsis_D: Optional hdmap condition (will be copied to static buffer)
            current_start: Current token position

        Returns:
            Output tensor (clone of static output buffer)
        """
        assert self._graph_captured, "Graph not captured. Call capture_graph first."
        assert self._static_x is not None
        assert self._static_timesteps is not None
        assert self._static_output is not None
        assert self._steady_network_cache is not None

        # Copy inputs to static buffers
        self._static_x.copy_(x_B_Ellipsis_D)
        self._static_timesteps.copy_(timesteps_B)
        self._static_rope_emb.copy_(rope_emb)
        if hdmap_condition_B_Ellipsis_D is not None:
            self._static_hdmap.copy_(hdmap_condition_B_Ellipsis_D)

        # Replay graph - always writes new k/v to the last positions.
        # Need to manually call `before_update` and `after_update` outside the graph.
        self._steady_network_cache.before_update(
            self_attn_seq_start=current_start,
            self_attn_seq_len=self._input_size,
        )
        self._graph.replay()
        self._steady_network_cache.after_update(
            self_attn_seq_start=current_start,
            self_attn_seq_len=self._input_size,
        )

        return self._static_output.clone()

    def forward(
        self,
        x_B_Ellipsis_D: torch.Tensor,
        timesteps_B: torch.Tensor,
        rope_emb: torch.Tensor,  # [L, 1, 1, D]
        network_cache: CosmosCausalDiTNetworkCache,
        condition_video_input_mask_B_Ellipsis_D: torch.Tensor,
        current_start: int = 0,
        hdmap_condition_B_Ellipsis_D: torch.Tensor | None = None,
        view_indices_B_V: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run forward pass in eager mode (no CUDA graph).

        This should be used during warmup when current_start < sink_token_size + local_attn_token_size.
        """

        if self.can_use_graph(current_start):
            if not self.is_graph_captured:
                # capture graph
                self.capture_graph(
                    x_B_Ellipsis_D=x_B_Ellipsis_D,
                    timesteps_B=timesteps_B,
                    rope_emb=rope_emb,
                    network_cache=network_cache,
                    condition_video_input_mask_B_Ellipsis_D=condition_video_input_mask_B_Ellipsis_D,
                    current_start=current_start,
                    hdmap_condition_B_Ellipsis_D=hdmap_condition_B_Ellipsis_D,
                    view_indices_B_V=view_indices_B_V,
                )

            # forward graph
            return self.forward_graph(
                x_B_Ellipsis_D=x_B_Ellipsis_D,
                timesteps_B=timesteps_B,
                rope_emb=rope_emb,
                hdmap_condition_B_Ellipsis_D=hdmap_condition_B_Ellipsis_D,
                current_start=current_start,
            )

        else:
            # eager mode
            return self.network(
                x_B_Ellipsis_D=x_B_Ellipsis_D,
                timesteps_B=timesteps_B,
                rope_emb=rope_emb,
                network_cache=network_cache,
                condition_video_input_mask_B_Ellipsis_D=condition_video_input_mask_B_Ellipsis_D,
                current_start=current_start,
                hdmap_condition_B_Ellipsis_D=hdmap_condition_B_Ellipsis_D,
                view_indices_B_V=view_indices_B_V,
            )

    @property
    def is_graph_captured(self) -> bool:
        return self._graph_captured

    def reset(self) -> None:
        """Reset the wrapper state."""
        self._graph = None
        self._graph_captured = False
        self._static_x = None
        self._static_timesteps = None
        self._static_rope_emb = None
        self._static_condition_mask = None
        self._static_hdmap = None
        self._static_view_indices = None
        self._static_output = None
        self._steady_network_cache = None
        self._input_size = 0


def prepare_network(
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    enable_hdmap_condition: bool = True,
    encode_with_pixel_shuffle: bool = False,
    enable_cross_view_attn: bool = False,
    cp_group_self_attn: ProcessGroup | None = None,
    cp_group_cross_view_attn: ProcessGroup | None = None,
    load_checkpoint_fn: Callable[[nn.Module], None] = lambda net: None,
    enable_torch_compile: bool = False,
    light_network: bool = False,
) -> CosmosCausalDiT:
    if enable_hdmap_condition:
        additional_concat_ch = 192 if encode_with_pixel_shuffle else 16
    else:
        additional_concat_ch = 0
    net = CosmosCausalDiT(
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=28 if not light_network else 1,
        num_heads=16,
        concat_padding_mask=True,
        use_adaln_lora=True,
        adaln_lora_dim=256,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=100352,
        crossattn_emb_channels=1024,
        timestep_scale=0.001,
        # support hdmap conditioning
        additional_concat_ch=additional_concat_ch,
        # support multi-view
        enable_cross_view_attn=enable_cross_view_attn,
    ).to(device=device, dtype=dtype)
    net.eval()
    net.set_context_parallel_group(cp_group_self_attn, cp_group_cross_view_attn)
    if load_checkpoint_fn is not None:
        load_checkpoint_fn(net)
    net.update_parameters_after_loading_checkpoint()
    if enable_torch_compile:
        net = torch.compile(net, mode="max-autotune-no-cudagraphs")
    return net


def run_network_denoising(
    denoising_timestamps: List[Tensor],  # list of tensors with shape [B,]
    denoising_sigmas: List[Tensor],  # list of tensors with shape [B,]
    network: CosmosCausalDiT | CosmosCausalDiTCUDAGraphWrapper,
    shape_B_Ellipsis_D: List[int],  # the shape of the input tensor x_B_Ellipsis_D (after CP)
    device: torch.device,
    dtype: torch.dtype,
    clean_latent_B_Ellipsis_D: Tensor | None = None,
    image_latent_B_Ellipsis_D: Tensor | None = None,
    mask_B_Ellipsis_1: Tensor | None = None,  # 1 means where to apply the image latent
    rng: torch.Generator | None = None,
    **network_kwargs,
):
    """
    Run multi-step denoising for a single block.
    """
    is_updating_kv_cache = clean_latent_B_Ellipsis_D is not None

    assert len(denoising_timestamps) == len(denoising_sigmas), (
        f"({len(denoising_timestamps)=}) must be equal to ({len(denoising_sigmas)=})"
    )
    num_steps = len(denoising_timestamps)
    if is_updating_kv_cache:
        assert num_steps == 1, "when updating kv cache, there should be only one step"

    def maybe_inject_image_latent_inplace(x: Tensor) -> None:
        if image_latent_B_Ellipsis_D is not None:
            assert mask_B_Ellipsis_1 is not None
            x.mul_(1.0 - mask_B_Ellipsis_1).add_(image_latent_B_Ellipsis_D * mask_B_Ellipsis_1)

    for step in range(num_steps):
        timesteps_B = denoising_timestamps[step]
        sigmas_B = denoising_sigmas[step]
        sigma_B_Ellipsis_1 = sigmas_B.view(-1, *([1] * (len(shape_B_Ellipsis_D) - 1)))

        # create noisy input
        noise = torch.randn(shape_B_Ellipsis_D, device=device, dtype=dtype, generator=rng)
        if clean_latent_B_Ellipsis_D is None:
            # first step is pure noise
            x_B_Ellipsis_D = noise
        else:
            # subsequent steps are noisy latent: add noise to the clean latent
            x_B_Ellipsis_D = (1 - sigma_B_Ellipsis_1) * clean_latent_B_Ellipsis_D + sigma_B_Ellipsis_1 * noise

        # inject back the conditional image latent.
        maybe_inject_image_latent_inplace(x_B_Ellipsis_D)

        # flow prediction
        flow_pred_B_Ellipsis_D = network.forward(
            x_B_Ellipsis_D=x_B_Ellipsis_D,
            timesteps_B=timesteps_B,
            **network_kwargs,
        )

        # if updating kv cache, we can return immediately after network forward pass.
        if is_updating_kv_cache:
            return clean_latent_B_Ellipsis_D

        # recover clean latent
        clean_latent_B_Ellipsis_D = x_B_Ellipsis_D - sigma_B_Ellipsis_1 * flow_pred_B_Ellipsis_D

    # inject back the conditional image latent.
    maybe_inject_image_latent_inplace(clean_latent_B_Ellipsis_D)

    return clean_latent_B_Ellipsis_D
