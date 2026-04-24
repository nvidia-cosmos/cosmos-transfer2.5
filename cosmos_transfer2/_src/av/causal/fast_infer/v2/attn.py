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

from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.backends.cuda import sdp_kernel
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from transformer_engine.pytorch.attention import DotProductAttention

try:
    # Build flash_attn.cute from source:
    # cd flash-attention/flash_attn/cute
    # uv build --wheel . -v --no-build-isolation --out-dir wheels/
    # pip install wheels/*.whl
    from flash_attn.cute.interface import _flash_attn_fwd as flash_attn_func_v4
except ImportError:
    flash_attn_func_v4 = None

# Need to disable load balance for torchcontext parallel to work.
from torch.distributed.tensor.experimental._attention import _cp_options, set_rotate_method

from cosmos_transfer2._src.av.causal.fast_infer.v2.tests.templated_benchmark import (
    ContextParallelOptions,
    _templated_ring_attention,
    _templated_ulysses_attention,
    torch_cudnn_attention,
)

set_rotate_method("allgather")

_cp_options.enable_load_balance = False


class TransformerEngineSDPA(DotProductAttention):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__(
            num_heads,
            head_dim,
            num_gqa_groups=num_heads,
            attention_dropout=0,
            qkv_format="bshd",
            attn_mask_type="no_mask",
        )

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        if cp_group is None:
            super().set_context_parallel_group(None, None, torch.cuda.Stream(), "p2p")
        else:
            cp_global_ranks = get_process_group_ranks(cp_group)
            super().set_context_parallel_group(cp_group, cp_global_ranks, torch.cuda.Stream(), "p2p")

    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        return super().forward(q, k, v, **kwargs).view(q.shape)

    def enable_all_backends(self) -> None:
        pass


class PytorchSDPA(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        impl: Literal[
            "flash", "math", "mem_efficient", "cudnn", "cudnn_manual_ring", "cudnn_manual_ulysses"
        ] = "cudnn_manual_ring",
    ):
        super().__init__()
        assert impl in ["flash", "math", "mem_efficient", "cudnn", "cudnn_manual_ring", "cudnn_manual_ulysses"], (
            "Invalid implementation"
        )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.enable_flash = impl == "flash"
        self.enable_math = impl == "math"
        self.enable_mem_efficient = impl == "mem_efficient"
        self.enable_cudnn = impl == "cudnn"

        self.enable_cudnn_manual_ring = impl == "cudnn_manual_ring"
        self.enable_cudnn_manual_ulysses = impl == "cudnn_manual_ulysses"

        self.device_mesh = None

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        if cp_group is None:
            self.device_mesh = None
            return
        self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")
        # Switch to flash attention if torch version < 2.8 and cudnn is enabled (cudnn doesn't support DTensor in older versions)
        torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if torch_version < (2, 8) and self.enable_cudnn:
            self.enable_cudnn = False
            self.enable_flash = True

    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        # Input qkv are assume to have shape [B, S, H, D], shared tensor.
        assert q.shape[2] == k.shape[2] == v.shape[2] == self.num_heads

        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)  # [B, H, S, D]
        v = v.transpose(1, 2)  # [B, H, S, D]

        if self.enable_cudnn_manual_ulysses:
            o = self._cudnn_manual_ulysses_impl(q, k, v, **kwargs)

        elif self.enable_cudnn_manual_ring:
            o = self._cudnn_manual_ring_impl(q, k, v, **kwargs)

        else:
            o = self._torch_native_impl(q, k, v, **kwargs)

        return o.transpose(1, 2)  # [B, S, H, D], shared tensor.

    def _torch_native_impl(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        with sdp_kernel(
            enable_flash=self.enable_flash,
            enable_math=self.enable_math,
            enable_mem_efficient=self.enable_mem_efficient,
            enable_cudnn=self.enable_cudnn,
        ):
            if self.device_mesh is not None:
                # # NOTE(ruilong): implementation 1.
                # q_dtensor = DTensor.from_local(q.contiguous(), self.device_mesh, [Shard(2)])
                # k_dtensor = DTensor.from_local(k.contiguous(), self.device_mesh, [Shard(2)])
                # v_dtensor = DTensor.from_local(v.contiguous(), self.device_mesh, [Shard(2)])
                # with _enable_cp_dispatcher():
                #     o_dtensor = F.scaled_dot_product_attention(q_dtensor, k_dtensor, v_dtensor, **kwargs)
                # o = o_dtensor.to_local().contiguous()

                # NOTE(ruilong): implementation 2.
                # Pass a dummy buffer to satisfy context_parallel's buffers[0].device
                # check (required in PyTorch 2.9+ where buffers cannot be empty).
                _dummy = torch.empty(self.device_mesh.size(), device=q.device)
                with context_parallel(
                    self.device_mesh,
                    buffers=(_dummy,),
                    buffer_seq_dims=(0,),
                    no_restore_buffers={_dummy},
                ):
                    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
            else:
                o = F.scaled_dot_product_attention(q, k, v, **kwargs)

        return o

    def _cudnn_manual_ring_impl(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        if self.device_mesh is None:
            return torch_cudnn_attention(q, k, v, return_lse=False)
        else:
            cp_options = ContextParallelOptions(
                mode="ring",
                ring_mesh=self.device_mesh,
                convert_to_fp32=True,
                op=torch_cudnn_attention,
                return_lse=False,
            )
            return _templated_ring_attention(q, k, v, cp_options)

    def _cudnn_manual_ulysses_impl(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        if self.device_mesh is None:
            return torch_cudnn_attention(q, k, v, return_lse=False)
        else:
            cp_options = ContextParallelOptions(
                mode="ulysses",
                ulysses_mesh=self.device_mesh,
                op=torch_cudnn_attention,
                return_lse=False,
            )
            return _templated_ulysses_attention(q, k, v, cp_options)

    def enable_all_backends(self) -> None:
        self.enable_flash = True
        self.enable_math = True
        self.enable_mem_efficient = True
        self.enable_cudnn = True


class FlashAttention4(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        version: Literal[2, 3, 4] = 4,
    ):
        super().__init__()
        assert version == 4, "Only version 4 is supported"
        assert flash_attn_func_v4 is not None, "FA4 is not installed"

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.device_mesh = None

    def set_context_parallel_group(self, cp_group: ProcessGroup) -> None:
        raise NotImplementedError("FlashAttention4 does not support context parallel")
        self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")

    @torch.compiler.disable
    def forward(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        # Input qkv are assume to have shape [B, S, H, D], shared tensor.
        assert q.shape[2] == k.shape[2] == v.shape[2] == self.num_heads
        return flash_attn_func_v4(q, k, v)[0]


def create_attn_op(
    num_heads: int,
    head_dim: int,
    choice: Literal["TE", "Torch", "FA4"] = "TE",
    **kwargs,
) -> Callable:
    """Create an attention operator."""
    if choice == "TE":
        return TransformerEngineSDPA(num_heads, head_dim, **kwargs)
    elif choice == "Torch":
        return PytorchSDPA(num_heads, head_dim, **kwargs)
    elif choice == "FA4":
        return FlashAttention4(num_heads, head_dim, **kwargs)
    else:
        raise ValueError(f"Invalid attn choice: {choice}")


def test_attn_op(world_size: int, rank: int):
    assert torch.cuda.is_available()
    assert dist.is_nccl_available()
    torch.cuda.set_device(f"cuda:{rank}")
    torch.cuda.manual_seed(0)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("cp",))
    rank = torch.distributed.get_rank()
    cp_group = device_mesh.get_group(mesh_dim=0)

    batch = 1
    nheads = 40
    qkv_len = 4680  # full sequence length
    dim = 128
    dtype = torch.bfloat16

    attn_op1 = create_attn_op(nheads, dim, choice="TE")
    attn_op2 = create_attn_op(nheads, dim, choice="Torch", impl="cudnn")
    attn_op3 = create_attn_op(nheads, dim, choice="Torch", impl="cudnn_manual_ring")
    attn_op4 = create_attn_op(nheads, dim, choice="Torch", impl="cudnn_manual_ulysses")
    if world_size > 1:
        attn_op1.set_context_parallel_group(cp_group)
        attn_op2.set_context_parallel_group(cp_group)
        attn_op3.set_context_parallel_group(cp_group)
        attn_op4.set_context_parallel_group(cp_group)

    q = torch.randn((batch, qkv_len, nheads, dim), dtype=dtype, device="cuda")
    k = torch.randn((batch, qkv_len, nheads, dim), dtype=dtype, device="cuda")
    v = torch.randn((batch, qkv_len, nheads, dim), dtype=dtype, device="cuda")
    if world_size > 1:
        q = distribute_tensor(q, device_mesh, [Shard(1)], src_data_rank=None).to_local().contiguous().clone()
        k = distribute_tensor(k, device_mesh, [Shard(1)], src_data_rank=None).to_local().contiguous().clone()
        v = distribute_tensor(v, device_mesh, [Shard(1)], src_data_rank=None).to_local().contiguous().clone()
        assert q.shape == (batch, qkv_len // world_size, nheads, dim)
        assert k.shape == (batch, qkv_len // world_size, nheads, dim)
        assert v.shape == (batch, qkv_len // world_size, nheads, dim)

    o1 = attn_op1(q, k, v)
    o2 = attn_op2(q, k, v)
    o3 = attn_op3(q, k, v)
    o4 = attn_op4(q, k, v)
    torch.testing.assert_close(o1, o2, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(o2, o3, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(o2, o4, atol=1e-3, rtol=1e-3)
    dist.destroy_process_group()


# PYTHONPATH=. torchrun --nproc_per_node=1 cosmos_transfer2/_src/av/causal/fast_infer/v2/attn.py
if __name__ == "__main__":
    import os

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    test_attn_op(world_size, rank)
