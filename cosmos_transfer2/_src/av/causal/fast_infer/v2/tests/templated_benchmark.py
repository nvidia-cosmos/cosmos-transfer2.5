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

# From: https://gist.github.com/stas00/16abbeecfdd1877b70c2b7e750c030e3

import argparse
from dataclasses import dataclass
from typing import Callable, List, Literal

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed import DeviceMesh


@dataclass
class ContextParallelOptions:
    mode: Literal["ring", "ulysses", "unified"] = "ring"
    ring_mesh: DeviceMesh | None = None
    ulysses_mesh: DeviceMesh | None = None
    convert_to_fp32: bool = True
    op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
    return_lse: bool = True


def _templated_ring_attention(
    query, key, value, cp_options: ContextParallelOptions
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    rank = cp_options.ring_mesh.get_rank()
    world_size = cp_options.ring_mesh.size()

    if world_size == 1:
        return cp_options.op(query, key, value)

    next_rank = (rank + 1) % world_size
    prev_out = prev_lse = None

    kv_buffer = torch.cat([key.flatten(), value.flatten()]).contiguous()
    kv_buffer = funcol.all_gather_tensor(kv_buffer, gather_dim=0, group=cp_options.ring_mesh.get_group())
    kv_buffer = kv_buffer.chunk(world_size)

    for i in range(world_size):
        if i > 0:
            kv = kv_buffer[next_rank]
            key = kv[: key.numel()].reshape_as(key)
            value = kv[key.numel() :].reshape_as(value)
            next_rank = (next_rank + 1) % world_size

        out, lse = cp_options.op(query, key, value)

        if cp_options.convert_to_fp32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        if prev_out is not None:
            out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
            lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - lse)
        prev_out = out
        prev_lse = lse

    out = out.to(query.dtype)
    if cp_options.return_lse:
        return out, lse
    else:
        return out


def _templated_ulysses_attention(
    query, key, value, cp_options: ContextParallelOptions
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    world_size = cp_options.ulysses_mesh.size()
    group = cp_options.ulysses_mesh.get_group()

    if world_size == 1:
        return cp_options.op(query, key, value)

    B, H, Sq_LOCAL, D = query.shape
    B, H, Sk_LOCAL, D = key.shape
    H_LOCAL = H // world_size
    query = query.reshape(B, world_size, H_LOCAL, Sq_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    key = key.reshape(B, world_size, H_LOCAL, Sk_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    value = value.reshape(B, world_size, H_LOCAL, Sk_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()

    query, key, value = (funcol.all_to_all_single(x, None, None, group=group) for x in (query, key, value))
    query, key, value = (x.flatten(0, 1).permute(1, 2, 0, 3).contiguous() for x in (query, key, value))
    out, lse = cp_options.op(query, key, value)
    out = out.reshape(B, H_LOCAL, world_size, Sq_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    out = funcol.all_to_all_single(out, None, None, group=group).wait()
    out = out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
    if cp_options.return_lse:
        lse = lse.reshape(B, H_LOCAL, world_size, Sq_LOCAL).permute(2, 1, 0, 3).contiguous()
        lse = funcol.all_to_all_single(lse, None, None, group=group).wait()
        lse = lse.flatten(0, 1).permute(1, 0, 2).contiguous()
        return out, lse
    else:
        return out


def _templated_unified_attention(
    query, key, value, cp_options: ContextParallelOptions
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    ring_size = cp_options.ring_mesh.size()
    ulysses_size = cp_options.ulysses_mesh.size()
    ulysses_group = cp_options.ulysses_mesh.get_group()
    world_size = ring_size * ulysses_size

    if world_size == 1:
        return cp_options.op(query, key, value)

    B, H, Sq_LOCAL, D = query.shape
    B, H, Sk_LOCAL, D = key.shape
    H_LOCAL = H // ulysses_size
    query = query.reshape(B, ulysses_size, H_LOCAL, Sq_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    key = key.reshape(B, ulysses_size, H_LOCAL, Sk_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    value = value.reshape(B, ulysses_size, H_LOCAL, Sk_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()

    query, key, value = (funcol.all_to_all_single(x, None, None, group=ulysses_group) for x in (query, key, value))
    query, key, value = (x.flatten(0, 1).permute(1, 2, 0, 3).contiguous() for x in (query, key, value))
    if cp_options.return_lse:
        out, lse = _templated_ring_attention(query, key, value, cp_options)
        out = out.reshape(B, H_LOCAL, ulysses_size, Sq_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        out = funcol.all_to_all_single(out, None, None, group=ulysses_group).wait()
        out = out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        lse = lse.reshape(B, H_LOCAL, ulysses_size, Sq_LOCAL).permute(2, 1, 0, 3).contiguous()
        lse = funcol.all_to_all_single(lse, None, None, group=ulysses_group).wait()
        lse = lse.flatten(0, 1).permute(1, 0, 2).contiguous()
        return out, lse
    else:
        out = _templated_ring_attention(query, key, value, cp_options)
        out = out.reshape(B, H_LOCAL, ulysses_size, Sq_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        out = funcol.all_to_all_single(out, None, None, group=ulysses_group).wait()
        out = out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        return out


def torch_cudnn_attention(
    query, key, value, return_lse: bool = True
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=None,
            compute_log_sumexp=True,
        )
    )
    if return_lse:
        return out, lse
    else:
        return out


def torch_flash_attention(
    query, key, value, return_lse: bool = True
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    out, lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask = (
        torch.ops.aten._scaled_dot_product_flash_attention(
            query=query,
            key=key,
            value=value,
        )
    )
    if return_lse:
        return out, lse
    else:
        return out


OPS = {
    "cudnn": torch_cudnn_attention,
    # "flash": torch_flash_attention,
}
WORLD_SIZE = -1
RANK = -1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[14080 * 2])
    parser.add_argument(
        "--ops",
        type=str,
        nargs="+",
        choices=list(OPS.keys()),
        default=list(OPS.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(
    ring_degree: int,
    ulysses_degree: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    seq_lens: List[int],
    ops: List[str],
):
    global WORLD_SIZE, RANK

    mesh_names = ["ring", "ulysses"]
    mesh_dims = [ring_degree, ulysses_degree]
    mesh = dist.device_mesh.init_device_mesh("cuda", mesh_dims, mesh_dim_names=mesh_names)
    cp_options = ContextParallelOptions()
    cp_options.ring_mesh = mesh["ring"]
    cp_options.ulysses_mesh = mesh["ulysses"]
    cp_options.convert_to_fp32 = True
    cp_attention = None
    num_warmups = 5
    num_repeats = 10
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if ring_degree > 1 and ulysses_degree > 1:
        cp_options.mode = "unified"
        cp_attention = _templated_unified_attention
    elif ulysses_degree > 1:
        cp_options.mode = "ulysses"
        cp_attention = _templated_ulysses_attention
    else:
        cp_options.mode = "ring"
        cp_attention = _templated_ring_attention

    results = {}
    for op_name in ops:
        op = OPS[op_name]
        cp_options.op = op
        results[op_name] = {}

        for seq_len in seq_lens:
            shape = (batch_size, num_heads, seq_len, head_dim)
            query = torch.randn(shape, device=device, dtype=dtype)
            key = torch.randn(shape, device=device, dtype=dtype)
            value = torch.randn(shape, device=device, dtype=dtype)

            dist.broadcast(query, src=0)
            dist.broadcast(key, src=0)
            dist.broadcast(value, src=0)
            dist.barrier()
            torch.cuda.synchronize()

            reference_out, reference_lse = torch_cudnn_attention(query, key, value)
            query, key, value = (x.chunk(WORLD_SIZE, dim=2)[RANK].contiguous() for x in (query, key, value))

            for _ in range(num_warmups):
                if WORLD_SIZE == 1:
                    out, lse = op(query, key, value)
                else:
                    out, lse = cp_attention(query, key, value, cp_options)
                out = funcol.all_gather_tensor(out, gather_dim=2, group=mesh._flatten().get_group())
                lse = funcol.all_gather_tensor(lse, gather_dim=2, group=mesh._flatten().get_group())
            torch.cuda.synchronize()

            diff = out - reference_out
            absdiff = torch.abs(diff)
            absmax = torch.max(absdiff)
            mae = torch.mean(absdiff)
            mse = torch.mean(diff * diff)
            if RANK == 0:
                print(f"op: {op_name}, seq_len: {seq_len}, absmax: {absmax:.5f}, mae: {mae:.5f}, mse: {mse:.5f}")

            # if not torch.allclose(out, reference_out, atol=1e-2, rtol=1e-2):
            #     raise ValueError(f"Output mismatch for op: {op_name}, seq_len: {seq_len}")
            # if not torch.allclose(lse, reference_lse, atol=1e-2, rtol=1e-2):
            #     raise ValueError(f"LSE mismatch for op: {op_name}, seq_len: {seq_len}")
            dist.barrier()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(num_repeats):
                if WORLD_SIZE == 1:
                    out, lse = op(query, key, value)
                else:
                    out, lse = cp_attention(query, key, value, cp_options)
            end_event.record()
            torch.cuda.synchronize()
            dist.barrier()
            elapsed_time = start_event.elapsed_time(end_event) / num_repeats
            results[op_name][seq_len] = elapsed_time

    if RANK == 0:
        print("Benchmark results:")
        for op_name, seq_times in results.items():
            print(f"\n\n===== op: {op_name} =====")
            for seq_len, time in seq_times.items():
                print(f"  {seq_len=}, {time:.5f} ms")


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)

    try:
        dist.init_process_group(backend="nccl")
        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        torch.cuda.set_device(RANK)

        if args.ring_degree * args.ulysses_degree != WORLD_SIZE:
            raise ValueError(
                f"ring_degree * ulysses_degree must equal world size, got {args.ring_degree} * {args.ulysses_degree} != {WORLD_SIZE}"
            )

        main(
            ring_degree=args.ring_degree,
            ulysses_degree=args.ulysses_degree,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            seq_lens=args.seq_lens,
            ops=args.ops,
        )
    finally:
        dist.destroy_process_group()
