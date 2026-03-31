#!/usr/bin/env python3
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
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""

import os
from collections import namedtuple
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm.auto import tqdm

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

from cosmos_transfer2._src.av.bidirectional.tokenizers.wan2pt1 import WANVAECache


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks(model, x, parallel, show_progress_bar, cache_mem: List | None = None):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        # parallel over input timesteps, iterate over blocks
        for idx, b in enumerate(model):
            if isinstance(b, MemBlock):
                prev_mem = cache_mem[idx]
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                if prev_mem is None:
                    # roll to the right and left pad with zeros
                    curr_mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T]
                else:
                    # roll to the right and left pad with the last frame in previous mem
                    curr_mem = torch.cat([prev_mem[:, -1:], _x[:, :-1]], dim=1)
                x = b(x, curr_mem.reshape(x.shape))
                cache_mem[idx] = _x
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        # need to fix :(
        out = []
        # iterate over input timesteps and also iterate over blocks.
        # because of the cursed TPool/TGrow blocks, this is not a nested loop,
        # it's actually a ***graph traversal*** problem! so let's make a queue
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        # in addition to manually managing our queue, we also need to manually manage our progressbar.
        # we'll update it for every source node that we consume.
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        # we'll also need a separate addressable memory per node as well
        mem = [None] * len(model) if cache_mem is None else cache_mem
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                # new source node consumed
                progress_bar.update(1)
            if i == len(model):
                # reached end of the graph, append result to output list
                out.append(xt)
            else:
                # fetch the block to process
                b = model[i]
                if isinstance(b, MemBlock):
                    # mem blocks are simple since we're visiting the graph in causal order
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt)
                        # ^ inplace might reduce mysterious pytorch memory allocations? doesn't help though
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt_new, i + 1))
                elif isinstance(b, TPool):
                    # pool blocks are miserable
                    if mem[i] is None:
                        mem[i] = []  # pool memory is itself a queue of inputs to pool
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        # pool mem is in invalid state, we should have pooled before this
                        raise ValueError("???")
                    elif len(mem[i]) < b.stride:
                        # pool mem is not yet full, go back to processing the work queue
                        pass
                    else:
                        # pool mem is ready, run the pool block
                        N, C, H, W = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N * b.stride, C, H, W))
                        # reset the pool mem
                        mem[i] = []
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C, H, W = xt.shape
                    # each tgrow has multiple successor nodes
                    for xt_next in reversed(xt.view(N, b.stride * C, H, W).chunk(b.stride, 1)):
                        # add successor to work queue
                        work_queue.insert(0, TWorkItem(xt_next, i + 1))
                else:
                    # normal block with no funny business
                    xt = b(xt)
                    # add successor to work queue
                    work_queue.insert(0, TWorkItem(xt, i + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    def __init__(
        self,
        checkpoint_path="taehv.pth",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        patch_size=1,
        latent_channels=16,
        model_type="wan21",
    ):
        """Initialize pretrained TAEHV from the given checkpoint.

        Arg:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
        """
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        # if checkpoint_path is not None and "taew2_2" in checkpoint_path:
        #     self.patch_size, self.latent_channels = 2, 48
        self.model_type = model_type
        if model_type == "wan22":
            self.patch_size, self.latent_channels = 2, 48
        if model_type == "hy15":
            act_func = nn.LeakyReLU(0.2, inplace=True)
        else:
            act_func = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64),
            act_func,
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            act_func,
            conv(n_f[3], self.image_channels * self.patch_size**2),
        )
        if checkpoint_path is not None:
            ext = os.path.splitext(checkpoint_path)[1].lower()

            if ext == ".pth":
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            elif ext == ".safetensors":
                state_dict = load_file(checkpoint_path, device="cpu")
            else:
                raise ValueError(f"Unsupported checkpoint format: {ext}. Supported formats: .pth, .safetensors")

            self.load_state_dict(self.patch_tgrow_layers(state_dict))

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def encode_video(self, x, parallel=True, show_progress_bar=False, cache: Optional[WANVAECache] = None):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            # pad at end to multiple of 4
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(
            self.encoder, x, parallel, show_progress_bar, cache_mem=cache.enc_feat_map if cache is not None else None
        )

    def decode_video(self, x, parallel=True, show_progress_bar=False, cache: Optional[WANVAECache] = None):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=12) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        # NOTE(qi): We only trim the first decoded chunk. Index 3 corresponds to the first MemBlock in the decoder.
        #           We can also use other ways to determine if it's the first decode.
        first_decode = True if cache is None else (cache.dec_feat_map[3] is None)
        # Combine with the original logic to determine if we should skip trimming.
        skip_trim = (self.is_cogvideox and x.shape[1] % 2 == 0) or (not first_decode)
        x = apply_model_with_memblocks(
            self.decoder, x, parallel, show_progress_bar, cache_mem=cache.dec_feat_map if cache is not None else None
        )
        if self.model_type == "hy15":
            x = x.clamp_(-1, 1)
        else:
            x = x.clamp_(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        if skip_trim:
            # skip trimming for cogvideox to make frame counts match.
            # this still doesn't have correct temporal alignment for certain frame counts
            # (cogvideox seems to pad at the start?), but for multiple-of-4 it's fine.
            return x
        return x[:, self.frames_to_trim :]

    def prepare_cache(self) -> WANVAECache:
        return WANVAECache(
            enc_conv_idx=[],
            enc_feat_map=[None] * len(self.encoder),
            dec_conv_idx=[],
            dec_feat_map=[None] * len(self.decoder),
        )

    def clear_cache(self):
        pass  # do nothing
