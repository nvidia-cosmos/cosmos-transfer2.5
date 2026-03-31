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
import torch.nn as nn

from cosmos_transfer2._src.av.bidirectional.tokenizers.wan2pt1 import WANVAECache

from .tae import TAEHV


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class WanVAE_tiny(nn.Module):
    def __init__(self, vae_path="taew2_1.pth", dtype=torch.bfloat16, device="cuda", need_scaled=False):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_path).to(self.dtype)
        self.temperal_downsample = [True, True, False]
        self.need_scaled = need_scaled

        if self.need_scaled:
            self.latents_mean = [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]

            self.latents_std = [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ]

            self.z_dim = 16

    @torch.no_grad()
    def decode(self, latents, cache: Optional[WANVAECache] = None):
        if self.need_scaled:
            latents_mean = (
                torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

        # low-memory, set parallel=True for faster + higher memory
        return (
            self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel=True, cache=cache)
            .transpose(1, 2)
            .mul_(2)
            .sub_(1)
        )

    @torch.no_grad()
    def encode_video(self, vid, cache: Optional[WANVAECache] = None):
        return self.taehv.encode_video(vid, cache=cache)

    @torch.no_grad()
    def decode_video(self, vid_enc, cache: Optional[WANVAECache] = None):
        return self.taehv.decode_video(vid_enc, cache=cache)

    def prepare_cache(self) -> WANVAECache:
        return self.taehv.prepare_cache()

    def clear_cache(self):
        self.taehv.clear_cache()
