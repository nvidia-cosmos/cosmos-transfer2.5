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

from cosmos_transfer2._src.av.bidirectional.tokenizers.interface import VideoTokenizerInterface
from cosmos_transfer2._src.av.bidirectional.tokenizers.wan2pt1 import WANVAECache

from .vae import WanVAE
from .vae_tiny import WanVAE_tiny


def load_vae_encoder(
    device: torch.device,
    dtype: torch.dtype,
    use_lightvae: bool,
    vae_path: Optional[str] = None,
    parallel_decode: bool = False,
):
    vae_config = {
        "vae_path": vae_path,
        "device": device,
        "parallel": False,
        "use_tiling": False,
        "cpu_offload": False,
        "dtype": dtype,
        "load_from_rank0": False,
        "use_lightvae": use_lightvae,
    }
    if use_lightvae:
        vae_config["parallel_decode"] = parallel_decode
    return WanVAE(**vae_config)


def load_vae_decoder(
    device: torch.device,
    dtype: torch.dtype,
    vae_path: Optional[str] = None,
    tae_path: Optional[str] = None,
    parallel: bool = False,
    process_group=None,
    parallel_decode: bool = True,
):
    use_tae = tae_path is not None
    need_scaled = use_tae and "lighttae" in tae_path
    vae_config = {
        "vae_path": vae_path,
        "device": device,
        "parallel": parallel,
        "process_group": process_group,
        "use_tiling": False,
        "cpu_offload": False,
        "use_lightvae": True,
        "dtype": dtype,
        "load_from_rank0": False,
    }
    if use_tae:
        vae_decoder = WanVAE_tiny(vae_path=tae_path, device=device, need_scaled=need_scaled).to(device)
    else:
        vae_config["parallel_decode"] = parallel_decode
        vae_decoder = WanVAE(**vae_config)
    return vae_decoder


def load_vae(
    device: torch.device,
    dtype: torch.dtype,
    use_lightvae: bool,
    use_tae: bool,
    vae_path: str,
    tae_path: Optional[str] = None,
    parallel: bool = False,
    process_group=None,
    parallel_decode: bool = True,
):
    vae_encoder = load_vae_encoder(device, dtype, use_lightvae, vae_path, parallel_decode)
    if use_tae:
        assert tae_path is not None, "tae_path is required when use_tae"
        vae_decoder = load_vae_decoder(
            device,
            dtype,
            vae_path=vae_path,
            tae_path=tae_path,
            parallel=parallel,
            process_group=process_group,
        )
    elif parallel:
        vae_decoder = load_vae_decoder(
            device,
            dtype,
            vae_path=vae_path,
            tae_path=None,
            parallel=parallel,
            process_group=process_group,
            parallel_decode=parallel_decode,
        )
    else:
        vae_decoder = vae_encoder
    return vae_encoder, vae_decoder


def load_image_and_video_mean(device: torch.device, dtype: torch.dtype, max_video_length: int = 960):
    return (
        torch.zeros(1, 1, 1, 1, 1, device=device, dtype=dtype),
        torch.ones(1, 1, 1, 1, 1, device=device, dtype=dtype),
        torch.zeros(1, 1, max_video_length, 1, 1, device=device, dtype=dtype),
        torch.ones(1, 1, max_video_length, 1, 1, device=device, dtype=dtype),
    )


##/mnt/scratch/alpasim/outputs/policy_test_5b1d_r1/rollouts/clipgt-5b1d592f-5745-4c8e-9ffd-f8ae2ecbe0c5/b41edb5e-1cb6-11f1-9005-fdd70da56493/videos/


class LightWan2pt1TokenizerInterface(VideoTokenizerInterface):
    def __init__(
        self,
        vae_path: str,
        tae_path: Optional[str] = None,
        use_lightvae: bool = True,
        chunk_duration: int = 81,
        max_video_length: int = 500,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        parallel: bool = False,
        process_group=None,
        parallel_decode: bool = True,
        **kwargs,
    ):
        self._dtype = dtype
        self._device = device
        use_tae = tae_path is not None
        self.model_encoder, self.model_decoder = load_vae(
            device=self._device,
            dtype=self._dtype,
            vae_path=vae_path,
            use_lightvae=use_lightvae,
            tae_path=tae_path,
            use_tae=use_tae,
            parallel=parallel,
            process_group=process_group,
            parallel_decode=parallel_decode,
        )
        self.chunk_duration = chunk_duration
        self.img_mean, self.img_std, self.video_mean, self.video_std = load_image_and_video_mean(
            self._device,
            self._dtype,
            max_video_length=max_video_length,
        )

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def reset_dtype(self):
        pass

    def clear_cache(self):
        self.model_encoder.clear_cache()
        self.model_decoder.clear_cache()

    def prepare_cache(self) -> WANVAECache:
        encoder_cache = self.model_encoder.prepare_cache()
        decoder_cache = self.model_decoder.prepare_cache()
        return WANVAECache(
            enc_conv_idx=encoder_cache.enc_conv_idx,
            enc_feat_map=encoder_cache.enc_feat_map,
            dec_conv_idx=decoder_cache.dec_conv_idx,
            dec_feat_map=decoder_cache.dec_feat_map,
        )

    def encode(self, state: torch.Tensor, cache: Optional[WANVAECache] = None) -> torch.Tensor:
        assert state.dtype == self._dtype, "state dtype must match dtype"
        latents = self.model_encoder.encode(state, cache=cache)
        assert latents.ndim == 5, "latents must be of 5D [V, C, T, H, W]"
        num_frames = latents.shape[2]
        if num_frames == 1:
            return (latents - self.img_mean) / self.img_std
        else:
            return (latents - self.video_mean[:, :, :num_frames]) / self.video_std[:, :, :num_frames]

    def decode(self, latent: torch.Tensor, cache: Optional[WANVAECache] = None) -> torch.Tensor:
        assert latent.dtype == self._dtype, "latent dtype must match dtype"
        num_frames = latent.shape[2]
        if num_frames == 1:
            unnormalized_latent = (latent * self.img_std) + self.img_mean
        else:
            unnormalized_latent = (latent * self.video_std[:, :, :num_frames]) + self.video_mean[:, :, :num_frames]
        assert latent.ndim == 5, "latent must be of 5D [V, C, T, H, W]"
        return self.model_decoder.decode(unnormalized_latent, cache=cache)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 4

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return 16

    @property
    def spatial_resolution(self):
        return 512

    def to(self, device: torch.device | str):
        self._device = device
        # move model to device
        self.model_encoder.to_cuda(self._device)
        if isinstance(self.model_decoder, WanVAE_tiny):
            self.model_decoder.taehv.to(self._device)
        else:
            self.model_decoder.to_cuda(self._device)

    @property
    def name(self):
        return "light_wan2pt1_tokenizer"
