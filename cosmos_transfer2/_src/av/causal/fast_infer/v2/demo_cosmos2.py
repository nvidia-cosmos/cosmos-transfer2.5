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
Demo script for Cosmos2 multiview video generation with hierarchical context parallelism.

This script implements hierarchical CP that splits along view dimension (V) first,
then temporal dimension (T) when world_size > n_cameras.

Usage:
    # V-only split (world_size = n_cameras)
    PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 \
        cosmos_transfer2/_src/av/causal/fast_infer/v2/demo_cosmos2.py \
        --reso 480p --n_cameras 4 --total_blocks 3 --context_parallel_size 4

    # V+T split (world_size = 2 × n_cameras)
    PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 --master_port=12345 \
        cosmos_transfer2/_src/av/causal/fast_infer/v2/demo_cosmos2.py \
        --reso 480p --n_cameras 4 --total_blocks 3 --context_parallel_size 8
"""

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.distributed import ProcessGroup

from cosmos_transfer2._src.av.bidirectional.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_transfer2._src.av.causal.fast_infer.utils.profile import NVTXRangeDecorator
from cosmos_transfer2._src.av.causal.fast_infer.v2.network_cosmos2 import (
    DEFAULT_4VIEWS_VIEW_INDICES,
    ContextParallelDim,
    CosmosCausalDiTCUDAGraphWrapper,
    CosmosCausalDiTNetworkCache,
    SinsoidalRotaryFrequency,
    run_network_denoising,
)
from cosmos_transfer2._src.av.causal.fast_infer.v2.text_encoder import (
    TextEncoder,
    get_reason1_embeddings,
)
from cosmos_transfer2._src.av.causal.fast_infer.v2.wan2pt1_tokenizer import WANVAECache

# # Block megatron import.
# # import megatron will trigger its JIT compilation, which is extremely slow and not needed.
# sys.modules["megatron"] = None
# sys.modules["megatron.core"] = None
from cosmos_transfer2._src.imaginaire.utils import log


class ProfileEvents:
    def __init__(self):
        # sequential events
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc_after_rotary = torch.cuda.Event(enable_timing=True)
        self.toc_after_encode = torch.cuda.Event(enable_timing=True)
        self.toc_after_denoise = torch.cuda.Event(enable_timing=True)
        self.toc_after_decode = torch.cuda.Event(enable_timing=True)
        self.toc_after_update_kv = torch.cuda.Event(enable_timing=True)
        # special events
        #  tic
        #   ╰───> after_generate ───> after_upsample ───> ready
        #                                                   ╰────> after_finalize
        self.tic_block = torch.cuda.Event(enable_timing=True)
        self.toc_block_after_upsample = torch.cuda.Event(enable_timing=True)

    def summary(self) -> dict[str, float]:
        return {
            "elapsed_time_rotary": self.tic.elapsed_time(self.toc_after_rotary),
            "elapsed_time_encode": self.toc_after_rotary.elapsed_time(self.toc_after_encode),
            "elapsed_time_denoise": self.toc_after_encode.elapsed_time(self.toc_after_denoise),
            "elapsed_time_decode": self.toc_after_denoise.elapsed_time(self.toc_after_decode),
            "elapsed_time_update_kv": self.toc_after_decode.elapsed_time(self.toc_after_update_kv),
            "time_to_decode": self.tic.elapsed_time(self.toc_after_decode),
            "time_to_update_kv": self.tic.elapsed_time(self.toc_after_update_kv),
            # --- #
            "time_to_upsample": self.tic_block.elapsed_time(
                self.toc_block_after_upsample
            ),  # Time to upsample the video
        }

    @staticmethod
    def create(repeats: int) -> list["ProfileEvents"]:
        return [ProfileEvents() for _ in range(repeats)]

    @staticmethod
    def finalize(events: list["ProfileEvents"], skip_first_n: int = 0) -> None:
        if skip_first_n > 0:
            events = events[skip_first_n:]

        n = len(events)

        ts = []
        for event in events:
            ts.append(event.summary())

        elapsed_time_rotary = sum(t["elapsed_time_rotary"] for t in ts)
        elapsed_time_encode = sum(t["elapsed_time_encode"] for t in ts)
        elapsed_time_denoise = sum(t["elapsed_time_denoise"] for t in ts)
        elapsed_time_decode = sum(t["elapsed_time_decode"] for t in ts)
        elapsed_time_update_kv = sum(t["elapsed_time_update_kv"] for t in ts)
        time_to_decode = sum(t["time_to_decode"] for t in ts)
        time_to_update_kv = sum(t["time_to_update_kv"] for t in ts)
        time_to_upsample = sum(t["time_to_upsample"] for t in ts)

        def perc1(t):
            return f"({t / time_to_decode * 100:06.3f}%)"

        log.info(f"Average Latency to Decode: {time_to_decode / n / 1000.0} seconds")
        log.info(f"   ├─{perc1(elapsed_time_rotary)} rotary frequency {elapsed_time_rotary / n:.4f} ms")
        log.info(f"   ├─{perc1(elapsed_time_encode)} VAE encode HD map {elapsed_time_encode / n:.4f} ms")
        log.info(f"   ├─{perc1(elapsed_time_denoise)} DiT denoise latent {elapsed_time_denoise / n:.4f} ms")
        log.info(f"   ╰─{perc1(elapsed_time_decode)} VAE decode {elapsed_time_decode / n:.4f} ms")
        log.info(f"Average Latency to Update KV: {time_to_update_kv / n / 1000.0} seconds")
        log.info(f"   ╰─update KV {elapsed_time_update_kv / n:.4f} ms")
        log.info(f"Average Latency to Upsample: {time_to_upsample / n / 1000.0} seconds")


def preprocess_input_image(image: np.ndarray, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Preprocess the image.

    Args:
        image: the image array with shape [V, H, W, 3] in range [0, 255]

    Returns:
        the preprocessed image tensor with shape [1, V, 3, H, W] in range [-1, 1]
    """
    assert image.ndim == 4, "image must have shape [V, H, W, 3]"
    image = torch.from_numpy(image).to(dtype=dtype, device=device) / 127.5 - 1.0  # range [-1, 1]
    image = rearrange(image, "v h w c -> 1 v c h w")
    return image


def preprocess_input_hdmap_video(
    hdmap_video: np.ndarray, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    """Preprocess the hdmap condition.

    Args:
        hdmap_video: the hdmap video array with shape [V, T, H, W, 3] in range [0, 255]

    Returns:
        the preprocessed hdmap condition tensor with shape [1, V, 3, T, H, W] in range [-1, 1]
    """
    assert hdmap_video.ndim == 5, "hdmap_video must have shape [V, T, H, W, 3]"
    hdmap_video = torch.from_numpy(hdmap_video).to(dtype=dtype, device=device) / 127.5 - 1.0  # range [-1, 1]
    hdmap_video = rearrange(hdmap_video, "v t h w c -> 1 v c t h w")
    return hdmap_video


def preprocess_input_text_prompt(
    text_prompt: str, reason1_ckpt_path: str, device: torch.device | str, text_encoder: TextEncoder | None = None
) -> torch.Tensor:
    """Preprocess the text prompt.

    Args:
        text_prompt: the text prompt string
        reason1_ckpt_path: the path to the reason1 checkpoint
        device: the device to map the text embeddings to
    Returns:
        the preprocessed text prompt tensor with shape [1, L, D]
    """
    if text_prompt == "construction":
        text_prompt = "Wide-angle urban street scene from a low, dashboard-level viewpoint. A straight two-lane road with a faded center line. Active road construction zone on the right side with orange and white striped barricades and construction drums lining the lane. Parked sedans and SUVs in neutral colors line the left curb. On the right, a white stucco mid-rise building with blue fabric awnings. On the left, a low commercial strip. Mature green trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday sunlight, natural colors, realistic materials, crisp shadows."
    elif text_prompt == "evening":
        text_prompt = "Wide-angle urban street scene from a low, dashboard-level viewpoint. A straight two-lane road covered in packed white snow and grey slush with visible tire tracks. Parked sedans and SUVs covered in a layer of snow line the curbs. On the right, a white stucco mid-rise building with snow-capped blue fabric awnings. On the left, a low commercial strip with dark trim. Leafless trees with branches heavy with snow punctuate both sides. Overcast grey sky with falling snowflakes. Soft, cold diffuse lighting, realistic materials, winter atmosphere."
    elif text_prompt == "blocker":
        text_prompt = "Wide-angle urban street scene from a low, dashboard-level viewpoint. A straight two-lane road with a faded center line and curbside parking. In the middle distance, a red and white striped rectangular road barrier blocks the center of the street. Parked sedans and SUVs in neutral colors line the curbs. On the right, a white stucco mid-rise building with blue fabric awnings. On the left, a low commercial strip. Mature green trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday sunlight, natural colors, realistic materials, crisp shadows, clean asphalt texture."
    elif text_prompt == "snow":
        text_prompt = "Wide-angle urban street scene from a low, dashboard-level viewpoint. A straight two-lane road covered in packed white snow and grey slush with visible tire tracks. Parked sedans and SUVs covered in a layer of snow line the curbs. On the right, a white stucco mid-rise building with snow-capped blue fabric awnings. On the left, a low commercial strip with dark trim. Leafless trees with branches heavy with snow punctuate both sides. Overcast grey sky with falling snowflakes. Soft, cold diffuse lighting, realistic materials, winter atmosphere."
    elif text_prompt == "night":
        text_prompt = "Wide-angle urban street scene from a low, dashboard-level viewpoint. Nighttime scene with heavy rain falling. A straight two-lane road with wet, slick black asphalt reflecting streetlights and car headlights. Parked sedans and SUVs line the curbs. On the right, a white stucco mid-rise building. On the left, a low commercial strip with glowing storefront signage. Streetlights cast a warm yellow glow against the darkness. Pitch black sky with visible rain streaks. High contrast shadows, cinematic lighting."

    text_prompt_cache_path = f"./data_local/text_prompt_{text_prompt[:20].replace(' ', '_')}_{text_prompt[-20:].replace(' ', '_')}_{len(text_prompt)}chars.pt"
    if not os.path.exists(text_prompt_cache_path):
        os.makedirs(os.path.dirname(text_prompt_cache_path), exist_ok=True)
        log.info(f"Computing text embeddings for {text_prompt} and saving to {text_prompt_cache_path}")
        text_embeddings = get_reason1_embeddings(
            text_prompt, reason1_ckpt_path=reason1_ckpt_path, device=device, text_encoder=text_encoder
        )
        torch.save(text_embeddings, text_prompt_cache_path)
    else:
        text_embeddings = torch.load(text_prompt_cache_path, map_location=device)
    return text_embeddings


def _cut_number_of_views_in_data_batch(data_batch: dict, n_cameras: int) -> dict:
    """Cut the number of views to n_cameras."""
    # Get number of frames per view and calculate number of views
    num_video_frames_per_view = int(data_batch.get("num_video_frames_per_view", 93).cpu().item())
    target_VT = n_cameras * num_video_frames_per_view

    # Extract video data: shape [B, C, V*T, H, W]
    video_data = data_batch["video"]  # uint8, range [0, 255]
    data_batch["video"] = video_data[:, :, :target_VT, :, :]

    if "control_input_hdmap_bbox" in data_batch:
        hdmap_data = data_batch["control_input_hdmap_bbox"]  # uint8, range [0, 255]
        data_batch["control_input_hdmap_bbox"] = hdmap_data[:, :, :target_VT, :, :]

    if "ai_caption" in data_batch:
        captions = data_batch["ai_caption"]  # List of strings per view or single string
        data_batch["ai_caption"] = [c[:n_cameras] for c in captions]
    return data_batch


def parse_multi_view_data(root_dir: str, res_H: int = 480, res_W: int = 832, n_cameras: int = 4):
    data_batch = torch.load(root_dir)
    data_batch = _cut_number_of_views_in_data_batch(data_batch, n_cameras)

    # Extract video data: shape [B, C, V*T, H, W]
    video_data = data_batch["video"]  # uint8, range [0, 255]
    B, C, VT, H, W = video_data.shape

    # Get number of frames per view and calculate number of views
    num_video_frames_per_view = int(data_batch.get("num_video_frames_per_view", 93).cpu().item())
    n_views = VT // num_video_frames_per_view
    assert n_views == n_cameras, f"Expected {n_cameras} views, got {n_views} from data"

    if H != res_H or W != res_W:
        log.warning(
            f"Height {H} and width {W} do not match the target resolution {res_H} and {res_W}. Just use dummy data"
        )

        first_frames = np.zeros((n_views, res_H, res_W, 3), dtype=np.uint8)
        prompts = [""] * n_views
        hdmap_video = np.zeros((n_views, num_video_frames_per_view, res_H, res_W, 3), dtype=np.uint8)
    else:
        # Extract first num_condition_frames for each view: [B, C, V*T, H, W] -> list of [V] numpy arrays [T_cond, H, W, 3]
        num_condition_frames = 1
        first_frames = []
        for v in range(n_views):
            start_frame_idx = v * num_video_frames_per_view
            end_frame_idx = start_frame_idx + num_condition_frames
            # Extract frames: [C, T_cond, H, W]
            frames_view = video_data[0, :, start_frame_idx:end_frame_idx, :, :].cpu().numpy()
            # Transpose to [T_cond, H, W, C]
            frames_view = np.transpose(frames_view, (1, 2, 3, 0))
            first_frames.append(frames_view.astype(np.uint8))
        first_frames = np.concatenate(first_frames, axis=0)  # [V, H, W, C]

        # Extract prompts (captions)
        prompts = data_batch.get("ai_caption", [[""]])[0]  # List of strings per view or single string
        log.info(f"prompts: {prompts}")

        # Extract hdmap condition if present: [B, C, V*T, H, W]
        hdmap_multiview_list = []
        if "control_input_hdmap_bbox" in data_batch:
            hdmap_data = data_batch["control_input_hdmap_bbox"]  # uint8, range [0, 255]
            log.info(f"HDMap data shape: {hdmap_data.shape}")

            for v in range(n_views):
                start_idx = v * num_video_frames_per_view
                end_idx = (v + 1) * num_video_frames_per_view
                hdmap_view = hdmap_data[0, :, start_idx:end_idx, :, :].cpu().numpy()  # [C, T, H, W]
                hdmap_view = np.transpose(hdmap_view, (1, 2, 3, 0))  # [T, H, W, C]
                hdmap_multiview_list.append(hdmap_view.astype(np.uint8))
        else:
            log.warning("No hdmap condition found in data_batch")
            # Create dummy hdmap if not present
            for v in range(n_views):
                hdmap_view = np.zeros((num_video_frames_per_view, H, W, 3), dtype=np.uint8)
                hdmap_multiview_list.append(hdmap_view)
        hdmap_video = np.stack(hdmap_multiview_list, axis=0)  # [V, T, H, W, 3]

    return first_frames, prompts, hdmap_video, DEFAULT_4VIEWS_VIEW_INDICES


@dataclass
class DiffusionModelCache:
    batch_size: int
    latent_shape: tuple[int, int, int]  # T H W before CP
    num_tokens_per_frame: int  # after CP
    num_tokens_per_block: int  # after CP
    image_latent_B_Ellipsis_D: torch.Tensor  # after CP
    text_embeddings: torch.Tensor  # [B, L_text, D_text]
    network_cache: CosmosCausalDiTNetworkCache  # after CP
    tokenizer_cache: WANVAECache
    detokenizer_cache: WANVAECache
    condition_video_input_mask_first_block_B_Ellipsis_D: torch.Tensor  # after CP
    condition_video_input_mask_B_Ellipsis_D: torch.Tensor  # after CP
    rotary_frequency: SinsoidalRotaryFrequency  # already setup for CP
    shape_B_Ellipsis_D: list[int]  # the shape of the input tensor x_B_Ellipsis_D (after CP)
    denoising_steps: list[torch.Tensor]  # [B,]
    denoising_sigmas: list[torch.Tensor]  # [B,]
    kvcache_step: torch.Tensor  # [B,]
    kvcache_sigma: torch.Tensor  # [B,]
    stream_kv: torch.cuda.Stream
    evt_kv_ready: torch.cuda.Event
    evt_latent_ready: torch.cuda.Event
    view_indices_B_V: torch.Tensor | None


class DiffusionModel(torch.nn.Module):
    """
    A diffusion model for generating video blocks in a causal manner.
    """

    def precompute_and_cache(
        self,
        batch_size: int,
        text_prompt_or_embeddings: list[str] | torch.Tensor,
        image_array_or_tensor: np.ndarray | torch.Tensor,
        prealloc_blocks: int | None = None,
        view_indices: list[int] | None = None,
        seed: int | None = None,
        *args,
        **kwargs,
    ) -> DiffusionModelCache:
        """Precompute and cache the model.

        Args:
            batch_size: the batch size, only support batch size 1 for now
            text_prompt_or_embeddings: the text prompt strings for the generation or the text embeddings. Required shape is [1, V, L, D]
            image_array_or_tensor: the first frame of I2V generation. If it is a numpy array, it is expected
               to have shape [V, H, W, 3] in range [0, 255]. If it is a tensor, it is expected to have shape [1, V, 3, H, W] in range [-1, 1]
            prealloc_blocks: the number of blocks to preallocate for KV cache. It is perfectly fine to generate more blocks than this,
                but the runtime will be degraded.

        Returns:
            the cached data for the model
        """
        assert batch_size == 1, "Only support batch size 1 for now"
        B = batch_size

        if seed is not None:
            # re-seed for every rollout.
            if torch.distributed.is_initialized():
                # Each rank uses a different seed
                rank = torch.distributed.get_rank()
                seed += rank
            else:
                rank = 0
            log.info(f"Rank {rank} using seed {seed}", rank0_only=False)
            self.rng.manual_seed(seed)

        # input data preprocessing
        if isinstance(text_prompt_or_embeddings, list):
            text_embeddings = [
                preprocess_input_text_prompt(
                    text_prompt,
                    reason1_ckpt_path=self.reason1_ckpt_path,
                    device=self.device,
                    text_encoder=self.text_encoder,
                )
                for text_prompt in text_prompt_or_embeddings
            ]
            text_embeddings = torch.stack(text_embeddings, dim=1)  # [B, V, L, D]
        else:
            text_embeddings = text_prompt_or_embeddings
        assert text_embeddings.ndim == 4, "text_embeddings must have shape [B, V, L, D]"
        if self.temporal_group is not None:
            text_embeddings = split_inputs_cp(text_embeddings, seq_dim=1, cp_group=self.temporal_group)

        if isinstance(image_array_or_tensor, np.ndarray):
            image_B_V_C_H_W = preprocess_input_image(image_array_or_tensor, dtype=self.dtype, device=self.device)
        else:
            image_B_V_C_H_W = image_array_or_tensor  # [B, V, 3, H, W]

        V = image_B_V_C_H_W.shape[1]
        res_H = image_B_V_C_H_W.shape[3]
        res_W = image_B_V_C_H_W.shape[4]

        # latent shape
        T = self.num_latents_per_block
        H = res_H // self.tokenizer.spatial_compression_factor
        W = res_W // self.tokenizer.spatial_compression_factor

        latent_tensor = torch.randn((B, V, 16, T, H, W), device=self.device, dtype=self.dtype)
        latent_tensor_B_Ellipsis_D = self.net.patchify_and_maybe_split_cp(
            latent_tensor,
            process_groups=[self.view_group, self.temporal_group],
            cp_dims=[self.view_group_cp_dim, self.temporal_group_cp_dim],
        )
        shape_B_Ellipsis_D = list(latent_tensor_B_Ellipsis_D.shape)

        image_BV_C_1_H_W = image_B_V_C_H_W.reshape(B * V, 3, 1, res_H, res_W)
        image_latent_BV_C_1_H_W = self.tokenizer.encode(
            image_BV_C_1_H_W.to(self.tokenizer_device), cache=self.tokenizer.prepare_cache()
        ).to(self.device)  # [B*V, 16, 1, H, W]
        image_latent_B_V_C_1_H_W = image_latent_BV_C_1_H_W.reshape(B, V, -1, 1, H, W)
        # pad zero to the 1-dim to T
        image_latent_B_V_C_1_H_W = F.pad(image_latent_B_V_C_1_H_W, (0, 0, 0, 0, 0, T - 1))
        image_latent_B_Ellipsis_D = self.net.patchify_and_maybe_split_cp(
            image_latent_B_V_C_1_H_W,
            process_groups=[self.view_group, self.temporal_group],
            cp_dims=[self.view_group_cp_dim, self.temporal_group_cp_dim],
        )

        # latent shape after patchify
        pT = T // self.net.patch_temporal  # patch T
        pH = H // self.net.patch_spatial  # patch H
        pW = W // self.net.patch_spatial  # patch W
        num_tokens_per_frame = pH * pW // (self.view_group.size() if self.view_group is not None else 1)
        num_tokens_per_block = pT * pH * pW // (self.view_group.size() if self.view_group is not None else 1)
        latent_shape = (T, H, W)

        # prepare rotary position embeddings
        rotary_frequency = SinsoidalRotaryFrequency(
            head_dim=self.net.model_channels // self.net.num_heads,
            len_h=pH,
            len_w=pW,
            len_t=pT,
            h_extrapolation_ratio=self.rope_hw_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_hw_extrapolation_ratio,
            t_extrapolation_ratio=1.0,
            device=self.device,
        )
        # rotary frequency CP is only along T dimension.
        rotary_frequency.set_context_parallel_group(self.view_group, self.view_group_cp_dim)

        # prepare all the caches
        tokenizer_cache: WANVAECache = self.tokenizer.prepare_cache()
        detokenizer_cache: WANVAECache = self.detokenizer.prepare_cache()

        if self.local_attn_size > 0:
            local_attn_token_size = self.local_attn_size * num_tokens_per_frame
            sink_token_size = self.sink_size * num_tokens_per_frame
        else:
            local_attn_token_size = -1
            sink_token_size = 0

        prealloc_self_attn_batch_size = batch_size * V
        if self.temporal_group is not None:
            prealloc_self_attn_batch_size = prealloc_self_attn_batch_size // self.temporal_group.size()

        condition_video_input_mask_first_block_B_V_1_T_H_W = torch.zeros(
            (batch_size, V, 1, T, H, W), device=self.device, dtype=self.dtype
        )
        condition_video_input_mask_first_block_B_V_1_T_H_W[..., :1, :, :] = 1.0
        condition_video_input_mask_first_block_B_Ellipsis_D = self.net.patchify_and_maybe_split_cp(
            condition_video_input_mask_first_block_B_V_1_T_H_W,
            process_groups=[self.view_group, self.temporal_group],
            cp_dims=[self.view_group_cp_dim, self.temporal_group_cp_dim],
        )

        condition_video_input_mask_B_V_1_T_H_W = torch.zeros(
            (batch_size, V, 1, T, H, W), device=self.device, dtype=self.dtype
        )
        condition_video_input_mask_B_Ellipsis_D = self.net.patchify_and_maybe_split_cp(
            condition_video_input_mask_B_V_1_T_H_W,
            process_groups=[self.view_group, self.temporal_group],
            cp_dims=[self.view_group_cp_dim, self.temporal_group_cp_dim],
        )

        upsampler_type = getattr(self, "upsampler_type", "none")
        if upsampler_type != "none":
            from projects.cosmos.sil.upsampler import create_upsampler

            self.upsampler = create_upsampler(
                upsampler_type,
                scale=self.upsampler_scale,
                input_W=res_W,
                input_H=res_H,
                model_path=self.upsampler_path,
                num_views=V,
            )

        # setup streams and events
        if self.kv_cache_on_side_stream:
            stream_kv = torch.cuda.Stream(device=self.device)
        else:
            stream_kv = torch.cuda.current_stream()
        evt_kv_ready = torch.cuda.Event()
        evt_latent_ready = torch.cuda.Event()
        stream_kv.record_event(evt_kv_ready)  # unblock the first block

        if V > 1:
            assert V == 4, f"Only 4 views are supported for multiview generation, but got {V}"
            if view_indices is None:
                view_indices = DEFAULT_4VIEWS_VIEW_INDICES
            assert len(view_indices) == V, f"view_indices must have length {V}, but got {len(view_indices)}"
            view_indices_B_V = torch.tensor(view_indices, dtype=torch.long, device=self.device).expand(batch_size, V)
            if self.temporal_group is not None:
                view_indices_B_V = split_inputs_cp(view_indices_B_V, seq_dim=1, cp_group=self.temporal_group)
        else:
            view_indices_B_V = None

        if self.use_cuda_graphs:
            net_wrapper_key = (V, T, H, W, sink_token_size, local_attn_token_size)
            if net_wrapper_key in self.net_wrapper_dict:
                # If the shape configuration is the same. We can reuse a existing cudagraph
                # wrapper, which has graph captured already.
                self.net_wrapper = self.net_wrapper_dict[net_wrapper_key]
                # Assuming the text embeddings might be different across runs.
                # Here we need to update the cache with the new text embeddings.
                network_cache = self.net_wrapper._steady_network_cache
                if network_cache is None:
                    network_cache = self.net.prepare_cache(
                        prealloc_self_attn_batch_size,
                        sink_token_size,
                        local_attn_token_size,
                        text_embeddings,
                    )
                else:
                    # Use the cached network cache
                    self.net_wrapper.refresh_cache(text_embeddings)
            else:
                # Create a fresh cudagraph wrapper with network
                self.net_wrapper = CosmosCausalDiTCUDAGraphWrapper(
                    network=self.net,
                    sink_token_size=sink_token_size,
                    local_attn_token_size=local_attn_token_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                # cache the cudagraph wrapper
                self.net_wrapper_dict[net_wrapper_key] = self.net_wrapper
                network_cache = self.net.prepare_cache(
                    prealloc_self_attn_batch_size,
                    sink_token_size,
                    local_attn_token_size,
                    text_embeddings,
                )

        else:
            self.net_wrapper = self.net
            network_cache = self.net.prepare_cache(
                prealloc_self_attn_batch_size,
                sink_token_size,
                local_attn_token_size,
                text_embeddings,
            )

        return DiffusionModelCache(
            batch_size=batch_size,
            latent_shape=latent_shape,
            num_tokens_per_frame=num_tokens_per_frame,
            num_tokens_per_block=num_tokens_per_block,
            image_latent_B_Ellipsis_D=image_latent_B_Ellipsis_D,
            text_embeddings=text_embeddings,
            network_cache=network_cache,
            tokenizer_cache=tokenizer_cache,
            detokenizer_cache=detokenizer_cache,
            condition_video_input_mask_first_block_B_Ellipsis_D=condition_video_input_mask_first_block_B_Ellipsis_D,
            condition_video_input_mask_B_Ellipsis_D=condition_video_input_mask_B_Ellipsis_D,
            rotary_frequency=rotary_frequency,
            shape_B_Ellipsis_D=shape_B_Ellipsis_D,
            denoising_steps=[t.expand(batch_size) for t in self.denoising_step_list],
            denoising_sigmas=[s.expand(batch_size) for s in self.denoising_sigma_list],
            kvcache_step=self.kvcache_step.expand(batch_size),
            kvcache_sigma=self.kvcache_sigma.expand(batch_size),
            stream_kv=stream_kv,
            evt_kv_ready=evt_kv_ready,
            evt_latent_ready=evt_latent_ready,
            view_indices_B_V=view_indices_B_V,
        )

    @torch.no_grad()
    @NVTXRangeDecorator("upsample_video_chunk")
    def upsample_video_chunk(self, video_chunk: torch.Tensor, is_final_chunk: bool = False) -> list[torch.Tensor]:
        """Upsample a video chunk.

        Args:
            video_chunk: The video chunk to be upsampled.
            is_final_chunk: Whether this is the final chunk of the video.

        Returns:
            The list of upsampled video chunks. If no upsampling is used, return an empty list.
        """
        upsampler = getattr(self, "upsampler", None)
        if upsampler is None:
            return []
        return upsampler(video_chunk, is_final_chunk=is_final_chunk)

    @torch.no_grad()
    @NVTXRangeDecorator("streaming_inference_one_block")
    def streaming_inference_one_block(
        self,
        block_index: int,
        cache: DiffusionModelCache,
        # stream_manager: StreamManager,
        hdmap: torch.Tensor,
        events: ProfileEvents | None = None,
        do_view_split_cp: bool = True,
        do_view_gather_cp: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Generate one block of video and return state for KV cache update.
        Args:
            block_index: the index of the block to be generated.
            cache: the cache of the current generation status.
            hdmap: the hdmap condition tensor with shape [1, V, 3, T, H, W] in range [-1, 1]

        Returns:
            video: the generated video tensor with shape [1, V, 3, T, H, W] in range [-1, 1]
            finalization_state: dict containing data needed for KV cache update
        """
        if events is not None:
            events.tic.record()

        with NVTXRangeDecorator("rotary frequency"):
            rope_emb = cache.rotary_frequency.shift_t(block_index * self.num_latents_per_block)
        if events is not None:
            events.toc_after_rotary.record()

        with NVTXRangeDecorator("encode HD map"):
            # Note: here we assume the hdmap comes in as the global data!

            if self.encode_with_pixel_shuffle:
                hdmap_condition_B_Ellipsis_D = self.encode_one_block_pixel_shuffle(
                    block_index, hdmap, do_view_split_cp=do_view_split_cp
                )
            else:
                hdmap_condition_B_Ellipsis_D = self.encode_one_block(
                    block_index, hdmap, cache.tokenizer_cache, do_view_split_cp=do_view_split_cp
                )
        if events is not None:
            events.toc_after_encode.record()

        with NVTXRangeDecorator("generate one block"):
            clean_latent_B_Ellipsis_D = self.generate_one_block(
                block_index, hdmap_condition_B_Ellipsis_D, cache, rope_emb
            )
        if events is not None:
            events.toc_after_denoise.record()

        with NVTXRangeDecorator("decode one block"):
            video_B_V_C_T_H_W = self.decode_one_block(
                clean_latent_B_Ellipsis_D, cache, do_view_gather_cp=do_view_gather_cp
            )
        if events is not None:
            events.toc_after_decode.record()

        # Return video + state needed for finalization
        finalization_state = {
            "block_index": block_index,
            "clean_latent_B_Ellipsis_D": clean_latent_B_Ellipsis_D,
            "hdmap_condition_B_Ellipsis_D": hdmap_condition_B_Ellipsis_D,
            "rope_emb": rope_emb,
            "cache": cache,
            "events": events,
        }

        return video_B_V_C_T_H_W, finalization_state

    @torch.no_grad()
    def finalize_block_generation(self, finalization_state: dict) -> None:
        """Update KV cache for efficient causal attention for next block .

        Args:
            finalization_state: dict returned from streaming_inference_one_block
        """
        with NVTXRangeDecorator("update KV cache"):
            self.update_kv_cache(
                block_index=finalization_state["block_index"],
                clean_latent_B_Ellipsis_D=finalization_state["clean_latent_B_Ellipsis_D"],
                hdmap_condition_B_Ellipsis_D=finalization_state["hdmap_condition_B_Ellipsis_D"],
                cache=finalization_state["cache"],
                rope_emb=finalization_state["rope_emb"],
            )

        events = finalization_state["events"]
        if events is not None:
            events.toc_after_update_kv.record()

    @torch.no_grad()
    def generate_one_block(
        self,
        block_index: int,
        hdmap_condition_B_Ellipsis_D: torch.Tensor,
        cache: DiffusionModelCache,
        rope_emb: torch.Tensor,
    ) -> torch.Tensor:
        stream = torch.cuda.current_stream()

        # wait for the previous block's KV cache update to finish
        stream.wait_event(cache.evt_kv_ready)

        num_tokens_per_block = cache.num_tokens_per_block  # num tokens after CP
        current_start = block_index * num_tokens_per_block

        # first block is conditioned on the input image
        if block_index == 0:
            condition_video_input_mask_B_Ellipsis_D = (
                cache.condition_video_input_mask_first_block_B_Ellipsis_D
            )  # [B, <TOKEN_MEMORY_LAYOUT>, D]
        else:
            condition_video_input_mask_B_Ellipsis_D = (
                cache.condition_video_input_mask_B_Ellipsis_D
            )  # [B, <TOKEN_MEMORY_LAYOUT>, D]
        assert condition_video_input_mask_B_Ellipsis_D.dim() == 5, (
            "condition_video_input_mask_B_Ellipsis_D must have shape [B, <TOKEN_MEMORY_LAYOUT>, D]"
        )

        network_kwargs = {
            "condition_video_input_mask_B_Ellipsis_D": condition_video_input_mask_B_Ellipsis_D,
            "network_cache": cache.network_cache,
            "current_start": current_start,
            "hdmap_condition_B_Ellipsis_D": hdmap_condition_B_Ellipsis_D,
            "rope_emb": rope_emb,
            "view_indices_B_V": cache.view_indices_B_V,
        }

        clean_latent_B_Ellipsis_D = run_network_denoising(
            denoising_timestamps=cache.denoising_steps,
            denoising_sigmas=cache.denoising_sigmas,
            network=self.net_wrapper,
            shape_B_Ellipsis_D=cache.shape_B_Ellipsis_D,
            device=self.device,
            dtype=self.dtype,
            image_latent_B_Ellipsis_D=cache.image_latent_B_Ellipsis_D if block_index == 0 else None,
            mask_B_Ellipsis_1=condition_video_input_mask_B_Ellipsis_D[..., :1] if block_index == 0 else None,
            rng=self.rng,
            **network_kwargs,
        )

        # mark the latent as ready
        stream.record_event(cache.evt_latent_ready)
        return clean_latent_B_Ellipsis_D

    @torch.no_grad()
    def update_kv_cache(
        self,
        block_index: int,
        clean_latent_B_Ellipsis_D: torch.Tensor,
        hdmap_condition_B_Ellipsis_D: torch.Tensor,
        cache: DiffusionModelCache,
        rope_emb: torch.Tensor,
    ) -> None:
        stream = cache.stream_kv
        with torch.cuda.stream(stream):
            # wait for the latent to be ready
            stream.wait_event(cache.evt_latent_ready)

            num_tokens_per_block = cache.num_tokens_per_block  # num tokens after CP
            current_start = block_index * num_tokens_per_block

            # first block is conditioned on the input image
            if block_index == 0:
                condition_video_input_mask_B_Ellipsis_D = (
                    cache.condition_video_input_mask_first_block_B_Ellipsis_D
                )  # [B, <TOKEN_MEMORY_LAYOUT>, D]
            else:
                condition_video_input_mask_B_Ellipsis_D = (
                    cache.condition_video_input_mask_B_Ellipsis_D
                )  # [B, <TOKEN_MEMORY_LAYOUT>, D]
            assert condition_video_input_mask_B_Ellipsis_D.dim() == 5, (
                "condition_video_input_mask_B_Ellipsis_D must have shape [B, <TOKEN_MEMORY_LAYOUT>, D]"
            )

            network_kwargs = {
                "condition_video_input_mask_B_Ellipsis_D": condition_video_input_mask_B_Ellipsis_D,
                "network_cache": cache.network_cache,
                "current_start": current_start,
                "hdmap_condition_B_Ellipsis_D": hdmap_condition_B_Ellipsis_D,
                "rope_emb": rope_emb,
                "view_indices_B_V": cache.view_indices_B_V,
            }

            # kv cache will be updated in place
            run_network_denoising(
                denoising_timestamps=[cache.kvcache_step],
                denoising_sigmas=[cache.kvcache_sigma],
                network=self.net,
                shape_B_Ellipsis_D=cache.shape_B_Ellipsis_D,
                device=self.device,
                dtype=self.dtype,
                clean_latent_B_Ellipsis_D=clean_latent_B_Ellipsis_D,
                rng=self.rng,
                **network_kwargs,
            )

            # mark the kv cache as ready
            stream.record_event(cache.evt_kv_ready)
        return

    @torch.no_grad()
    def decode_one_block(
        self, clean_latent_B_Ellipsis_D: torch.Tensor, cache: DiffusionModelCache, do_view_gather_cp: bool = True
    ) -> torch.Tensor:
        # gather T
        clean_latent_B_V_C_T_H_W = self.net.unpatchify_and_maybe_gather_cp(
            pH=cache.latent_shape[1] // 2,
            pW=cache.latent_shape[2] // 2,
            x=clean_latent_B_Ellipsis_D,
            process_groups=[self.view_group],
            cp_dims=[self.view_group_cp_dim],
        )
        # VAE decode
        B, V, C, T, H, W = clean_latent_B_V_C_T_H_W.shape
        clean_latent_BV_C_T_H_W = clean_latent_B_V_C_T_H_W.reshape(B * V, C, T, H, W)
        video = self.detokenizer.decode(clean_latent_BV_C_T_H_W, cache=cache.detokenizer_cache)  # range [-1, 1]
        video = (1.0 + video) / 2.0  # range [0, 1]
        video = video.reshape(B, V, *video.shape[1:])  # [B, V, 3, T, H, W] in range [0, 1]
        # gather V
        if (self.temporal_group is not None) and do_view_gather_cp:
            video = cat_outputs_cp(video, seq_dim=1, cp_group=self.temporal_group)
        return video

    def _get_encode_cp_setup(
        self, do_view_split_cp: bool
    ) -> tuple[list[ProcessGroup | None], list[ContextParallelDim]]:
        # If input video is already split across view CP, we do not need to split it again.
        #   - view_group: ProcessGroup for ranks processing same view (for T gathering)
        #   - temporal_group: ProcessGroup for ranks at same T slice (for V gathering)
        if do_view_split_cp:
            process_groups = [self.view_group, self.temporal_group]
            cp_dims = [self.view_group_cp_dim, self.temporal_group_cp_dim]
        else:
            process_groups = [self.view_group]
            cp_dims = [self.view_group_cp_dim]
        return process_groups, cp_dims

    @torch.no_grad()
    def encode_one_block_pixel_shuffle(
        self,
        block_index: int,
        video_B_V_C_T_H_W: torch.Tensor,  # [B, V, C, T, H, W] in range [-1, 1]
        frame_selection_mode: Literal["first_frame", "last_frame"] = "last_frame",
        do_view_split_cp: bool = True,
    ) -> torch.Tensor:
        process_groups, cp_dims = self._get_encode_cp_setup(do_view_split_cp)

        assert video_B_V_C_T_H_W.ndim == 6, "input is expected to be 6D tensor with shape [B, V, C, T, H, W]"
        T = video_B_V_C_T_H_W.shape[-3]
        if frame_selection_mode == "first_frame":
            if block_index == 0:
                indices = [0] + list(range(1, T, 4))
            else:
                indices = list(range(0, T, 4))
        elif frame_selection_mode == "last_frame":
            if block_index == 0:
                indices = [0] + list(range(4, T, 4))
            else:
                indices = list(range(3, T, 4))
        else:
            raise ValueError(f"Invalid frame selection mode: {frame_selection_mode}")
        frames = video_B_V_C_T_H_W[..., indices, :, :]
        latent = rearrange(frames, "... c t (h h8) (w w8) -> ... (c h8 w8) t h w", h8=8, w8=8)

        # patchify the hdmap condition
        latent_B_Ellipsis_D = self.net.patchify_and_maybe_split_cp(
            latent, process_groups=process_groups, cp_dims=cp_dims
        )
        return latent_B_Ellipsis_D  # [B, <TOKEN_MEMORY_LAYOUT>, D]

    @torch.no_grad()
    def encode_one_block(
        self,
        block_index: int,
        video_B_V_C_T_H_W: torch.Tensor,  # [B, V, C, T, H, W] in range [-1, 1]
        tokenizer_cache: WANVAECache,
        do_view_split_cp: bool = True,
    ) -> torch.Tensor:
        process_groups, cp_dims = self._get_encode_cp_setup(do_view_split_cp)

        assert video_B_V_C_T_H_W.ndim == 6, "input is expected to be 6D tensor with shape [B, V, C, T, H, W]"
        B, V, C, T, H, W = video_B_V_C_T_H_W.shape
        video_BV_C_T_H_W = video_B_V_C_T_H_W.reshape(B * V, C, T, H, W)
        stride = 1 if block_index == 0 else 4
        start = 0
        latent = []
        while start < video_BV_C_T_H_W.shape[2]:
            end = start + stride
            video_chunk = video_BV_C_T_H_W[:, :, start:end, :, :]
            latent_chunk = self.tokenizer.encode(video_chunk, cache=tokenizer_cache)  # [B, C, T, H, W]
            latent.append(latent_chunk)
            stride = 4  # after the first chunk, stride is always 4
            start = end
        latent_BV_C_T_H_W = torch.cat(latent, dim=2)
        latent_B_V_C_T_H_W = latent_BV_C_T_H_W.reshape(B, V, *latent_BV_C_T_H_W.shape[1:])

        # patchify the hdmap condition
        latent_B_Ellipsis_D = self.net.patchify_and_maybe_split_cp(
            latent_B_V_C_T_H_W, process_groups=process_groups, cp_dims=cp_dims
        )
        return latent_B_Ellipsis_D  # [B, <TOKEN_MEMORY_LAYOUT>, D]

    @torch.no_grad()
    def get_current_video_frame_range(self, block_index: int) -> tuple[int, int]:
        T = self.num_latents_per_block
        if block_index == 0:
            start = 0
            end = (T - 1) * 4 + 1
        else:
            start = (T - 1) * 4 + 1 + (block_index - 1) * T * 4
            end = (T - 1) * 4 + 1 + block_index * T * 4
        return start, end
