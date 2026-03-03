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
This script is based on projects/cosmos/diffusion/v2/inference/vid2vid.py

To run inference on the training data (as visualization/debugging), use:
```bash
EXP=buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform-0/checkpoints/iter_000006500/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference --seed 0 --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --max_samples 1 --save_root results/
```

```bash
EXP=transfer2p5_2b_mv_7train7_res720p_fps10_t24_frombase2p5avfinetune_mads_only_allcaption_uniform_nofps_wm_condition
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2_mv2/transfer2p5_2b_mv_7train5_res720p_fps10_t24_frombase2p5avfinetune_mads_only_allcaption_uniform_nofps_wm_condition-0/checkpoints/iter_000005500
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference --seed 0 --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --max_samples 30 --save_root results/transfer2_multiview/7train5_5500 --num_conditional_frames 0
```

"""

import math
import os
import random
from typing import Optional

import einops
import numpy as np
import torch
from loguru import logger
from megatron.core import parallel_state

from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.predict2.utils.model_loader import load_model_from_checkpoint


def set_seeds(seed: int, deterministic: bool = False):
    """
    Set all random seeds for maximum reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable all deterministic settings
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # CuDNN settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True)

        # Set environment variables for additional reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic CUBLAS
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(f"All random seeds set to {seed}, deterministic mode: {deterministic}")


def to_model_input(data_batch, model):
    """
    Similar to misc.to, but avoid converting uint8 "video" to float
    """
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


def _num_conditional_frames_for_batch(
    chunk_idx: int,
    tokenizer,
    num_conditional_pixel_frames: int | list[int],
    chunk_overlap: int,
):
    """
    Builds "num_conditional_frames" for a batch.

    Inputs to generate_autoregressive_from_batch are in raw/pixel frames but the model works with latent frames.
    This helper applies the conversion.
    """
    if chunk_idx == 0:
        if isinstance(num_conditional_pixel_frames, list):
            return [tokenizer.get_latent_num_frames(raw_frames) for raw_frames in num_conditional_pixel_frames]
        return tokenizer.get_latent_num_frames(num_conditional_pixel_frames)

    return tokenizer.get_latent_num_frames(chunk_overlap)


class ControlVideo2WorldInference:
    """
    Handles the Vid2Vid inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        context_parallel_size: int = 1,
        hierarchical_cp: bool = False,
        experiment_opts: Optional[list[str]] = None,
    ):
        """
        Initializes the Vid2VidInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
            experiment_opts (tuple[str, ...]): Experiment options overrides.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        if "RANK" in os.environ:
            self._init_distributed(hierarchical_cp=hierarchical_cp)

        if experiment_opts is None:
            experiment_opts = []
        # data_train is training-only, remove it during inference to avoid missing config errors in non-internal builds
        if not INTERNAL:
            experiment_opts.append("~data_train")
        # Load the model and config
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py",
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
        )

        # Enable context parallel on the model if using context parallelism
        self.rank0 = True
        if self.context_parallel_size > 1:
            # A2A+P2P enables using CP sizes larger than number of attention heads.
            cp_comm_type = "a2a+p2p" if hierarchical_cp else "p2p"
            model.net.enable_context_parallel(self.process_group, cp_comm_type=cp_comm_type)
            self.rank0 = distributed.get_rank() == 0

        self.model = model
        self.config = config
        self.batch_size = 1

    def _init_distributed(self, hierarchical_cp: bool = False, num_attention_heads: int = 16):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # We use as big A2A size as possible, given the CP size and number of attention heads.
        a2a_size = math.gcd(self.context_parallel_size, num_attention_heads)
        p2p_size = max(1, self.context_parallel_size // a2a_size)
        hierarchical_context_parallel_sizes = [a2a_size, p2p_size] if hierarchical_cp else None

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
            hierarchical_context_parallel_sizes=hierarchical_context_parallel_sizes,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()
        logger.info(f"Initialized context parallel with size {self.context_parallel_size}")
        logger.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def generate_from_batch(
        self,
        data_batch,
        guidance: float,
        seed: int,
        num_steps: int,
        use_negative_prompt: bool,
    ):
        """Generate video tensor from batch.

        Returns:
            tensor of shape (1, 3, v * t, h, w)
            where t is the number of frames, v is the number of views, h is the height, and w is the width
            The values are in the range [0, 1]
        """
        data_batch = to_model_input(data_batch, self.model)
        if self.model.config.text_encoder_config is not None and self.model.config.text_encoder_config.compute_online:
            self.model.inplace_compute_text_embeddings_online(data_batch)

        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        self.model.eval()
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            seed=seed,  # Fixed seed for reproducibility
            num_steps=num_steps,
            is_negative_prompt=use_negative_prompt,
        )
        # (bsz = 1, c = 3, t = n_camera * t, h, w)
        return ((self.model.decode(sample) + 1.0) / 2.0).clamp(0, 1)

    def generate_autoregressive_from_batch(
        self,
        full_batch: dict[str, torch.Tensor],
        n_views: int,
        chunk_overlap: int,
        chunk_size: int,
        guidance: float,
        seed: int,
        num_conditional_frames: int | list[int],
        num_steps: int,
        use_negative_prompt: bool,
        hint_keys: str = "hdmap_bbox",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate video using autoregressive sliding window approach.

        Args:
            full_batch: Full batch containing all video frames loaded by the dataloader
            n_views: Number of camera views
            num_video_frames_per_view: Chunk size (model's native capacity)
            chunk_overlap: Number of overlapping frames between chunks
            guidance: Guidance scale for generation
            seed: Random seed for generation
            num_conditional_frames: Number of conditional pixel frames (scalar or per-view list)
            num_steps: Number of sampling steps for the model.
            use_negative_prompt: Whether to use default negative prompt.

        Returns:
            Tuple of (generated video tensor, control video tensor)
            Both tensors contain values between 0 and 1.
        """
        if "num_conditional_frames" in full_batch:
            raise ValueError("num_conditional_frames should be passed as an argument")

        # Extract full video and control tensors
        full_control = full_batch[f"control_input_{hint_keys}"]  # Shape: [1, 3, total_frames, H, W]
        batch_size, channels, total_frames, height, width = full_batch[
            "video"
        ].shape  # Shape: [1, 3, total_frames, H, W]
        chunk_overlap = int(chunk_overlap)
        overlap = chunk_overlap

        # Calculate frames per view from the loaded video
        frames_per_view = total_frames // n_views
        if self.rank0:
            logger.info(f"Total frames loaded: {total_frames}, Frames per view: {frames_per_view}, Views: {n_views}")

        # Initialize output video list
        generated_chunks = []

        # Calculate number of chunks needed
        effective_chunk_size = chunk_size - overlap
        num_chunks = max(1, (frames_per_view - overlap + effective_chunk_size - 1) // effective_chunk_size)

        if self.rank0:
            logger.info(f"Generating {num_chunks} chunks with overlap {overlap}")

        # Generate first chunk using original input videos
        current_input_video = full_batch["video"].clone()
        tokenizer = self.model.tokenizer

        for chunk_idx in range(num_chunks):
            # Calculate frame range for this chunk
            start_frame = chunk_idx * effective_chunk_size
            end_frame = min(start_frame + chunk_size, frames_per_view)

            if start_frame >= frames_per_view:
                break

            if self.rank0:
                logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}, frames {start_frame}-{end_frame}")

            # Create chunk batch (extract 29-frame window from full video)
            chunk_batch = self._create_chunk_batch(
                full_batch, current_input_video, full_control, start_frame, end_frame, n_views, hint_keys
            )

            chunk_batch["num_conditional_frames"] = _num_conditional_frames_for_batch(
                chunk_idx, tokenizer, num_conditional_frames, chunk_overlap
            )

            # Generate chunk
            chunk_video = self.generate_from_batch(
                chunk_batch,
                guidance=guidance,
                seed=int(seed) + chunk_idx,
                num_steps=num_steps,
                use_negative_prompt=use_negative_prompt,
            )[0]  # C_T_H_W
            chunk_video = einops.rearrange(chunk_video, "C (V T) H W -> V C T H W", V=n_views)
            # Store generated chunk (remove overlap from previous chunks)
            if chunk_idx == 0:
                generated_chunks.append(chunk_video)
            else:
                # Remove overlap frames from the beginning of this chunk
                generated_chunks.append(chunk_video[:, :, overlap:])

            # Update input video for next iteration using generated frames
            if chunk_idx < num_chunks - 1:  # Not the last chunk
                # Handle num_conditional_frames being either int or list[int]
                if chunk_idx == 0:
                    overlap_value = (
                        max(num_conditional_frames)
                        if isinstance(num_conditional_frames, list)
                        else num_conditional_frames
                    )
                else:
                    overlap_value = chunk_overlap

                current_input_video = self._update_input_video_with_generated(
                    current_input_video,
                    chunk_video,
                    start_frame,
                    end_frame,
                    n_views,
                    overlap_value,
                )

        # Concatenate all chunks along time dimension
        final_video = torch.cat(generated_chunks, dim=2)

        # Return the corresponding control video for the same time range, between 0 and 1
        final_control = einops.rearrange(full_control[0].float() / 255.0, "C (V T) H W -> V C T H W", V=n_views)[
            :, :, : final_video.shape[2]
        ]

        final_video = einops.rearrange(final_video, "V C T H W -> C (V T) H W")
        final_control = einops.rearrange(final_control, "V C T H W -> C (V T) H W")
        return final_video.cpu(), final_control.cpu()

    def _create_chunk_batch(
        self,
        original_batch: dict[str, torch.Tensor],
        input_video: torch.Tensor,
        control_video: torch.Tensor,
        start_frame: int,
        end_frame: int,
        n_views: int,
        hint_keys: str,
    ) -> dict[str, torch.Tensor]:
        """
        Create a batch for a specific chunk by extracting a start_frame:end_frame window from each view.

        Args:
            original_batch: Original batch with metadata
            input_video: Full input video tensor [1, 3, total_frames, H, W]
            control_video: Full control video tensor [1, 3, total_frames, H, W]
            start_frame: Start frame index within each view
            end_frame: End frame index within each view
            n_views: Number of views (7)

        Returns:
            Batch dictionary with (end_frame - start_frame)*n_views frame tensors
        """
        chunk_batch = {}

        # Copy non-video fields from original batch
        for key, value in original_batch.items():
            if key not in ["video", f"control_input_{hint_keys}"]:
                chunk_batch[key] = value

        # Calculate frames per view in the full video
        input_video_NVCTHW = einops.rearrange(input_video, "N C (V T) H W -> N V C T H W", V=n_views)
        input_video_chunk = input_video_NVCTHW[:, :, :, start_frame:end_frame, :, :]
        control_video_NVCTHW = einops.rearrange(control_video, "N C (V T) H W -> N V C T H W", V=n_views)
        control_video_chunk = control_video_NVCTHW[:, :, :, start_frame:end_frame, :, :]
        view_indices = einops.rearrange(original_batch["view_indices"], "N (V T) -> N V T", V=n_views)
        view_indices_chunk = view_indices[:, :, start_frame:end_frame]

        input_video_chunk = einops.rearrange(input_video_chunk, "N V C T H W -> N C (V T) H W")
        control_video_chunk = einops.rearrange(control_video_chunk, "N V C T H W -> N C (V T) H W")
        view_indices_chunk = einops.rearrange(view_indices_chunk, "N V T -> N (V T)")
        chunk_batch["video"] = input_video_chunk.clone()
        chunk_batch[f"control_input_{hint_keys}"] = control_video_chunk.clone()
        chunk_batch["num_video_frames_per_view"] = torch.tensor(
            [
                end_frame - start_frame,
            ]
        ).to(original_batch["num_video_frames_per_view"])
        chunk_batch["view_indices"] = view_indices_chunk.clone()
        chunk_batch["fps"] = torch.tensor([30.0]).to(original_batch["fps"])

        return chunk_batch

    def _update_input_video_with_generated(
        self, current_input, generated_chunk, start_frame, end_frame, n_views, overlap
    ):
        """
        Update input video with generated frames for next iteration.

        Args:
            current_input: Full input video [1, 3, total_frames, H, W]
            generated_chunk: Generated chunk [ 7, 3, 29, H, W]
            start_frame: Start pixel-frame index within each view
            end_frame: End pixel-frame index within each view
            n_views: Number of views
            overlap: Number of overlapping pixel frames

        Returns:
            Updated input video with generated frames replacing future frames
        """
        chunk_frames_per_view = generated_chunk.shape[2]  # Should be 29

        update_start = start_frame + overlap  # Skip overlap frames
        update_end = end_frame
        gen_view_start = overlap
        gen_view_end = chunk_frames_per_view
        actual_update_end = min(update_end, current_input.shape[2])
        actual_gen_end = gen_view_end

        frames_to_copy = min(actual_update_end - update_start, actual_gen_end - gen_view_start)
        if self.rank0:
            logger.info(f"Frames to copy: {frames_to_copy}")
            logger.info(f"Update start: {update_start}, Update end: {update_end}")
            logger.info(f"Gen view start: {gen_view_start}, Gen view end: {gen_view_end}")
        current_input_NVCTHW = einops.rearrange(current_input.clone(), "N C (V T) H W -> N V C T H W", V=n_views)
        generated_chunk_NVCTHW = generated_chunk.unsqueeze(0)
        generated_chunk_NVCTHW = (generated_chunk_NVCTHW * 255.0).to(current_input.dtype)

        current_input_NVCTHW[:, :, :, update_start : update_start + frames_to_copy] = generated_chunk_NVCTHW[
            :, :, :, gen_view_start : gen_view_start + frames_to_copy
        ]
        updated_input = einops.rearrange(current_input_NVCTHW, "N V C T H W -> N C (V T) H W", V=n_views)

        return updated_input

    def cleanup(self):
        """Clean up distributed resources."""
        if "RANK" in os.environ:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
