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
Fast streaming inference for Cosmos2 video generation with hierarchical context parallelism.

This script loads model checkpoints from HuggingFace Hub.
"""

import argparse
import os

import mediapy as media
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from huggingface_hub import snapshot_download
from torch.distributed import ProcessGroup

from cosmos_transfer2._src.av.causal.causal_training.schedulers.flow_match import FlowMatchScheduler
from cosmos_transfer2._src.av.causal.fast_infer.utils.profile import NVTXRangeDecorator
from cosmos_transfer2._src.av.causal.fast_infer.v2.context_parallel_strategy import create_hierarchical_cp_groups
from cosmos_transfer2._src.av.causal.fast_infer.v2.demo_cosmos2 import (
    DiffusionModel as _BaseDiffusionModel,
)
from cosmos_transfer2._src.av.causal.fast_infer.v2.demo_cosmos2 import (
    DiffusionModelCache,
    ProfileEvents,
    preprocess_input_hdmap_video,
)
from cosmos_transfer2._src.av.causal.fast_infer.v2.network_cosmos2 import (
    ContextParallelDim,
    prepare_network,
    run_network_denoising,
)
from cosmos_transfer2._src.av.causal.fast_infer.v2.text_encoder import (
    TextEncoder,
    TextEncoderConfig,
    get_reason1_embeddings,
)
from cosmos_transfer2._src.av.causal.fast_infer.v2.wan2pt1_tokenizer import Wan2pt1VAEInterface
from cosmos_transfer2._src.av.causal.light_tokenizers import LightWan2pt1TokenizerInterface
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

# ---------------------------------------------------------------------------
# HuggingFace checkpoint layout (inside av_closed_loop/)
# ---------------------------------------------------------------------------
HF_REPO_ID = "nvidia-cosmos-ea/Cosmos-Experimental"
HF_SUBFOLDER = "av_closed_loop"

DEFAULT_PROMPT = (
    "Driving scene from a front-facing car camera. Urban environment with roads, vehicles, pedestrians, "
    "traffic signs, and buildings. Clear visibility, realistic lighting, photorealistic quality. "
    "High resolution dashcam footage of city driving."
)

# ---------------------------------------------------------------------------
# Example data shipped on HuggingFace (inside av_closed_loop/examples/)
# ---------------------------------------------------------------------------
# Key: (reso_bucket,)  Value: (video_filename, hdmap_filename)
_HF_EXAMPLES: dict[str, tuple[str, str]] = {
    "480p": ("example_video_480p.mp4", "example_hdmap_480p.mp4"),
    "720p": ("example_video_720p.mp4", "example_hdmap_720p.mp4"),
}

# ---------------------------------------------------------------------------
# Default checkpoint registry for HuggingFace
# ---------------------------------------------------------------------------
# Key: (reso_bucket, pixel_shuffle)
# Value: filename of the .pt state dict inside av_closed_loop/
_HF_CHECKPOINTS: dict[tuple, tuple[str, float]] = {
    # (reso_bucket, pixel_shuffle): (filename, rope_hw_extrapolation_ratio)
    ("480p", False): ("dit_480p_vae.pt", 2.0),
    ("480p", True): ("dit_480p_pixshuffle.pt", 2.0),
    ("720p", False): ("dit_720p_vae.pt", 3.0),
    ("720p", True): ("dit_720p_pixshuffle.pt", 3.0),
}


def resolve_hf_checkpoint(reso: str, pixel_shuffle: bool) -> tuple[str, float]:
    """Look up the HF checkpoint filename for the given config.

    Returns:
        (filename, rope_hw_extrapolation_ratio)
    """
    reso_bucket = "480p" if reso == "480p" else "720p"
    key = (reso_bucket, pixel_shuffle)
    if key not in _HF_CHECKPOINTS:
        raise ValueError(f"No HF checkpoint for reso={reso}, pixel_shuffle={pixel_shuffle}")
    return _HF_CHECKPOINTS[key]


def download_hf_checkpoints(
    repo_id: str = HF_REPO_ID,
    subfolder: str = HF_SUBFOLDER,
    local_dir: str | None = None,
    token: str | None = None,
) -> str:
    """Download the av_closed_loop folder from HuggingFace Hub.

    Uses HF_HOME env var for cache location if local_dir is not specified.

    Returns the path to the downloaded subfolder.
    """
    if local_dir is None:
        local_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "cosmos")
    log.info(f"Downloading checkpoints from {repo_id}/{subfolder} to {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{subfolder}/**",
        local_dir=local_dir,
        token=token,
    )
    local_subfolder = os.path.join(local_dir, subfolder)
    log.info(f"Checkpoints downloaded to {local_subfolder}")
    return local_subfolder


def preprocess_input_text_prompt(
    text_prompt: str, reason1_ckpt_path: str, device: torch.device | str, text_encoder: TextEncoder | None = None
) -> torch.Tensor:
    """Preprocess the text prompt (simplified version without keyword mappings)."""
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


def parse_single_view_data(
    video_path: str,
    hdmap_path: str,
    prompt: str,
    res_H: int = 480,
    res_W: int = 832,
    fps: int = 10,
):
    """Parse single-view data from user-provided video and hdmap paths."""
    n_subsample = {10: 3, 20: 2, 30: 1}[fps]

    # first frame
    gt_video = media.read_video(video_path)
    first_frame = media.resize_image(gt_video[0], (res_H, res_W))  # (H, W, 3)
    first_frame = first_frame[np.newaxis, ...]  # (1, H, W, 3)
    log.info(f"Loaded first frame with shape {first_frame.shape}")

    prompt_list = [prompt]
    log.info(f"Text prompt: {prompt}")

    # hdmap condition
    hdmap_video = media.read_video(hdmap_path)
    hdmap_video = hdmap_video[::n_subsample]
    hdmap_video = media.resize_video(hdmap_video, (res_H, res_W))  # (T, H, W, 3)
    hdmap_video = hdmap_video[np.newaxis, ...]  # (1, T, H, W, 3)
    log.info(f"Loaded hdmap condition with shape {hdmap_video.shape}")
    return first_frame, prompt_list, hdmap_video


class DiffusionModel(_BaseDiffusionModel):
    """DiffusionModel for HuggingFace checkpoint loading.

    Extends the base DiffusionModel with:
    - Direct .pt checkpoint loading (no DCP/S3)
    - Qwen tokenizer patching for HF public models
    - Configurable KV cache writes during denoising
    """

    def __init__(
        self,
        ckpt_path: str,
        reason1_ckpt_path: str,
        reso: str = "720p",
        light_vae_tokenizer: bool = False,
        light_vae_detokenizer: bool = True,
        light_vae_path: str | None = None,
        compile_net: bool = False,
        num_frames_per_block: int = 12,
        context_noise: int = 128,
        denoising_step_list: list[int] = [1000, 750, 500, 250],
        warp_denoising_step: bool = True,
        device: torch.device = torch.device("cuda:0"),
        tokenizer_device: torch.device = torch.device("cuda:0"),
        detokenizer_device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        encode_with_pixel_shuffle: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
        enable_cross_view_attn: bool = False,
        skip_ckpt: bool = False,
        view_group: ProcessGroup | None = None,
        temporal_group: ProcessGroup | None = None,
        use_cuda_graphs: bool = False,
        kv_cache_on_side_stream: bool = False,
        no_tae: bool = False,
        no_vae_parallel: bool = False,
        vae_chunk_parallel: bool = True,
        rope_hw_extrapolation_ratio: float | None = None,
        no_kv_cache_during_denoise: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.num_frames_per_block = num_frames_per_block
        self.context_noise = context_noise
        self.no_kv_cache_during_denoise = no_kv_cache_during_denoise
        self.device = device
        self.dtype = dtype
        self.tokenizer_device = tokenizer_device
        self.detokenizer_device = detokenizer_device
        self.reason1_ckpt_path = reason1_ckpt_path
        self.light_vae_tokenizer = light_vae_tokenizer
        self.light_vae_detokenizer = light_vae_detokenizer
        self.light_vae_path = light_vae_path
        self.encode_with_pixel_shuffle = encode_with_pixel_shuffle
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.view_group = view_group
        self.view_group_cp_dim = ContextParallelDim.T if enable_cross_view_attn else ContextParallelDim.HW
        self.temporal_group = temporal_group
        self.temporal_group_cp_dim = ContextParallelDim.V
        self.kv_cache_on_side_stream = kv_cache_on_side_stream
        self.use_cuda_graphs = use_cuda_graphs

        self.rng = torch.Generator(device=self.device)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            seed += rank
        self.rng.manual_seed(seed)

        self.reso = reso
        assert self.reso in ["480p", "720p", "704p"]

        # Resolve RoPE extrapolation ratio
        default_rope_ratio = 2.0 if reso == "480p" else 3.0
        self.rope_hw_extrapolation_ratio = rope_hw_extrapolation_ratio or default_rope_ratio

        def load_checkpoint_fn(net: torch.nn.Module) -> None:
            if skip_ckpt:
                return
            log.info(
                f"Loading checkpoint from {ckpt_path} (rope_hw_extrapolation_ratio={self.rope_hw_extrapolation_ratio})"
            )
            state_dict = torch.load(ckpt_path, map_location="cpu")
            net.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded checkpoint from {ckpt_path}")

        # diffusion network
        self.net = prepare_network(
            device=device,
            dtype=dtype,
            enable_hdmap_condition=True,
            encode_with_pixel_shuffle=encode_with_pixel_shuffle,
            enable_cross_view_attn=enable_cross_view_attn,
            cp_group_self_attn=view_group,
            cp_group_cross_view_attn=temporal_group,
            load_checkpoint_fn=load_checkpoint_fn,
            enable_torch_compile=compile_net,
        )

        # define scheduler
        num_train_timestep = 1000
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(num_train_timestep, training=True)

        if light_vae_tokenizer or light_vae_detokenizer:
            assert light_vae_path is not None, "light_vae_path is required when using light VAE tokenizer/detokenizer"

        def create_vae_interface(
            use_light_vae: bool,
            vae_device: torch.device,
            parallel: bool = False,
            parallel_decode: bool = False,
            process_group: ProcessGroup | None = None,
        ) -> LightWan2pt1TokenizerInterface | Wan2pt1VAEInterface:
            if use_light_vae:
                tae = None if no_tae else f"{light_vae_path}/lighttaew2_1.safetensors"
                return LightWan2pt1TokenizerInterface(
                    vae_path=f"{light_vae_path}/lightvaew2_1.safetensors",
                    tae_path=tae,
                    device=vae_device,
                    parallel=parallel,
                    process_group=process_group,
                    parallel_decode=parallel_decode,
                )
            return Wan2pt1VAEInterface(
                vae_path=f"{light_vae_path}/Wan2.1_VAE.pth",
                reset_cache_in_encoding=False,
                reset_cache_in_decoding=False,
                device=vae_device,
            )

        vae_parallel = view_group is not None and view_group.size() > 1 and not no_vae_parallel
        self.tokenizer = create_vae_interface(
            light_vae_tokenizer,
            tokenizer_device,
            parallel=vae_parallel,
            process_group=view_group,
            parallel_decode=vae_chunk_parallel,
        )
        share_tokenizer_instance = (
            light_vae_tokenizer == light_vae_detokenizer and tokenizer_device == detokenizer_device
        )
        if share_tokenizer_instance:
            self.detokenizer = self.tokenizer
        else:
            self.detokenizer = create_vae_interface(
                light_vae_detokenizer,
                detokenizer_device,
                parallel=vae_parallel,
                process_group=view_group,
                parallel_decode=vae_chunk_parallel,
            )
        self.num_latents_per_block = num_frames_per_block // self.detokenizer.temporal_compression_factor

        # define denoising steps
        if warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[
                num_train_timestep - torch.tensor(denoising_step_list, dtype=torch.long)
            ]
        else:
            self.denoising_step_list = torch.tensor(denoising_step_list, dtype=torch.long)
        self.denoising_step_list = self.denoising_step_list.to(self.device, self.dtype)
        self.denoising_sigma_list = self.scheduler.timestep_to_sigma(self.denoising_step_list)
        self.kvcache_step = torch.tensor([self.context_noise], dtype=self.dtype, device=self.device)
        self.kvcache_sigma = self.scheduler.timestep_to_sigma(self.kvcache_step)

        # Patch get_checkpoint_path so the Qwen tokenizer is loaded directly
        # from HuggingFace (public model).
        import cosmos_transfer2._src.imaginaire.utils.checkpoint_db as _ckpt_db

        _original_get_ckpt = _ckpt_db.download_checkpoint

        def _hf_get_ckpt(uri):
            if "Qwen_tokenizer" in uri and "Qwen/" in uri:
                return uri.rstrip("/").split("Qwen_tokenizer/")[-1]
            return _original_get_ckpt(uri)

        _ckpt_db.download_checkpoint = _hf_get_ckpt
        _ckpt_db.get_checkpoint_path = _hf_get_ckpt

        config = TextEncoderConfig(embedding_concat_strategy="full_concat", ckpt_path=reason1_ckpt_path)
        self.text_encoder = TextEncoder(config, device=device)

        _ckpt_db.download_checkpoint = _original_get_ckpt
        _ckpt_db.get_checkpoint_path = _original_get_ckpt

        self.net_wrapper_dict = {}

    @torch.no_grad()
    def generate_one_block(
        self,
        block_index: int,
        hdmap_condition_B_Ellipsis_D: torch.Tensor,
        cache: DiffusionModelCache,
        rope_emb: torch.Tensor,
    ) -> torch.Tensor:
        stream = torch.cuda.current_stream()
        stream.wait_event(cache.evt_kv_ready)

        num_tokens_per_block = cache.num_tokens_per_block
        current_start = block_index * num_tokens_per_block

        if block_index == 0:
            condition_video_input_mask_B_Ellipsis_D = cache.condition_video_input_mask_first_block_B_Ellipsis_D
        else:
            condition_video_input_mask_B_Ellipsis_D = cache.condition_video_input_mask_B_Ellipsis_D
        assert condition_video_input_mask_B_Ellipsis_D.dim() == 5

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
            eager_mode=not self.no_kv_cache_during_denoise,
            **network_kwargs,
        )

        stream.record_event(cache.evt_latent_ready)
        return clean_latent_B_Ellipsis_D


def main():
    from pathlib import Path

    from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

    init_environment()

    parser = argparse.ArgumentParser(description="Cosmos2 fast streaming video generation (HuggingFace)")

    # --- HuggingFace / checkpoint args ---
    parser.add_argument(
        "--hf_repo_id", type=str, default=HF_REPO_ID, help=f"HuggingFace repo ID (default: {HF_REPO_ID})"
    )
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace access token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Local directory with pre-downloaded checkpoints. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default=None,
        help="Filename of the DiT model checkpoint inside checkpoint_dir. Auto-resolved if not set.",
    )
    parser.add_argument("--output_folder", type=str, default="results", help="Output folder of results")

    # --- Data args ---
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video file (mp4)")
    parser.add_argument("--hdmap_path", type=str, default=None, help="Path to HD map video file (mp4)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for generation")

    # --- Model args ---
    parser.add_argument("--light_vae", action="store_true", help="Use light VAE")
    parser.add_argument(
        "--light_vae_tokenizer", action="store_true", default=None, help="Use light VAE tokenizer. Follows --light_vae."
    )
    parser.add_argument(
        "--light_vae_detokenizer",
        action="store_true",
        default=None,
        help="Use light VAE detokenizer. Follows --light_vae.",
    )
    parser.add_argument("--no_tae", action="store_true", help="Disable TAE decoder")
    parser.add_argument("--no_vae_parallel", action="store_true", help="Disable parallel VAE decoding")
    parser.add_argument("--no_vae_chunk_parallel", action="store_true", help="Disable parallel VAE chunk decoding")
    parser.add_argument("--compile_net", action="store_true", help="Compile the network")
    parser.add_argument("--encode_with_pixel_shuffle", action="store_true", help="Encode HDMap with pixel shuffle")
    parser.add_argument("--local_attn_size", type=int, default=-1, help="Local attention size. Default: -1 (global)")
    parser.add_argument("--sink_size", type=int, default=0, help="Sink size. Default: 0")
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA graphs for DiT blocks")
    parser.add_argument(
        "--denoising_steps", type=str, default="1000,750,500,250", help="Comma-separated denoising timesteps"
    )
    parser.add_argument("--context_noise", type=int, default=128, help="Timestep for KV cache commits. Default: 128")
    parser.add_argument(
        "--no_kv_cache_during_denoise", action="store_true", help="Disable KV cache writes during denoising"
    )
    parser.add_argument("--total_blocks", type=int, default=10, help="Total blocks to generate. Default: 10")
    parser.add_argument("--reso", choices=["480p", "720p", "704p"], default="704p", help="Resolution. Default: 704p")
    parser.add_argument("--num_frames_per_block", type=int, default=12, help="Frames per block. Default: 12")
    parser.add_argument("--skip_ckpt", action="store_true", help="Skip checkpoint loading")
    parser.add_argument(
        "--rope_hw_extrapolation_ratio", type=float, default=None, help="RoPE extrapolation ratio. Auto-detected."
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of the video/model. Default: 30")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel world size")
    parser.add_argument("--export_fps", type=int, default=10, help="Export FPS. Default: 10")
    parser.add_argument("--kv_cache_on_side_stream", action="store_true", help="Use side stream for KV cache update")

    args = parser.parse_args()

    context_parallel_size = args.context_parallel_size

    # Distributed is already initialized by init_environment() if RANK is set.
    if context_parallel_size > 1:
        if not dist.is_initialized():
            raise RuntimeError("Context parallel requires torch.distributed.run --nproc_per_node=N")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if context_parallel_size != world_size:
            raise ValueError(f"context_parallel_size ({context_parallel_size}) must match world_size ({world_size})")

        T = args.num_frames_per_block // 4
        groups = create_hierarchical_cp_groups(world_size, rank, 1, T)
        view_group = groups.THW_group if groups.THW_group.size() > 1 else None
        temporal_group = groups.V_group if groups.V_group.size() > 1 else None
    else:
        rank = 0
        view_group, temporal_group = None, None
    log.info(f"view_group: {view_group}, temporal_group: {temporal_group}")

    init_output_dir(Path(args.output_folder))

    # --- Resolve checkpoint directory ---
    if args.checkpoint_dir is not None:
        ckpt_dir = args.checkpoint_dir
    else:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        ckpt_dir = download_hf_checkpoints(repo_id=args.hf_repo_id, token=hf_token)
    log.info(f"Checkpoint directory: {ckpt_dir}")

    # Resolve checkpoint filename
    if args.ckpt_name is not None:
        ckpt_filename = args.ckpt_name
        rope_hw_extrapolation_ratio = args.rope_hw_extrapolation_ratio
    else:
        ckpt_filename, default_rope_ratio = resolve_hf_checkpoint(args.reso, args.encode_with_pixel_shuffle)
        rope_hw_extrapolation_ratio = args.rope_hw_extrapolation_ratio or default_rope_ratio
        log.info(f"Auto-resolved checkpoint: {ckpt_filename} (rope_ratio={rope_hw_extrapolation_ratio})")

    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
    reason1_ckpt_path = os.path.join(ckpt_dir, "reason1")
    light_vae_path = os.path.join(ckpt_dir, "Autoencoders")

    # Parse denoising steps
    denoising_step_list = [int(x.strip()) for x in args.denoising_steps.split(",")]
    log.info(f"Denoising steps: {denoising_step_list}")

    log.info(f"Local attention size: {args.local_attn_size}")
    log.info(f"Sink size: {args.sink_size}")
    light_vae_tokenizer = args.light_vae if args.light_vae_tokenizer is None else args.light_vae_tokenizer
    light_vae_detokenizer = args.light_vae if args.light_vae_detokenizer is None else args.light_vae_detokenizer

    # Set device
    if context_parallel_size > 1:
        local_rank = dist.get_rank() % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0")
    dtype = torch.bfloat16

    if args.reso == "480p":
        res_H, res_W = 480, 832
    elif args.reso == "720p":
        res_H, res_W = 720, 1280
    elif args.reso == "704p":
        res_H, res_W = 704, 1280
    else:
        raise ValueError(f"Invalid resolution: {args.reso}")

    # --- Resolve data paths ---
    video_path = args.video_path
    hdmap_path = args.hdmap_path
    if video_path is None or hdmap_path is None:
        reso_bucket = "480p" if args.reso == "480p" else "720p"
        example_video, example_hdmap = _HF_EXAMPLES[reso_bucket]
        examples_dir = os.path.join(ckpt_dir, "examples")
        if video_path is None:
            video_path = os.path.join(examples_dir, example_video)
        if hdmap_path is None:
            hdmap_path = os.path.join(examples_dir, example_hdmap)
        log.info(f"Using example data: video={video_path}, hdmap={hdmap_path}")

    # --- Load data ---
    view_indices = None
    first_frame, prompt, hdmap_video = parse_single_view_data(
        video_path=video_path,
        hdmap_path=hdmap_path,
        prompt=args.prompt,
        res_H=res_H,
        res_W=res_W,
        fps=args.fps,
    )
    log.info(f"first_frame shape: {first_frame.shape}")

    # Preprocess
    hdmap_condition = preprocess_input_hdmap_video(hdmap_video, dtype=dtype, device=device)
    do_view_split_cp = True
    log.info(f"hdmap_condition shape: {hdmap_condition.shape}")

    diffusion_model = DiffusionModel(
        ckpt_path=ckpt_path,
        reason1_ckpt_path=reason1_ckpt_path,
        reso=args.reso,
        light_vae_tokenizer=light_vae_tokenizer,
        light_vae_detokenizer=light_vae_detokenizer,
        light_vae_path=light_vae_path,
        compile_net=args.compile_net,
        encode_with_pixel_shuffle=args.encode_with_pixel_shuffle,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        context_noise=args.context_noise,
        denoising_step_list=denoising_step_list,
        num_frames_per_block=args.num_frames_per_block,
        enable_cross_view_attn=False,
        skip_ckpt=args.skip_ckpt,
        use_cuda_graphs=args.use_cuda_graphs,
        device=device,
        tokenizer_device=device,
        detokenizer_device=device,
        dtype=dtype,
        view_group=view_group,
        temporal_group=temporal_group,
        kv_cache_on_side_stream=args.kv_cache_on_side_stream,
        no_tae=args.no_tae,
        no_vae_parallel=args.no_vae_parallel,
        vae_chunk_parallel=not args.no_vae_chunk_parallel,
        rope_hw_extrapolation_ratio=rope_hw_extrapolation_ratio,
        no_kv_cache_during_denoise=args.no_kv_cache_during_denoise,
    )

    total_blocks = args.total_blocks

    def _run(model: DiffusionModel, total_blocks: int, cache: DiffusionModelCache | None = None):
        if cache is None:
            cache = model.precompute_and_cache(
                batch_size=1,
                text_prompt_or_embeddings=prompt,
                image_array_or_tensor=first_frame,
                view_indices=view_indices,
                seed=42,
            )

        events = ProfileEvents.create(total_blocks)
        generated_video_blocks = []
        finalization_state = None

        for block_index in range(total_blocks):
            current_event = events[block_index]
            current_event.tic_block.record()

            if finalization_state is not None:
                model.finalize_block_generation(finalization_state)

            current_start, current_end = model.get_current_video_frame_range(block_index)
            assert current_end <= hdmap_condition.shape[-3], (
                f"current_end: {current_end} is out of bounds of HDMap data with frames: {hdmap_condition.shape[-3]}"
            )
            current_hdmap = hdmap_condition[..., current_start:current_end, :, :]

            video, finalization_state = model.streaming_inference_one_block(
                block_index, cache, current_hdmap, current_event, do_view_split_cp=do_view_split_cp
            )

            generated_video_blocks.append(video)
            current_event.toc_block_after_upsample.record()

        if finalization_state is not None:
            model.finalize_block_generation(finalization_state)

        return generated_video_blocks, events

    # warmup
    if rank == 0:
        log.info("Warming up...")
    with NVTXRangeDecorator("warmup"):
        _ = _run(diffusion_model, total_blocks=2)

    cache = diffusion_model.precompute_and_cache(
        batch_size=1,
        text_prompt_or_embeddings=prompt,
        image_array_or_tensor=first_frame,
        view_indices=view_indices,
        seed=42,
    )

    # actual generation
    if rank == 0:
        log.info("Actual generation and profiling...")
    s_event = torch.cuda.Event(enable_timing=True)
    e_event = torch.cuda.Event(enable_timing=True)
    if context_parallel_size > 1:
        dist.barrier()
    torch.cuda.synchronize()

    s_event.record()
    with NVTXRangeDecorator("main"):
        generated_video_blocks, events = _run(diffusion_model, total_blocks=total_blocks, cache=cache)
    if context_parallel_size > 1:
        dist.barrier()
    e_event.record()

    torch.cuda.synchronize()
    total_time = s_event.elapsed_time(e_event) / 1000.0
    if rank == 0:
        ProfileEvents.finalize(events, skip_first_n=3)
    log.info(f"E2E latency: {total_time / total_blocks} seconds per block")

    full_video = torch.cat(generated_video_blocks, dim=-3)
    log.info(f"[Rank {rank}] Full video shape: {full_video.shape}")

    # Save video
    if rank == 0:
        total_frames = full_video.shape[-3]

        full_hdmap_condition = preprocess_input_hdmap_video(hdmap_video, dtype=dtype, device=device)
        full_hdmap_condition = (full_hdmap_condition[..., :total_frames, :, :] + 1.0) / 2.0

        canvas = torch.cat([full_hdmap_condition, full_video], dim=-2)
        canvas = rearrange(canvas, "b v c t h w -> b c t h (v w)")
        output_name = f"{args.output_folder}/cosmos2_{res_H}_{total_blocks}"
        log.info(f"Saving video to {output_name}.mp4...")
        save_img_or_video(canvas[0], output_name, fps=args.export_fps)

        torch.cuda.synchronize()
        log.info(f"Video saved successfully! Shape: {full_video.shape}")

    # Cleanup
    import gc

    del diffusion_model
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if context_parallel_size > 1:
        dist.barrier()

    cleanup_environment()

    if rank == 0:
        log.info("Done!")


if __name__ == "__main__":
    main()
