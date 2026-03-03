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

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
from typing import Mapping

import decord
import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    ConditionLocation,
    ConditionLocationList,
)
from cosmos_transfer2._src.predict2_multiview.datasets.local import LocalMultiViewDataset
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import AugmentationConfig, collate_fn
from cosmos_transfer2._src.transfer2.inference.utils import color_message
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2.multiview_config import (
    MULTIVIEW_CAMERA_KEYS,
    MultiviewInferenceArguments,
    MultiviewSetupArguments,
)

RESOLUTIONS: Mapping = {
    "720p": (720, 1280),
}


DEFAULT_CAMERA_KEYS = MULTIVIEW_CAMERA_KEYS
DEFAULT_CAMERA_VIEW_MAPPING = {camera_key: idx for idx, camera_key in enumerate(DEFAULT_CAMERA_KEYS)}
DEFAULT_CAMERA_PREFIX_MAPPING = {
    "front_wide": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "cross_right": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "rear_right": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "rear": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "rear_left": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "cross_left": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "front_tele": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}


def setup_config(
    resolution_hw: tuple[int, int],
    num_video_frames_per_view: int,
    fps_downsample_factor: int,
    camera_keys: tuple[str, ...] | None = None,
    single_caption_camera_name: str | None = "front_wide",
) -> AugmentationConfig:
    camera_keys = camera_keys or DEFAULT_CAMERA_KEYS
    if not camera_keys:
        raise ValueError("At least one camera key must be provided for multiview inference.")
    invalid_keys = set(camera_keys) - set(DEFAULT_CAMERA_KEYS)
    if invalid_keys:
        raise ValueError(f"Unknown camera keys provided: {', '.join(sorted(invalid_keys))}")
    if single_caption_camera_name not in camera_keys:
        single_caption_camera_name = camera_keys[0]

    kwargs = dict(
        resolution_hw=resolution_hw,
        fps_downsample_factor=fps_downsample_factor,
        num_video_frames=num_video_frames_per_view,
        camera_keys=camera_keys,
        camera_view_mapping={key: DEFAULT_CAMERA_VIEW_MAPPING[key] for key in camera_keys},
        camera_caption_key_mapping={k: f"caption_{k}" for k in camera_keys},
        camera_video_key_mapping={k: f"video_{k}" for k in camera_keys},
        camera_control_key_mapping={k: f"control_{k}" for k in camera_keys},
        add_view_prefix_to_caption=False,
        camera_prefix_mapping={k: DEFAULT_CAMERA_PREFIX_MAPPING[k] for k in camera_keys},
        single_caption_camera_name=single_caption_camera_name,
    )
    return AugmentationConfig(**kwargs)


class MultiviewInference:
    def __init__(self, args: MultiviewSetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")

        # Disable gradient calculations for inference
        torch.enable_grad(False)

        self.setup_args = args

        self.pipe = ControlVideo2WorldInference(
            # pyrefly: ignore  # bad-argument-type
            experiment_name=args.experiment,
            # pyrefly: ignore  # bad-argument-type
            ckpt_path=args.checkpoint_path,
            # pyrefly: ignore  # bad-argument-type
            context_parallel_size=args.context_parallel_size,
        )

        self.rank0 = True

        # pyrefly: ignore  # unsupported-operation
        if args.context_parallel_size > 1:
            self.rank0 = torch.distributed.get_rank() == 0

        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.pipe.config, args.output_dir / "config.yaml")

        if self.rank0 and args.enable_guardrails:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            # pyrefly: ignore  # bad-assignment
            self.text_guardrail_runner = None
            # pyrefly: ignore  # bad-assignment
            self.video_guardrail_runner = None

    def generate(self, samples: list[MultiviewInferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir)
            if output_path is not None:
                output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: MultiviewInferenceArguments, output_dir: Path) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

        # setup the control and input videos dict
        input_video_file_dict = {}
        control_video_file_dict = {}
        fps = set()
        min_control_frames = float("inf")
        for key, value in sample.input_and_control_paths.items():  # pyrefly: ignore # bad-assignment
            if "_input" in key and value is not None:
                input_video_file_dict[key.removesuffix("_input")] = value
                assert value  # make mypy happy
                fps.add(decord.VideoReader(value.as_posix()).get_avg_fps())
            elif "_control" in key:
                control_video_file_dict[key.removesuffix("_control")] = value
                assert value  # make mypy happy
                vr = decord.VideoReader(value.as_posix())
                fps.add(vr.get_avg_fps())
                min_control_frames = min(min_control_frames, len(vr))

        if len(fps) != 1:
            raise ValueError(f"Control and video files have inconsistent FPS: {fps}")
        fps = fps.pop()
        desired_fps = sample.fps
        if fps % desired_fps != 0:
            raise ValueError(f"Video file fps {fps} is not evenly divisible by desired FPS {desired_fps}")
        fps_downsample_factor = int(fps / desired_fps)
        log.info(
            f"Files have FPS of {fps}, and desired FPS is {desired_fps}. Downsampling by factor of {fps_downsample_factor}"
        )

        # Calculate number of video frames to load
        assert self.pipe.config.model.config.state_t >= 1
        chunk_size = self.pipe.model.tokenizer.get_pixel_num_frames(  # pyrefly: ignore # missing-attribute
            self.pipe.config.model.config.state_t
        )

        # Check if have enough frames after downsampling for even one chunk
        available_control_frames_after_downsample = int(min_control_frames) // fps_downsample_factor
        if available_control_frames_after_downsample < chunk_size:
            raise ValueError(
                f"Not enough frames in control videos. Need at least {chunk_size} frames after "
                f"downsampling (fps_downsample_factor={fps_downsample_factor}), but only have "
                f"{available_control_frames_after_downsample} frames (from {int(min_control_frames)} source frames). "
            )

        # Clamp num_chunks in autoregressive mode to what's available. This makes it easy for a user to set a high
        # num_chunks value and let the control length limit the actual number of chunks.
        effective_num_chunks = sample.num_chunks
        if sample.enable_autoregressive:
            max_chunks = 1 + (available_control_frames_after_downsample - chunk_size) // (
                chunk_size - sample.chunk_overlap
            )
            max_chunks = max(1, max_chunks)  # At least 1 chunk

            if sample.num_chunks > max_chunks:
                log.warning(
                    f"Requested num_chunks={sample.num_chunks} requires more frames than available. "
                    f"Clamping to max_chunks={max_chunks} (available_frames={available_control_frames_after_downsample}, "
                    f"chunk_size={chunk_size}, overlap={sample.chunk_overlap})"
                )
                effective_num_chunks = max_chunks

        num_video_frames_per_view = chunk_size
        if sample.enable_autoregressive:
            num_video_frames_per_view += (num_video_frames_per_view - sample.chunk_overlap) * (effective_num_chunks - 1)

        camera_keys = sample.active_camera_keys
        primary_caption_view = "front_wide" if "front_wide" in camera_keys else camera_keys[0]
        augmentation_config = setup_config(
            resolution_hw=RESOLUTIONS[self.pipe.config.model.config.resolution],
            num_video_frames_per_view=num_video_frames_per_view,
            fps_downsample_factor=fps_downsample_factor,
            camera_keys=camera_keys,
            single_caption_camera_name=primary_caption_view,
        )
        if SMOKE:
            log.warning(f"Reducing the number of views to 1 for smoke test. Generated quality will be sub-optimal.")
            augmentation_config.camera_keys = augmentation_config.camera_keys[:1]
        log.info(f"Generating local multiview dataset with following config: {augmentation_config}")
        if sample.enable_autoregressive:
            self.pipe.config.model.config.condition_locations = ConditionLocationList(
                [ConditionLocation.FIRST_RANDOM_N]
            )

        # run text guardrail on the prompt
        if self.rank0:
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on prompt...")
                assert sample.prompt is not None
                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                    message = f"Guardrail blocked generation. Prompt: {sample.prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        # if number_of_condtional_frames=0, input videos are optional use control videos instead as mock input
        if sample.num_conditional_frames == 0:
            input_video_file_dict = control_video_file_dict

        assert sample.prompt is not None  # make mypy happy
        dataset = LocalMultiViewDataset(
            video_file_dicts=[input_video_file_dict],
            prompts=[sample.prompt],
            control_file_dicts=[control_video_file_dict],
            augmentation_config=augmentation_config,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )

        if len(dataloader) == 0:
            raise ValueError("No input data found")

        for _, batch in enumerate(dataloader):
            batch["control_weight"] = sample.control_weight
            if sample.enable_autoregressive:
                num_conditional_frames_per_view = [
                    getattr(sample, k).num_conditional_frames_per_view for k in augmentation_config.camera_keys
                ]
                if all(frames == 0 for frames in num_conditional_frames_per_view):
                    log.info(f"Using single conditional frames value: {sample.num_conditional_frames}")
                    num_conditional_frames = sample.num_conditional_frames
                else:
                    log.info(f"Using per-view conditional frames: {num_conditional_frames_per_view}")
                    num_conditional_frames = num_conditional_frames_per_view
            else:
                num_conditional_frames = sample.num_conditional_frames

            if sample.enable_autoregressive:
                log.info(f"------ Generating video with autoregressive mode ------")
                video, control = self.pipe.generate_autoregressive_from_batch(
                    batch,
                    n_views=len(augmentation_config.camera_keys),
                    chunk_overlap=sample.chunk_overlap,
                    chunk_size=chunk_size,
                    guidance=sample.guidance,
                    seed=sample.seed,
                    num_conditional_frames=num_conditional_frames,
                    num_steps=sample.num_steps,
                    use_negative_prompt=True,
                )
            else:
                log.info(f"------ Generating video ------")
                if isinstance(num_conditional_frames, list):
                    num_conditional_latent_frames = [
                        self.pipe.model.tokenizer.get_latent_num_frames(  # pyrefly: ignore # missing-attribute
                            pixel_frames
                        )
                        for pixel_frames in num_conditional_frames
                    ]
                else:
                    num_conditional_latent_frames = (
                        self.pipe.model.tokenizer.get_latent_num_frames(  # pyrefly: ignore # missing-attribute
                            num_conditional_frames
                        )
                    )
                batch["num_conditional_frames"] = num_conditional_latent_frames
                video = self.pipe.generate_from_batch(
                    batch,
                    guidance=sample.guidance,
                    seed=sample.seed,
                    num_steps=sample.num_steps,
                    use_negative_prompt=True,
                )
                control = None

            if self.rank0:
                if not sample.enable_autoregressive:
                    video = video[0]

                # Run video guardrail on the normalized video
                if self.video_guardrail_runner is not None:
                    log.info("Running guardrail check on video...")
                    frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        message = "Guardrail blocked generation. Video"
                        log.critical(message)
                        if self.setup_args.keep_going:
                            return None
                        else:
                            raise Exception(message)
                    else:
                        log.success("Passed guardrail on generated video")

                    # Convert processed frames back to tensor format
                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0

                    video = processed_video.to(video.device, dtype=video.dtype)
                else:
                    log.warning("Guardrail checks on video are disabled")

                def save_combined_video():
                    save_img_or_video(video, str(output_path), fps=sample.fps, quality=8)
                    log.success(f"Generated video saved to {output_path}.mp4")

                if sample.save_combined_views:
                    save_combined_video()
                    return f"{output_path}.mp4"

                total_frames = video.shape[1]
                n_views = len(augmentation_config.camera_keys)

                # Calculate frames per view from actual video tensor shape
                # Video shape is C (V T) H W where V is number of views and T is frames per view
                if total_frames % n_views != 0:
                    raise ValueError(f"Video frames ({total_frames}) not divisible by number of views ({n_views}).")

                frames_per_view = total_frames // n_views
                if frames_per_view <= 0:
                    raise ValueError("Cannot split views because frames_per_view is not positive.")

                inferred_views = n_views

                # Split the concatenated video into per-view tensors
                camera_keys = list(augmentation_config.camera_keys)
                view_tensors = []
                for view_index in range(inferred_views):
                    start = view_index * frames_per_view
                    end = start + frames_per_view
                    view_tensor = video[:, start:end]
                    view_name = camera_keys[view_index] if view_index < len(camera_keys) else f"view_{view_index}"
                    view_tensors.append((view_name, view_tensor))

                # Save individual view videos
                output_messages = []
                for view_name, view_tensor in view_tensors:
                    view_output_path = f"{output_path}_{view_name}"
                    save_img_or_video(view_tensor, view_output_path, fps=sample.fps, quality=8)
                    output_messages.append(f"{view_output_path}.mp4")

                # Save grid video
                grid_rows, grid_cols = 3, 3
                c, t, h, w = view_tensors[0][1].shape
                grid_tensor = torch.zeros((c, t, grid_rows * h, grid_cols * w), dtype=video.dtype, device=video.device)

                num_views_in_grid = min(len(view_tensors), grid_rows * grid_cols)
                for idx in range(num_views_in_grid):
                    row, col = idx // grid_cols, idx % grid_cols
                    grid_tensor[:, :, row * h : (row + 1) * h, col * w : (col + 1) * w] = view_tensors[idx][1]

                grid_output_path = f"{output_path}_grid"
                save_img_or_video(grid_tensor, grid_output_path, fps=sample.fps, quality=8)
                output_messages.append(
                    f"{grid_output_path}.mp4 ({num_views_in_grid} views in {grid_rows}x{grid_cols} grid)"
                )

                # Log all outputs at once
                if output_messages:
                    log.success(
                        color_message(f"Generated videos saved to:\n" + "\n".join(output_messages) + "\n", "green")
                    )

        return f"{output_path}.mp4"
