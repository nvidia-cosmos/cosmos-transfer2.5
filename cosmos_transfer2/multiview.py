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
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import AugmentationConfig
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2.multiview_config import MultiviewInferenceArguments, MultiviewSetupArguments, MULTIVIEW_CAMERA_KEYS

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
        add_view_prefix_to_caption=True,
        camera_prefix_mapping={k: DEFAULT_CAMERA_PREFIX_MAPPING[k] for k in camera_keys},
        single_caption_camera_name=single_caption_camera_name,
    )
    return AugmentationConfig(**kwargs)


class MultiviewInference:
    def __init__(self, args: MultiviewSetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")

        # Enable deterministic inference
        os.environ["NVTE_FUSED_ATTN"] = "0"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.enable_grad(False)  # Disable gradient calculations for inference

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
        for key, value in sample.input_and_control_paths.items():
            if "_input" in key and value is not None:
                input_video_file_dict[key.removesuffix("_input")] = value
                assert value  # make mypy happy
                fps.add(decord.VideoReader(value.as_posix()).get_avg_fps())
            elif "_control" in key:
                control_video_file_dict[key.removesuffix("_control")] = value
                assert value  # make mypy happy
                fps.add(decord.VideoReader(value.as_posix()).get_avg_fps())

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
        chunk_size = (
            1 + (self.pipe.config.model.config.state_t - 1) * 4
        )  # tokenizer downsamples by 4x in temporal dimension
        num_video_frames_per_view = chunk_size
        if sample.enable_autoregressive:
            num_video_frames_per_view += (num_video_frames_per_view - sample.chunk_overlap) * (sample.num_chunks - 1)

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
                    batch["num_conditional_frames"] = sample.num_conditional_frames
                else:
                    log.info(f"Using per-view conditional frames: {num_conditional_frames_per_view}")
                    batch["num_conditional_frames"] = num_conditional_frames_per_view
            else:
                batch["num_conditional_frames"] = sample.num_conditional_frames

            if sample.enable_autoregressive:
                log.info(f"------ Generating video with autoregressive mode ------")
                video, control = self.pipe.generate_autoregressive_from_batch(
                    batch,
                    n_views=len(augmentation_config.camera_keys),
                    chunk_overlap=sample.chunk_overlap,
                    chunk_size=chunk_size,
                    guidance=sample.guidance,
                    seed=sample.seed,
                    num_conditional_frames=batch["num_conditional_frames"],
                    num_steps=sample.num_steps,
                )
            else:
                log.info(f"------ Generating video ------")
                video = self.pipe.generate_from_batch(
                    batch, guidance=sample.guidance, seed=sample.seed, num_steps=sample.num_steps
                )
                control = None

            if self.rank0:
                if not sample.enable_autoregressive:
                    video = video[0]
                # Normalize video from [-1, 1] to [0, 1]
                video = video.clamp(-1, 1) / 2 + 0.5

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

                save_img_or_video(video, str(output_path), fps=sample.fps, quality=8)
                log.success(f"Generated video saved to {output_path}.mp4")

        return f"{output_path}.mp4"
