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

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
)
from cosmos_transfer2._src.transfer2_multiview.datasets.local_dataset import (
    LocalMultiviewAugmentorConfig,
    LocalMultiviewDatasetBuilder,
)
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2.multiview_config import MultiviewInferenceArguments, MultiviewSetupArguments

NUM_DATALOADER_WORKERS = 8


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

        driving_dataloader_config = MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[
            self.pipe.config.model.config.resolution
        ]
        driving_dataloader_config.n_views = sample.n_views

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

        # setup the control and input videos dict
        input_video_file_dict = {}
        control_video_file_dict = {}
        for key, value in sample.input_and_control_paths.items():
            if "_input" in key and value is not None:
                input_video_file_dict[key.removesuffix("_input")] = value
            elif "_control" in key:
                control_video_file_dict[key.removesuffix("_control")] = value

        # if number_of_condtional_frames=0, input videos are optional use control videos instead as mock input
        if sample.num_conditional_frames == 0:
            input_video_file_dict = control_video_file_dict
        dataset = LocalMultiviewDatasetBuilder(
            input_video_file_dict=input_video_file_dict, control_video_file_dict=control_video_file_dict
        ).build_dataset(
            LocalMultiviewAugmentorConfig(
                resolution=self.pipe.config.model.config.resolution,
                driving_dataloader_config=driving_dataloader_config,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_DATALOADER_WORKERS,
        )

        if len(dataloader) == 0:
            raise ValueError("No input data found")

        for _, batch in enumerate(dataloader):
            batch["ai_caption"] = [sample.prompt]
            batch["control_weight"] = sample.control_weight
            batch["num_conditional_frames"] = sample.num_conditional_frames
            log.info(f"------ Generating video ------")
            video = self.pipe.generate_from_batch(batch, guidance=sample.guidance, seed=sample.seed)
            if self.rank0:
                video = (1.0 + video[0]) / 2
                # run video guardrail on the video
                if self.rank0 and self.video_guardrail_runner is not None:
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

                save_img_or_video(video, str(output_path), fps=sample.fps)
                log.success(f"Generated video saved to {output_path}.mp4")
        return f"{output_path}.mp4"
