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
from typing import Union

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.inference.utils import color_message, get_prompt_from_path
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
)
from cosmos_transfer2._src.transfer2_multiview.datasets.local_dataset import (
    LocalMultiviewAugmentorConfig,
    LocalMultiviewDatasetBuilder,
)
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2.config import MODEL_CHECKPOINTS, ModelKey, MultiviewParams

_DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(variant="drive")]
NUM_DATALOADER_WORKERS = 8


class MultiviewInference:
    def __init__(
        self,
        num_gpus=8,
        experiment="",
        ckpt_path="",
        disable_guardrails=False,
        offload_guardrail_models=True,
    ):
        if not ckpt_path:
            ckpt_path = _DEFAULT_CHECKPOINT.path
        if not experiment:
            experiment = _DEFAULT_CHECKPOINT.experiment

        log.info(f"Using {experiment=} and {ckpt_path=}")

        # Enable deterministic inference
        os.environ["NVTE_FUSED_ATTN"] = "0"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.pipe = ControlVideo2WorldInference(
            experiment_name=experiment,
            ckpt_path=ckpt_path,
            context_parallel_size=num_gpus,
        )

        self.rank0 = True
        self.guardrail_enabled = not disable_guardrails

        if num_gpus > 1:
            self.rank0 = torch.distributed.get_rank() == 0

        if self.rank0 and self.guardrail_enabled:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(offload_model_to_cpu=True)
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(offload_model_to_cpu=True)
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

    def infer(self, params: Union[MultiviewParams, dict]):
        if isinstance(params, dict):
            p = MultiviewParams.create(params)
        else:
            p = params
        log.info(f"params: {p}")
        driving_dataloader_config = MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[
            self.pipe.config.model.config.resolution
        ]
        driving_dataloader_config.n_views = p.n_views

        prompt, _ = get_prompt_from_path(p.prompt_path, p.prompt)

        # run text guardrail on the prompt
        if self.rank0:
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on prompt...")
                if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                    log.critical("Guardrail blocked control2world generation. Prompt: {prompt}")
                    exit(1)
                else:
                    log.success("Passed guardrail on prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        # setup the control and input videos dict
        input_video_file_dict = {}
        control_video_file_dict = {}
        for key, value in p.input_and_control_paths.items():
            if "_input" in key:
                input_video_file_dict[key.removesuffix("_input")] = value
            elif "_control" in key:
                control_video_file_dict[key.removesuffix("_control")] = value

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
            batch["ai_caption"] = [prompt]
            batch["control_weight"] = p.control_weight
            batch["num_conditional_frames"] = p.num_conditional_frames
            log.info(f"------ Generating video ------")
            video = self.pipe.generate_from_batch(batch, guidance=p.guidance, seed=p.seed)
            if self.rank0:
                video = (1.0 + video[0]) / 2
                # run video guardrail on the video
                if self.rank0 and self.video_guardrail_runner is not None:
                    log.info("Running guardrail check on video...")
                    frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        log.critical("Guardrail blocked video2world generation.")
                        exit(1)
                    else:
                        log.success("Passed guardrail on generated video")

                    # Convert processed frames back to tensor format
                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                    video = processed_video.to(video.device, dtype=video.dtype)
                else:
                    log.warning("Guardrail checks on video are disabled")

                save_img_or_video(video, f"{p.output_dir}/output", fps=p.fps)
                log.info(color_message(f"Generated video saved to {p.output_dir}/output.mp4\n", "green"))
