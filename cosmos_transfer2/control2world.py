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

import json
import os
from typing import Union

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import color_message, get_prompt_from_path
from cosmos_transfer2.config import MODEL_CHECKPOINTS, Control2WorldParams, ModelKey


# this is a simplified version of cosmos_transfer2/_src/transfer2/inference/inference_vid2vid_control_batch.py
class Control2WorldInference:
    def __init__(
        self,
        num_gpus: int,
        hint_key: list[str] = ["edge"],
        disable_guardrails: bool = False,
        offload_guardrail_models: bool = True,
    ) -> None:
        self.hint_keys = hint_key
        log.info(f"Using {self.hint_keys=}")

        checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=self.hint_keys[0])]  # type: ignore
        if len(self.hint_keys) > 1:
            # experiment name for multi-control model as per research script
            experiment_name = "multibranch_720p_t24_spaced_layer4_cr1_sdev2_hqv1p1_20250715_basev2_25k_inference"
        else:
            experiment_name = checkpoint.experiment

        registered_exp_name = EXPERIMENTS[experiment_name].registered_exp_name
        exp_override_opts = EXPERIMENTS[experiment_name].command_args
        log.critical(f"Using {experiment_name=} {registered_exp_name=} {exp_override_opts=}")

        # multi-control model checkpoint paths for single control model checkpoint paths are None
        checkpoint_paths = (
            [
                checkpoint.path
                for checkpoint in [MODEL_CHECKPOINTS[ModelKey(variant=hint_key)] for hint_key in self.hint_keys]
            ]
            if len(self.hint_keys) > 1
            else checkpoint.path
        )  # type: ignore
        log.critical(f"Loading model for {self.hint_keys=} using {checkpoint_paths=}")

        self.device_rank = 0
        self.num_gpus = num_gpus
        self.guardrail_enabled = not disable_guardrails

        process_group = None
        if num_gpus > 1:
            from megatron.core import parallel_state

            from cosmos_transfer2._src.imaginaire.utils import distributed

            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
            process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(process_group)

        if self.guardrail_enabled and self.device_rank == 0:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(offload_model_to_cpu=True)
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(offload_model_to_cpu=True)
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

        # Initialize the inference class
        self.inference_pipeline = ControlVideo2WorldInference(
            registered_exp_name=registered_exp_name,
            checkpoint_paths=checkpoint_paths,
            s3_credential_path="",
            exp_override_opts=exp_override_opts,
            process_group=process_group,
        )

    def infer(self, params: Union[Control2WorldParams, dict]) -> None:
        if isinstance(params, dict):
            p = Control2WorldParams.create(params)
        else:
            p = params
        torch.manual_seed(p.seed)

        prompt, neg_prompt = get_prompt_from_path(p.prompt_path, p.prompt)
        if neg_prompt is None:
            neg_prompt = p.negative_prompt
        else:
            log.info(f"Overriding default negative prompt with: {neg_prompt}")

        # run text guardrail on the prompt
        if self.device_rank == 0:
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on prompt...")
                if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                    log.critical("Guardrail blocked control2world generation. Prompt: {prompt}")
                    exit(1)
                else:
                    log.success("Passed guardrail on prompt")
                if not guardrail_presets.run_text_guardrail(neg_prompt, self.text_guardrail_runner):
                    log.critical("Guardrail blocked control2world generation. Negative prompt: {neg_prompt}")
                    exit(1)
                else:
                    log.success("Passed guardrail on negative prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        input_control_video_paths = p.control_modalities
        log.info(f"input_control_video_paths: {json.dumps(input_control_video_paths, indent=4)}")

        sigma_max = None if p.sigma_max is None else float(p.sigma_max)
        # control_weight is a string because of multi-control. for all hint_keys, seperate the control_weight by comma
        if len(self.hint_keys) > 1:
            control_weight_str = str(p.multicontrol_weight)
        else:
            control_weight_str = str(p.control_weight)

        # Run model inference
        output_video, fps, _ = self.inference_pipeline.generate_img2world(
            video_path=p.video_path,
            prompt=prompt,
            negative_prompt=neg_prompt,
            image_context_path=p.image_context_path,
            guidance=p.guidance,
            seed=p.seed,
            resolution=p.resolution,
            control_weight=control_weight_str,
            sigma_max=sigma_max,
            hint_key=self.hint_keys,
            input_control_video_paths=input_control_video_paths,
            show_control_condition=p.show_control_condition,
            show_input=p.show_input,
            keep_input_resolution=not p.not_keep_input_resolution,
        )

        # Save video
        if self.device_rank == 0:
            os.makedirs(os.path.dirname(p.output_dir), exist_ok=True)
            save_path = os.path.join(p.output_dir, "output")
            output_video = (1.0 + output_video[0]) / 2

            # run video guardrail on the video
            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    log.critical("Guardrail blocked video2world generation.")
                    exit(1)
                else:
                    log.success("Passed guardrail on generated video")

                # Convert processed frames back to tensor format
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                output_video = processed_video.to(output_video.device, dtype=output_video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")

            # Remove batch dimension and normalize to [0, 1] range
            save_img_or_video(output_video, save_path, fps=fps)
            # save prompt
            prompt_save_path = f"{save_path}.txt"
            with open(prompt_save_path, "w") as f:
                f.write(prompt)
            message = color_message(message=f"Generated video saved to {save_path}.mp4\n", color="green")
            log.info(str(message))

        torch.cuda.empty_cache()

    def __del__(self):
        # If using distributed, make sure to clean up properly
        if self.num_gpus > 1:
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
            import torch.distributed as dist

            dist.destroy_process_group()
