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

import time
from pathlib import Path

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import compile_tokenizer_if_enabled
from cosmos_transfer2.config import (
    MODEL_CHECKPOINTS,
    InferenceArguments,
    ModelKey,
    SetupArguments,
    is_rank0,
    path_to_str,
)


class Control2WorldInference:
    def __init__(
        self,
        args: SetupArguments,
        batch_hint_keys: list[str],
    ) -> None:
        log.debug(f"{args.__class__.__name__}({args})({batch_hint_keys})")
        self.setup_args = args
        self.batch_hint_keys = batch_hint_keys
        if len(self.batch_hint_keys) == 1:
            # pyrefly: ignore  # bad-argument-type
            checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=self.batch_hint_keys[0])]
            self.checkpoint_list = [checkpoint.path]
            self.experiment = checkpoint.experiment
        else:
            # pyrefly: ignore  # bad-argument-type
            self.checkpoint_list = [MODEL_CHECKPOINTS[ModelKey(variant=key)].path for key in self.batch_hint_keys]
            self.experiment = "multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference"

        log.debug(f"Loading keys for batch hints {self.batch_hint_keys=}")
        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.device_rank = 0

        process_group = None
        # pyrefly: ignore  # unsupported-operation
        if args.context_parallel_size > 1:
            from megatron.core import parallel_state

            distributed.init()

            # pyrefly: ignore  # bad-argument-type
            parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
            process_group = parallel_state.get_context_parallel_group()

        if args.enable_guardrails and self.device_rank == 0:
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

        # Initialize the inference class
        self.inference_pipeline = ControlVideo2WorldInference(
            registered_exp_name=EXPERIMENTS[self.experiment].registered_exp_name,
            checkpoint_paths=self.checkpoint_list,
            s3_credential_path="",
            exp_override_opts=EXPERIMENTS[self.experiment].command_args,
            process_group=process_group,
            use_cp_wan=args.enable_parallel_tokenizer,
            wan_cp_grid=args.parallel_tokenizer_grid,
        )

        compile_tokenizer_if_enabled(self.inference_pipeline, args.compile_tokenizer.value)

        if self.device_rank == 0:
            log.info(f"Found {len(self.batch_hint_keys)} hint keys across all samples")
            if len(self.batch_hint_keys) > 1:
                log.warning(
                    "Loading the multicontrol model. Multicontrol inference is not strictly equal to single control"
                )

            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.inference_pipeline.config, config_path)
            log.info(f"Saved config to {config_path}")
        self.benchmark_times = []

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir, sample_id=i_sample)
            if output_path is not None:
                output_paths.append(output_path)

        if is_rank0() and len(self.benchmark_times) > 0:
            avg_time = sum(self.benchmark_times) / len(self.benchmark_times)
            log.info("=" * 50)
            log.info("BENCHMARK RESULTS")
            log.info("=" * 50)
            log.info(f"Benchmark runs: {[f'{t:.2f}s' for t in self.benchmark_times]}")
            log.info(f"Average time (last {self.benchmark_times} runs): {avg_time:.2f} seconds")
            log.info("=" * 50)
        return output_paths

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path, sample_id: int = 0) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        assert sample.prompt is not None
        prompt: str = sample.prompt

        assert sample.negative_prompt is not None
        negative_prompt: str = sample.negative_prompt

        if self.device_rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            # run text guardrail on the prompt
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on prompt...")

                if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                    message = f"Guardrail blocked generation. Prompt: {prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on prompt")

                if not guardrail_presets.run_text_guardrail(
                    negative_prompt,
                    self.text_guardrail_runner,
                ):
                    message = f"Guardrail blocked generation. Negative prompt: {negative_prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on negative prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")

        input_control_video_paths = sample.control_modalities
        log.info(f"Processing the following paths: {input_control_video_paths}")

        sigma_max = None if sample.sigma_max is None else float(sample.sigma_max)

        # control_weight is a string because of multi-control
        control_weight = ""
        for key in self.batch_hint_keys:
            # pyrefly: ignore  # missing-attribute
            control_weight += sample.control_weight_dict.get(key, "0.0") + ","
        control_weight = control_weight[:-1]

        # Measure the time in case of benchmarking, but only for samples which aren't warm-up samples.
        if sample_id > 0 and self.setup_args.benchmark:
            torch.cuda.synchronize()
            start_time = time.time()
        else:
            start_time = None

        # Run model inference
        output_video, control_video_dict, fps, _ = self.inference_pipeline.generate_img2world(
            # pyrefly: ignore  # bad-argument-type
            video_path=path_to_str(sample.video_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_context_path=path_to_str(sample.image_context_path),
            guidance=sample.guidance,
            seed=sample.seed,
            resolution=sample.resolution,
            control_weight=control_weight,
            sigma_max=sigma_max,
            hint_key=sample.hint_keys,
            # pyrefly: ignore  # bad-argument-type
            input_control_video_paths=input_control_video_paths,
            show_control_condition=sample.show_control_condition,
            seg_control_prompt=sample.seg_control_prompt,
            show_input=sample.show_input,
            keep_input_resolution=not sample.not_keep_input_resolution,
            preset_blur_strength=sample.preset_blur_strength,
            preset_edge_threshold=sample.preset_edge_threshold,
            num_conditional_frames=sample.num_conditional_frames,
            num_video_frames_per_chunk=sample.num_video_frames_per_chunk,
            num_steps=sample.num_steps,
        )

        if start_time is not None:
            torch.cuda.synchronize()
            self.benchmark_times.append(time.time() - start_time)

        # Save video
        if self.device_rank == 0:
            output_video = (1.0 + output_video[0]) / 2
            for key in control_video_dict:
                control_video_dict[key] = (1.0 + control_video_dict[key][0]) / 2
                save_img_or_video(control_video_dict[key], f"{output_path}_control_{key}", fps=fps)
                log.info(f"{key} control video saved to {output_path}_control_{key}.mp4")

            # run video guardrail on the video
            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception("Guardrail blocked video2world generation.")
                else:
                    log.success("Passed guardrail on generated video")

                # Convert processed frames back to tensor format
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                output_video = processed_video.to(output_video.device, dtype=output_video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")

            # Remove batch dimension and normalize to [0, 1] range
            save_img_or_video(output_video, str(output_path), fps=fps)
            # save prompt
            prompt_save_path = f"{output_path}.txt"
            with open(prompt_save_path, "w") as f:
                f.write(sample.prompt)
            log.success(f"Generated video saved to {output_path}.mp4")

        torch.cuda.empty_cache()
        return f"{output_path}.mp4"
