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

from pathlib import Path

import torch
from loguru import logger

from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2_multiview.scripts.mv_visualize_helper import arrange_video_visualization
from cosmos_transfer2._src.transfer2_multiview.inference import inference_cli
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2.config import MODEL_CHECKPOINTS
from cosmos_transfer2.robot_multiview_control_agibot_config import (
    RobotMultiviewControlAgibotInferenceArguments,
    RobotMultiviewControlAgibotSetupArguments,
    validate_control_params,
)


class TextVideoControlDataset(torch.utils.data.Dataset):
    """Simple dataset that just provides video IDs and metadata."""

    def __init__(
        self,
        input_root: str | Path,
        args: RobotMultiviewControlAgibotSetupArguments,
        inference_args: list[RobotMultiviewControlAgibotInferenceArguments],
    ):
        self.input_root = Path(input_root)
        self.args = args
        self.data = inference_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """Just return metadata - actual loading happens in inference loop using inference_cli functions."""
        inf_args = self.data[index]
        return {
            "name": inf_args.name,
            # Per-camera view paths
            "head_color_input": str(inf_args.head_color.input_path),
            "head_color_control": str(inf_args.head_color.control_path) if inf_args.head_color.control_path else "",
            "hand_left_input": str(inf_args.hand_left.input_path),
            "hand_left_control": str(inf_args.hand_left.control_path) if inf_args.hand_left.control_path else "",
            "hand_right_input": str(inf_args.hand_right.input_path),
            "hand_right_control": str(inf_args.hand_right.control_path) if inf_args.hand_right.control_path else "",
            # Other parameters
            "seed": inf_args.seed,
            "guidance": inf_args.guidance,
            "num_steps": inf_args.num_steps,
            "use_negative_prompt": inf_args.use_negative_prompt,
            "add_camera_prefix": inf_args.add_camera_prefix,
            "prompt": inf_args.prompt if inf_args.prompt else "",
            "num_conditional_frames": inf_args.num_conditional_frames,
            "num_video_frames_per_chunk": inf_args.num_video_frames_per_chunk,
            "target_height": inf_args.target_height,
            "target_width": inf_args.target_width,
            "fps": inf_args.fps,
            "preset_edge_threshold": inf_args.preset_edge_threshold,
            "preset_blur_strength": inf_args.preset_blur_strength,
            "control_weight": inf_args.control_weight,
            "save_combined_views": inf_args.save_combined_views,
            "enable_autoregressive": inf_args.enable_autoregressive,
            "num_chunks": inf_args.num_chunks,
            "chunk_overlap": inf_args.chunk_overlap,
        }


class RobotMultiviewControlAgibotInference:
    """Class-based interface for robot multiview control-conditioned inference (Transfer2.5 Agibot models)."""

    def __init__(self, setup_args: RobotMultiviewControlAgibotSetupArguments):
        """Initialize the inference pipeline.

        Args:
            setup_args: Setup configuration including model path, device settings, etc.
        """
        logger.debug(f"{setup_args.__class__.__name__}({setup_args})")

        # Disable gradient calculations for inference
        torch.enable_grad(False)

        self.setup_args = setup_args

        # Load model checkpoint
        checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
        experiment = setup_args.experiment or checkpoint.experiment
        checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

        assert experiment is not None, "experiment must be set"
        assert setup_args.context_parallel_size is not None, "context_parallel_size must be set"

        # Initialize Control-conditioned model
        # Override base_load_from to None since we're using fully trained checkpoints
        self.pipe = ControlVideo2WorldInference(
            experiment_name=experiment,
            ckpt_path=checkpoint_path,
            context_parallel_size=setup_args.context_parallel_size,
            experiment_opts=["model.config.base_load_from=null"],
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

        # Determine if this is rank 0 for distributed processing
        self.rank0 = True

        if setup_args.context_parallel_size > 1:
            self.rank0 = distributed.get_rank() == 0

    def generate(
        self,
        samples: list[RobotMultiviewControlAgibotInferenceArguments],
        output_dir: Path,
    ) -> list[str]:
        """Generate videos for a list of samples.

        Args:
            samples: List of inference arguments for each sample to generate
            output_dir: Directory where output videos will be saved

        Returns:
            List of output video paths
        """
        assert len(samples) > 0, "At least one sample must be provided"

        sample_names = [sample.name for sample in samples]
        logger.info(f"Generating {len(samples)} samples: {sample_names}")

        # Create dataset - just metadata, actual loading happens in inference loop
        dataset = TextVideoControlDataset(
            input_root=self.setup_args.input_root,
            args=self.setup_args,
            inference_args=samples,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=self.setup_args.dataloader_num_workers,
        )

        logger.info(f"Starting inference loop. Dataset has {len(dataset)} samples.")

        output_paths: list[str] = []
        for batch_idx, data_batch in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(dataset)}")
            output_path = self._generate_sample(samples[batch_idx], data_batch, output_dir)
            if output_path is not None:
                output_paths.append(output_path)

        return output_paths

    def _generate_sample(
        self,
        inf_args: RobotMultiviewControlAgibotInferenceArguments,
        data_batch: dict,
        output_dir: Path,
    ) -> str | None:
        """Generate a single video sample.

        Args:
            inf_args: Inference arguments for this sample
            data_batch: Data batch from dataloader
            output_dir: Directory where output videos will be saved

        Returns:
            Path to the generated video, or None if generation failed
        """
        name = data_batch["name"][0]

        # Build per-camera view paths dictionary
        # Note: input_path can be None for T2V mode with depth/seg controls
        camera_input_paths = {
            "head_color": Path(data_batch["head_color_input"][0]) if data_batch["head_color_input"][0] else None,
            "hand_left": Path(data_batch["hand_left_input"][0]) if data_batch["hand_left_input"][0] else None,
            "hand_right": Path(data_batch["hand_right_input"][0]) if data_batch["hand_right_input"][0] else None,
        }

        camera_control_paths = {
            "head_color": Path(data_batch["head_color_control"][0]) if data_batch["head_color_control"][0] else None,
            "hand_left": Path(data_batch["hand_left_control"][0]) if data_batch["hand_left_control"][0] else None,
            "hand_right": Path(data_batch["hand_right_control"][0]) if data_batch["hand_right_control"][0] else None,
        }

        # Validate control-type-specific parameters
        validation_warnings = validate_control_params(inf_args, self.setup_args.control_type)
        for warning in validation_warnings:
            logger.warning(warning)

        # Compute effective target_frames based on autoregressive settings
        target_frames = data_batch["num_video_frames_per_chunk"][0].item()
        enable_autoregressive = data_batch["enable_autoregressive"][0].item()

        if enable_autoregressive:
            num_chunks = data_batch["num_chunks"][0].item()
            chunk_overlap = data_batch["chunk_overlap"][0].item()
            # Calculate total frames needed for autoregressive generation
            target_frames = target_frames + (target_frames - chunk_overlap) * (num_chunks - 1)
            logger.info(
                f"Autoregressive mode enabled: generating {num_chunks} chunks with "
                f"{chunk_overlap} frame overlap, total frames: {target_frames}"
            )

        video, model_data_batch = inference_cli.generate_multiview_control_video(
            vid2world_cli=self.pipe,
            camera_input_paths=camera_input_paths,
            camera_control_paths=camera_control_paths,
            camera_order=list(inference_cli.CAMERA_TO_VIEW_INDEX_AGIBOT.keys()),
            camera_to_view_index=inference_cli.CAMERA_TO_VIEW_INDEX_AGIBOT,
            camera_to_caption_prefix=inference_cli.CAMERA_TO_CAPTION_PREFIX_AGIBOT,
            control_type=self.setup_args.control_type,
            target_frames=target_frames,
            target_size=(data_batch["target_height"][0].item(), data_batch["target_width"][0].item()),
            num_conditional_frames=data_batch["num_conditional_frames"][0].item(),
            fps=data_batch["fps"][0].item(),
            preset_edge_threshold=data_batch["preset_edge_threshold"][0],
            preset_blur_strength=data_batch["preset_blur_strength"][0],
            add_camera_prefix=data_batch["add_camera_prefix"][0].item(),
            guidance=data_batch["guidance"][0].item(),
            seed=data_batch["seed"][0].item(),
            num_steps=data_batch["num_steps"][0].item(),
            use_negative_prompt=data_batch["use_negative_prompt"][0].item(),
            control_weight=data_batch["control_weight"][0].item(),
            enable_autoregressive=enable_autoregressive,
            chunk_overlap=data_batch["chunk_overlap"][0].item() if enable_autoregressive else 1,
            prompt_override=data_batch["prompt"][0] if data_batch["prompt"][0] else None,
            input_root_for_captions=self.setup_args.input_root,
        )

        if self.rank0:
            save_root = Path(output_dir)
            save_root.mkdir(parents=True, exist_ok=True)
            # Use 'name' field for output filename (consistent with other inference scripts)
            # save_img_or_video automatically adds .mp4 extension, so don't add it here
            output_path = save_root / name

            fps = int(data_batch["fps"][0].item())

            if data_batch["save_combined_views"][0]:
                # Save combined video with all views arranged horizontally
                video_arranged = arrange_video_visualization(video, model_data_batch, method="width", dataset="agibot")
                save_img_or_video(video_arranged[0], str(output_path), fps=fps)
                logger.info(f"Saved video to {output_path}.mp4")
                return f"{output_path}.mp4"
            else:
                # Save individual view videos and grid
                # video shape: (1, C, V*T, H, W)
                camera_keys = list(model_data_batch["camera_keys_selection"][0])
                n_views = len(camera_keys)

                # Get video tensor and reshape to separate views
                video_tensor = video[0]  # Remove batch dimension: (C, V*T, H, W)
                C, VT, H, W = video_tensor.shape
                T = VT // n_views

                if VT % n_views != 0:
                    raise ValueError(f"Video frames ({VT}) not divisible by number of views ({n_views})")

                # Split into individual views
                view_tensors = []
                for view_index in range(n_views):
                    start = view_index * T
                    end = start + T
                    view_tensor = video_tensor[:, start:end, :, :]  # (C, T, H, W)
                    view_name = camera_keys[view_index]
                    view_tensors.append((view_name, view_tensor))

                # Save individual view videos
                output_messages = []
                for view_name, view_tensor in view_tensors:
                    view_output_path = f"{output_path}_{view_name}"
                    save_img_or_video(view_tensor, view_output_path, fps=fps, quality=8)
                    output_messages.append(f"{view_output_path}.mp4")

                # Save grid video (for agibot: 1 row x 3 cols)
                grid_rows, grid_cols = 1, 3
                grid_tensor = torch.zeros(
                    (C, T, grid_rows * H, grid_cols * W), dtype=video_tensor.dtype, device=video_tensor.device
                )

                for idx in range(min(len(view_tensors), grid_rows * grid_cols)):
                    row, col = idx // grid_cols, idx % grid_cols
                    grid_tensor[:, :, row * H : (row + 1) * H, col * W : (col + 1) * W] = view_tensors[idx][1]

                grid_output_path = f"{output_path}_grid"
                save_img_or_video(grid_tensor, grid_output_path, fps=fps, quality=8)
                output_messages.append(f"{grid_output_path}.mp4 ({n_views} views in {grid_rows}x{grid_cols} grid)")

                # Log all outputs at once
                if output_messages:
                    logger.info("Generated videos saved to:\n" + "\n".join(output_messages))

                return f"{output_path}.mp4"

        return None

    def cleanup(self):
        """Clean up distributed resources."""
        # Synchronize all processes before cleanup
        # pyrefly: ignore  # unsupported-operation
        if self.setup_args.context_parallel_size > 1:
            torch.distributed.barrier()

        # Clean up distributed resources
        self.pipe.cleanup()
