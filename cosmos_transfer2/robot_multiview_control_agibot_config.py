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

"""Configuration for Transfer2.5 Agibot control-conditioned multiview inference."""

from typing import Annotated, Literal, Protocol

import pydantic
import torch
import tyro

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2.config import (
    CommonInferenceArguments,
    CommonSetupArguments,
    ModelKey,
    ModelVariant,
    ResolvedDirectoryPath,
    ResolvedFilePath,
)

# Shared mapping: control_type -> ModelVariant for inference and post-train experiments.
CONTROL_TYPE_TO_MODEL_VARIANT: dict[str, ModelVariant] = {
    "depth": ModelVariant.ROBOT_MULTIVIEW_AGIBOT_DEPTH,
    "edge": ModelVariant.ROBOT_MULTIVIEW_AGIBOT_EDGE,
    "vis": ModelVariant.ROBOT_MULTIVIEW_AGIBOT_VIS,
    "seg": ModelVariant.ROBOT_MULTIVIEW_AGIBOT_SEG,
}


class ControlVideoLoadFn(Protocol):
    """Protocol for control video loading function."""

    def __call__(
        self,
        text: str,
        video: torch.Tensor,
        control_video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        *,
        width: int,
        height: int,
        control_type: str,
    ) -> list[dict]: ...


class AgibotViewConfig(pydantic.BaseModel):
    """Configuration for a single agibot camera view."""

    model_config = pydantic.ConfigDict(extra="forbid")
    input_path: ResolvedFilePath | None = None
    """Path to the input video for this view. Required for I2V mode (num_conditional_frames=1) and edge/vis controls (used to generate control on-the-fly). For depth/seg controls in T2V mode (num_conditional_frames=0), can be omitted - will use control video as mock input if not provided."""
    control_path: ResolvedFilePath | None = None
    """Path to the control video for this view. Required for depth/seg controls, omit for edge/vis (generated on-the-fly)"""


class RobotMultiviewControlAgibotSetupArguments(CommonSetupArguments):
    """Setup arguments for Transfer2.5 Agibot control-conditioned multiview inference."""

    config_file: str = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"

    input_root: ResolvedDirectoryPath
    """Directory containing input videos, control videos, and captions (recommended for best quality)"""

    num_views: pydantic.PositiveInt = 3
    """Number of camera views (3 for Agibot: head_color, hand_left, hand_right)"""

    control_type: Literal["depth", "edge", "vis", "seg"]
    """Type of control signal (depth, edge, vis, seg)"""

    dataloader_num_workers: pydantic.NonNegativeInt = 0
    """Number of workers to use in dataloader"""

    @pydantic.model_validator(mode="before")
    @classmethod
    def set_model_from_control_type(cls, data: dict) -> dict:
        """Set model field based on control_type for parent class validation."""
        if isinstance(data, dict) and "control_type" in data:
            data["model"] = f"robot/multiview-agibot-{data['control_type']}"
        return data

    @pydantic.model_validator(mode="after")
    def validate_context_parallel_size(self):
        """Validate that context_parallel_size meets Agibot model requirements."""
        if self.context_parallel_size is not None and self.context_parallel_size < 4:
            raise ValueError(
                f"Agibot multiview models require minimum 4 GPUs due to context parallelism constraints. "
                f"Got context_parallel_size={self.context_parallel_size}. "
                f"Use: torchrun --nproc_per_node=<NUM_GPUS> (where NUM_GPUS >= 4)"
            )
        return self

    @classmethod
    def model_key_for_control_type(cls, control_type: Literal["depth", "edge", "vis", "seg"]) -> ModelKey:
        """Return the model key for a control type (e.g. for checkpoint lookup without building a full instance)."""
        return ModelKey(variant=CONTROL_TYPE_TO_MODEL_VARIANT[control_type])

    @property
    def model_key(self) -> ModelKey:
        """Get model key based on control type."""
        return self.model_key_for_control_type(self.control_type)


class RobotMultiviewControlAgibotInferenceArguments(CommonInferenceArguments):
    """Inference arguments for Transfer2.5 Agibot control-conditioned multiview."""

    head_color: Annotated[AgibotViewConfig, tyro.conf.Suppress] = pydantic.Field(default_factory=AgibotViewConfig)
    """Head color camera view configuration"""

    hand_left: Annotated[AgibotViewConfig, tyro.conf.Suppress] = pydantic.Field(default_factory=AgibotViewConfig)
    """Left hand camera view configuration"""

    hand_right: Annotated[AgibotViewConfig, tyro.conf.Suppress] = pydantic.Field(default_factory=AgibotViewConfig)
    """Right hand camera view configuration"""

    prompt: str | None = None
    """Text prompt override applied to all camera views (highest priority). If not provided, will try to load per-camera captions from files, then fall back to camera prefix (recommended: provide descriptive prompts for best quality)"""

    add_camera_prefix: bool = True
    """Whether to add camera-specific prefix to captions"""

    num_conditional_frames: Literal[0, 1] = 1
    """Number of conditional frames to condition on. 0: control-to-video, 1: image-to-video"""

    num_video_frames_per_chunk: pydantic.PositiveInt = 93
    """Number of video frames per chunk in the chunk-wise long video generation"""

    target_height: pydantic.PositiveInt = 720
    """Target height for inference"""

    target_width: pydantic.PositiveInt = 1280
    """Target width for inference"""

    fps: pydantic.PositiveInt = 10
    """Frames per second for output video"""

    num_steps: int = pydantic.Field(
        default=1 if SMOKE else 35, description="Number of diffusion denoising steps for generation"
    )
    """Number of diffusion sampling steps (higher values = better quality but slower generation)"""

    use_negative_prompt: bool = True
    """Whether to use negative prompting during generation"""

    preset_edge_threshold: str | None = "medium"
    """Edge detection strength (choices: very_low, low, medium, high, very_high). Only used when control_type='edge', ignored for other control types"""

    preset_blur_strength: str | None = "medium"
    """Blur strength (choices: very_low, low, medium, high, very_high). Only used when control_type='vis', ignored for other control types"""

    control_weight: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0
    """Control weight (range 0.0 to 1.0): how strongly the output adheres to the control signal. 1.0 = full control, 0.0 = no control."""

    save_combined_views: bool = True
    """Save a single concatenated video containing all views side-by-side. If False, saves individual split views and a grid view."""

    # Autoregressive inference mode
    enable_autoregressive: bool = False
    """Enable autoregressive mode to generate videos longer than the model's native temporal capacity."""

    num_chunks: int = pydantic.Field(
        default=2,
        ge=1,
        description="Number of chunks to process auto-regressively",
    )
    """Number of chunks to generate in autoregressive mode. Only used when enable_autoregressive=True."""

    chunk_overlap: int = pydantic.Field(
        default=1, description="Number of overlapping frames between consecutive chunks"
    )
    """Number of overlapping frames between consecutive chunks for temporal consistency. Only used when enable_autoregressive=True."""

    @pydantic.model_validator(mode="after")
    def validate_input_paths(self):
        """Validate that input_path is provided when required.

        NOTE: This validator can only check num_conditional_frames, not control_type (which is in setup_args).
        Full validation happens at runtime in inference_cli.py where both values are available.

        Rules:
        - I2V mode (num_conditional_frames=1): input_path always required
        - T2V mode (num_conditional_frames=0): input_path optional for depth/seg (runtime check will use control as mock)
          WARNING: For edge/vis controls, input_path is still required (used to generate control on-the-fly)
        """
        # For I2V mode, input_path is required for all cameras
        if self.num_conditional_frames > 0:
            for camera in ["head_color", "hand_left", "hand_right"]:
                view_config = getattr(self, camera)
                if view_config.input_path is None:
                    raise ValueError(
                        f"input_path is required for {camera} camera view when num_conditional_frames={self.num_conditional_frames}. "
                        f"For T2V mode with depth/seg controls, set num_conditional_frames=0 to allow optional input paths."
                    )
        # For T2V mode (num_conditional_frames=0), we cannot validate here because control_type is unknown
        # - For edge/vis: input_path IS REQUIRED (will fail at runtime if missing)
        # - For depth/seg: input_path is optional (runtime will use control as mock input)
        return self


def validate_control_params(
    inference_args: RobotMultiviewControlAgibotInferenceArguments,
    control_type: Literal["depth", "edge", "vis", "seg"],
) -> list[str]:
    """Validate that control-type-specific parameters are used correctly.

    Args:
        inference_args: Inference arguments to validate
        control_type: Control type from setup arguments

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check preset_edge_threshold
    if (
        inference_args.preset_edge_threshold is not None
        and inference_args.preset_edge_threshold != "medium"
        and control_type != "edge"
    ):
        warnings.append(
            f"preset_edge_threshold='{inference_args.preset_edge_threshold}' is set but control_type='{control_type}'. "
            f"This parameter is only used for control_type='edge' and will be ignored."
        )

    # Check preset_blur_strength
    if (
        inference_args.preset_blur_strength is not None
        and inference_args.preset_blur_strength != "medium"
        and control_type != "vis"
    ):
        warnings.append(
            f"preset_blur_strength='{inference_args.preset_blur_strength}' is set but control_type='{control_type}'. "
            f"This parameter is only used for control_type='vis' and will be ignored."
        )

    return warnings
