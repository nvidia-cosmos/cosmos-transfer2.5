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

"""Configuration for Plenoptic Multiview Camera inference."""

from typing import Literal, Protocol

import pydantic
import torch

from cosmos_transfer2._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS

if not EXPERIMENTAL_CHECKPOINTS:
    raise ImportError(
        "Plenoptic multiview inference requires experimental checkpoints. "
        "Set COSMOS_EXPERIMENTAL_CHECKPOINTS=1 environment variable to enable."
    )

from cosmos_transfer2.config import (
    CommonInferenceArguments,
    CommonSetupArguments,
    Guidance,
    ModelKey,
    ModelVariant,
    ResolvedDirectoryPath,
    ResolvedFilePath,
    get_model_literal,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_MANY_CAMERA)

# Default camera motion types available
CAMERA_MOTION_TYPES = (
    "static",
    "rot_left",
    "rot_right",
    "arc_left",
    "arc_right",
    "azimuth_left",
    "azimuth_right",
    "tilt_up",
    "tilt_down",
    "translate_up_rot",
    "translate_down_rot",
    "elevation_up_1",
    "elevation_up_2",
    "zoom_in",
    "zoom_out",
    "distance_away_1",
    "distance_away_2",
)

DEFAULT_NEGATIVE_PROMPT_PLENOPTIC = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    "poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, poorly edited content, "
    "jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
)


class CameraLoadFn(Protocol):
    """Protocol for camera loading functions."""

    def __call__(
        self,
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        width: int,
        height: int,
        input_video_res: str,
        focal_length: int,
        camera_list: list[str],
        extrinsic_scale: float,
    ) -> list[dict]: ...


class PlenopticSetupArguments(CommonSetupArguments):
    """Setup arguments for plenoptic multiview inference."""

    config_file: str = "cosmos_transfer2/_src/predict2/camera/configs/multiview_camera/config.py"

    base_path: ResolvedDirectoryPath
    """Directory containing camera extrinsics/intrinsics and input data"""

    num_input_frames: pydantic.PositiveInt = 24
    """Number of latent conditional frames to condition on"""
    num_output_frames: pydantic.PositiveInt = 93
    """Number of output frames to generate"""
    num_input_video: pydantic.PositiveInt = 4
    """Number of input/conditioning camera views"""
    num_output_video: pydantic.PositiveInt = 1
    """Number of output camera views to generate"""
    input_video_res: Literal["480p"] = "480p"
    """Input video resolution (only 480p supported)"""
    camera_load_create_fn: str = "cosmos_transfer2.plenoptic.load_plenoptic_camera_fn"
    """Function to load camera intrinsic and extrinsic data"""
    dataloader_num_workers: pydantic.NonNegativeInt = 0
    """Number of workers for data loading"""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.ROBOT_MULTIVIEW_MANY_CAMERA]) = DEFAULT_MODEL_KEY.name


class PlenopticInferenceArguments(CommonInferenceArguments):
    """Inference arguments for a single plenoptic multiview sample."""

    input_path: ResolvedFilePath
    """Path to the input video file"""

    camera_sequence: list[str] = pydantic.Field(
        default_factory=lambda: ["static", "rot_left", "arc_right", "azimuth_right"]
    )
    """Ordered list of camera motion types to generate. First is input/conditioning view."""

    # pyrefly: ignore  # bad-override
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT_PLENOPTIC
    """Negative prompt for classifier-free guidance"""
    guidance: Guidance = 7
    """Guidance scale value"""
    fps: pydantic.PositiveInt = 30
    """Output video FPS"""
    focal_length: pydantic.PositiveInt = 24
    """Focal length for camera intrinsics in mm (default: 24)"""
    extrinsic_scale: float = 1.5
    """Scale factor for camera extrinsics"""

    @pydantic.model_validator(mode="after")
    def validate_camera_sequence(self):
        """Validate camera sequence."""
        if len(self.camera_sequence) < 2:
            raise ValueError("camera_sequence must have at least 2 cameras (1 input + 1 output)")
        invalid_cameras = set(self.camera_sequence) - set(CAMERA_MOTION_TYPES)
        if invalid_cameras:
            raise ValueError(f"Invalid camera types: {invalid_cameras}. Valid types: {CAMERA_MOTION_TYPES}")
        return self


PlenopticInferenceOverrides = get_overrides_cls(PlenopticInferenceArguments, exclude=["name", "camera_sequence"])
