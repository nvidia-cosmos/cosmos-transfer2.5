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

from typing import Annotated, ClassVar

import pydantic

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2.config import (
    MODEL_CHECKPOINTS,
    CommonInferenceArguments,
    CommonSetupArguments,
    ModelKey,
    ModelVariant,
    ResolvedFilePath,
    get_model_literal,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.AUTO_MULTIVIEW)
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_MODEL_KEY]
MULTIVIEW_CAMERA_KEYS: tuple[str, ...] = (
    "front_wide",
    "cross_right",
    "rear_right",
    "rear",
    "rear_left",
    "cross_left",
    "front_tele",
)


class MultiviewSetupArguments(CommonSetupArguments):
    """Arguments for multiview setup."""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.AUTO_MULTIVIEW]) = DEFAULT_MODEL_KEY.name


class ViewConfig(pydantic.BaseModel):
    """Configuration for a single view."""

    model_config = pydantic.ConfigDict(extra="forbid")
    input_path: ResolvedFilePath | None = None
    """Path to the input video for this view, required when num_conditional_frames > 0"""
    control_path: ResolvedFilePath | None = None
    """Path to the control video for this view, required for every view"""

    # Autoregressive inference mode
    num_conditional_frames_per_view: int = pydantic.Field(
        default=0,
        description="Number of video frames to condition on per view. Must be one of [0, 1, 5].",
        ge=0,
        le=5,
    )


class MultiviewInferenceArguments(CommonInferenceArguments):
    """All the required values to generate image from text at a given resolution."""

    view_key_order: ClassVar[tuple[str, ...]] = MULTIVIEW_CAMERA_KEYS
    num_conditional_frames: int = pydantic.Field(default=1)
    """Number of video frames to condition on."""
    control_weight: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0
    """Control weight for generation."""
    front_wide: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Front wide view configuration."""
    rear: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear view configuration."""
    rear_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear left view configuration."""
    rear_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Rear right view configuration."""
    cross_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Cross left view configuration."""
    cross_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Cross right view configuration."""
    front_tele: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    """Front tele view configuration."""

    fps: pydantic.PositiveInt = 10
    """Frames per second for output video."""
    num_steps: int = pydantic.Field(
        default=1 if SMOKE else 35, description="Number of diffusion denoising steps for generation"
    )
    """Number of diffusion sampling steps (higher values = better quality but slower generation)."""

    # Autoregressive inference mode
    enable_autoregressive: bool = False
    """Enable autoregressive mode to generate videos longer than the model's native temporal capacity."""
    num_chunks: int = pydantic.Field(
        default=2,
        ge=1,
        description="Number of chunks to process auto-regressively",
    )
    """Number of frames the model generates per view in a single forward pass (chunk size, typically 29 or 61)."""
    chunk_overlap: int = pydantic.Field(
        default=1, description="Number of overlapping frames between consecutive chunks"
    )
    """Number of overlapping frames between consecutive chunks for temporal consistency."""

    save_combined_views: bool = True
    """Save a single concatenated video containing all views side-by-side. If False, saves individual split views and a grid view."""

    @pydantic.model_validator(mode="after")
    def validate_input_paths(self):
        """Validate that input_path is provided when num_conditional_frames > 0."""
        active_views = self.active_view_configs
        if not active_views:
            raise ValueError("At least one view configuration with a control_path must be provided.")

        if self.num_conditional_frames > 0 and not self.enable_autoregressive:
            missing_input_paths = [
                view_name for view_name, view_config in active_views if view_config.input_path is None
            ]
            if missing_input_paths:
                raise ValueError(
                    "input_path is required for all active views when num_conditional_frames > 0. "
                    f"Missing input_path for views: {', '.join(missing_input_paths)}"
                )

        if self.enable_autoregressive:
            # Check per-view frame counts when autoregressive mode is enabled.
            num_conditional_frames_per_view = [
                view_config.num_conditional_frames_per_view for _, view_config in active_views
            ]
            # check if any view has num_conditional_frames_per_view not in (0, 1, 5)
            if any(frames not in (0, 1, 5) for frames in num_conditional_frames_per_view):
                raise ValueError(
                    "num_conditional_frames_per_view must be one of [0, 1, 5] "
                    f"for views: {[view_name for view_name, _ in active_views]}"
                )
            # Check if some (but not all) views have non-zero values
            if any(frames == 0 for frames in num_conditional_frames_per_view) and not all(
                frames == 0 for frames in num_conditional_frames_per_view
            ):
                raise ValueError(
                    "num_conditional_frames_per_view must be consistent across all active views in autoregressive mode. "
                    "Either set it for all views or leave all at default (0)."
                )

            if any(frames > 0 for frames in num_conditional_frames_per_view):
                missing_input_paths = [
                    view_name for view_name, view_config in active_views if view_config.input_path is None
                ]
                if missing_input_paths:
                    raise ValueError(
                        "input_path is required for all active views when num_conditional_frames > 0. "
                        f"Missing input_path for views: {', '.join(missing_input_paths)}"
                    )
        return self

    @property
    def active_view_configs(self) -> list[tuple[str, ViewConfig]]:
        """Return the ordered list of view configs that have a control path supplied."""
        active_views: list[tuple[str, ViewConfig]] = []
        for view_name in self.view_key_order:
            view_config = getattr(self, view_name)
            if view_config.control_path is not None:
                active_views.append((view_name, view_config))
        return active_views

    @property
    def active_camera_keys(self) -> tuple[str, ...]:
        """Ordered camera keys with control data."""
        return tuple(view_name for view_name, _ in self.active_view_configs)

    @property
    def input_and_control_paths(self):
        input_and_control_paths: dict[str, ResolvedFilePath | None] = {}
        for view_name, view_config in self.active_view_configs:
            input_and_control_paths[f"{view_name}_input"] = view_config.input_path
            assert view_config.control_path is not None
            input_and_control_paths[f"{view_name}_control"] = view_config.control_path
        return input_and_control_paths


MultiviewInferenceOverrides = get_overrides_cls(
    MultiviewInferenceArguments,
    exclude=[
        "name",
        "front_wide",
        "rear",
        "rear_left",
        "rear_right",
        "cross_left",
        "cross_right",
        "front_tele",
    ],
)
