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

"""DepthAnythingV2 model for frame-by-frame depth estimation."""

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from cosmos_transfer2._src.transfer2.auxiliary.depth_anything.utils import get_model_cache_path

logger = logging.getLogger(__name__)

WEIGHTS_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2Model:
    """
    Simplified DepthAnythingV2 model matching the source implementation.

    Main method:
        generate_float16_array_from_video_array(video_array, max_frames) -> np.ndarray[np.float16]
    """

    def __init__(
        self,
        model_checkpoint: str = WEIGHTS_NAME,
        device: Optional[str] = None,
    ):
        self.model_checkpoint = model_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image_processor = None

    def setup(self) -> None:
        """Load the model and image processor."""
        if self.model is not None:
            logger.info("Model already loaded, skipping setup")
            return

        logger.info(f"Loading DepthAnythingV2 model from {self.model_checkpoint}")

        try:
            cache_path = get_model_cache_path(self.model_checkpoint)
            logger.info(f"Using cache directory: {cache_path}")
        except Exception as e:
            logger.warning(f"Could not set up cache directory: {e}")
            cache_path = None

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_checkpoint,
            cache_dir=cache_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_checkpoint,
            cache_dir=cache_path,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")

    def generate_float16_array_from_video_array(
        self, video_npy_array: np.ndarray, max_frames: Optional[int] = None
    ) -> npt.NDArray[np.float16]:
        """
        Generate depth maps from video array.

        Args:
            video_npy_array: Video array with shape (T, H, W, 3) and dtype uint8
            max_frames: Maximum number of frames to process

        Returns:
            Depth array with shape (T, H, W) and dtype float16
        """
        # vid array shape: [T, H, W, 3]
        frame_width, frame_height = video_npy_array.shape[2], video_npy_array.shape[1]
        original_size = (frame_width, frame_height)

        depth_frames = []
        for frame in video_npy_array:
            inputs = self.image_processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth[0].unsqueeze(0).unsqueeze(0),
                size=original_size[::-1],  # should be (height, width)
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().numpy().astype(np.float16)
            depth_frames.append(depth)
            if max_frames is not None and len(depth_frames) >= max_frames:
                break
        depth_frames = np.stack(depth_frames)
        return depth_frames
