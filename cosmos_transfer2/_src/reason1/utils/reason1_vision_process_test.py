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

"""
Usage:
pytest -s cosmos_transfer2/_src/reason1/utils/reason1_vision_process_test.py --L1

"""

import os

import numpy as np
import pytest
import torch
from qwen_vl_utils import process_vision_info

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.utils.helper_test import RunIf
from cosmos_transfer2._src.reason1.datasets.video_decoder_qwen import pixels_to_token
from cosmos_transfer2._src.reason1.utils.reason1_vision_process import (
    process_vision_info as process_vision_info_reason1,
)
from cosmos_transfer2._src.reason1.utils.video_preprocess import tensor_to_pil_images

video_path_s3 = "s3://cosmos_reasoning/benchmark/agibot_reasoning_20250226/v4/clips/327-684224-head_color-0-268-0.mp4"
imaginaire_cache_dir = os.path.expanduser(os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire"))
video_path_local = os.path.join(imaginaire_cache_dir, os.path.basename(video_path_s3))
backend_args = {"backend": "s3", "path_mapping": None, "s3_credential_path": "credentials/pdx_cosmos_benchmark.secret"}


def save_images_as_video(images, output_path, fps=30, codec="mp4v"):
    """
    Save a list of PIL images as a video file.

    Args:
        images (list): List of PIL Image objects
        output_path (str): Path where the video will be saved
        fps (int): Frames per second for the video (default: 30)
        codec (str): Video codec to use (default: 'mp4v')

    Returns:
        str: Path to the saved video file
    """
    import cv2
    import numpy as np

    if not images:
        raise ValueError("No images provided")

    # Get dimensions from the first image
    height, width = images[0].size[1], images[0].size[0]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for i, image in enumerate(images):
            # Convert PIL image to OpenCV format (RGB to BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Write frame to video
            out.write(cv_image)

            # Progress indicator
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(f"Writing frame {i + 1}/{len(images)}")

    finally:
        # Release video writer
        out.release()

    print(f"Video saved to: {output_path}")
    return output_path


@pytest.mark.L1
@RunIf(
    requires_file=[
        "credentials/pdx_cosmos_benchmark.secret",
    ],
)
def test_vision_process_video():
    if not os.path.exists(video_path_local):
        easy_io.copyfile_to_local(video_path_s3, video_path_local, dst_type="file", backend_args=backend_args)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path_local,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    _, video_inputs_, video_kwargs_ = process_vision_info_reason1(messages, return_video_kwargs=True)

    if isinstance(video_inputs, list):
        for video_input, video_input_ in zip(video_inputs, video_inputs_):
            torch.testing.assert_close(video_input, video_input_)
    else:
        assert video_inputs == video_inputs_

    assert video_kwargs == video_kwargs_

    log.info("test_vision_process passed")


@pytest.mark.L1
@RunIf(
    requires_file=[
        "credentials/pdx_cosmos_benchmark.secret",
    ],
)
def test_vision_process_image():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)
    image_inputs_, _, _ = process_vision_info_reason1(messages, return_video_kwargs=True)

    if isinstance(image_inputs, list):
        for image_input, image_input_ in zip(image_inputs, image_inputs_):
            np.testing.assert_array_equal(np.array(image_input), np.array(image_input_))
    else:
        assert image_inputs == image_inputs_

    log.info("test_vision_process passed")


@pytest.mark.L1
@RunIf(
    requires_file=[
        "credentials/pdx_cosmos_benchmark.secret",
    ],
)
def test_vision_process_video_timestamp():
    if not os.path.exists(video_path_local):
        easy_io.copyfile_to_local(video_path_s3, video_path_local, dst_type="file", backend_args=backend_args)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path_local,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    _, video_inputs_, video_kwargs_ = process_vision_info_reason1(
        messages, return_video_kwargs=True, timestamp_video=True
    )

    # original implementation returns a list of tensors
    for idx, video_input in enumerate(video_inputs):
        # tensor of video frames from original qwen implementation
        video_inputs[idx] = tensor_to_pil_images(video_input)

    for video_input, video_input_ in zip(video_inputs, video_inputs_):
        assert len(video_input) == len(video_input_), (
            f"videos must be same length. original: {len(video_input)}, reason1: {len(video_input_)}"
        )
        for frame_input, frame_input_ in zip(video_input, video_input_):
            assert frame_input.size[0] == frame_input_.size[0], (
                f"frames must be same width. original: {frame_input.size()}, reason1: {frame_input_.size()}"
            )
            assert frame_input.size[1] == frame_input_.size[1] - 28, (
                f"frames height must be 28 pixels smaller. original: {frame_input.size()}, reason1: {frame_input_.size()}"
            )
            assert np.array_equal(np.array(frame_input), np.array(frame_input_)[:-28])

    assert video_kwargs == video_kwargs_

    # optionally save videos for debugging
    if os.environ.get("LOGURU_LEVEL") == "DEBUG":
        save_images_as_video(video_inputs[0], "output_original.mp4")
        save_images_as_video(video_inputs_[0], "output_reason1.mp4")

    log.info("test_vision_process passed")


@pytest.mark.L1
@RunIf(
    requires_file=[
        "credentials/pdx_cosmos_benchmark.secret",
    ],
)
def test_vision_process_video_max_num_vision_tokens(max_num_vision_tokens: int = 1024):
    if not os.path.exists(video_path_local):
        easy_io.copyfile_to_local(video_path_s3, video_path_local, dst_type="file", backend_args=backend_args)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path_local,
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 256 * 28 * 28,
                    "total_pixels": 20480 * 28 * 28,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    _, video_inputs_, video_kwargs_ = process_vision_info_reason1(
        messages, return_video_kwargs=True, max_num_vision_tokens=max_num_vision_tokens
    )

    num_frames = len(video_inputs_[0])
    width, height = video_inputs_[0][0].size
    print(f"video_inputs_: {num_frames} frames, {width}x{height}")
    print(f"video_kwargs_: {video_kwargs_}")

    tokens = pixels_to_token(width * height * num_frames, patch_size=14, temporal_patch_size=2)
    print(f"tokens: {tokens}")
    assert tokens <= max_num_vision_tokens, (
        f"tokens must be less than or equal to {max_num_vision_tokens}. got {tokens}"
    )
    log.info("test_vision_process passed")
