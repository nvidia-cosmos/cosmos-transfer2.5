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
Inference script for constructing data_batch from videos, control videos (world_scenario / depth / seg / edge / vis), and captions (recommended),
then running transfer2_multiview model.


# mads

Expected directory structure for MADS dataset:
```
input_root/
├── videos/                                    # Input video folder
│   ├── ftheta_camera_front_wide_120fov/      # or camera_front_wide_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── ftheta_camera_cross_right_120fov/     # or camera_cross_right_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── ftheta_camera_rear_right_70fov/       # or camera_rear_right_70fov/
│   ├── ftheta_camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
│   ├── ftheta_camera_rear_left_70fov/        # or camera_rear_left_70fov/
│   ├── ftheta_camera_cross_left_120fov/      # or camera_cross_left_120fov/
│   └── ftheta_camera_front_tele_30fov/       # or camera_front_tele_30fov/
│
├── world_scenario/                            # Control video folder (world scenario)
│   ├── ftheta_camera_front_wide_120fov/      # or camera_front_wide_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── ftheta_camera_cross_right_120fov/     # or camera_cross_right_120fov/
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── ftheta_camera_rear_right_70fov/       # or camera_rear_right_70fov/
│   ├── ftheta_camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
│   ├── ftheta_camera_rear_left_70fov/        # or camera_rear_left_70fov/
│   ├── ftheta_camera_cross_left_120fov/      # or camera_cross_left_120fov/
│   └── ftheta_camera_front_tele_30fov/       # or camera_front_tele_30fov/
│
└── captions/                                  # Caption folder (recommended, uses default prompt if not present)
    ├── ftheta_camera_front_wide_120fov/      # or camera_front_wide_120fov/
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── ftheta_camera_cross_right_120fov/     # or camera_cross_right_120fov/
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── ftheta_camera_rear_right_70fov/       # or camera_rear_right_70fov/
    ├── ftheta_camera_rear_tele_30fov/        # or camera_rear_tele_30fov/
    ├── ftheta_camera_rear_left_70fov/        # or camera_rear_left_70fov/
    ├── ftheta_camera_cross_left_120fov/      # or camera_cross_left_120fov/
    └── ftheta_camera_front_tele_30fov/       # or camera_front_tele_30fov/

Notes for MADS:
- The videos/ folder is required (input videos)
- The world_scenario/ folder is required (control signal videos)
- The captions/ folder is recommended for best quality; if not present, a preset default driving scene description is used
- Each camera's subfolder name supports two formats: "ftheta_{camera_name}" or "{camera_name}"
- video_id must be consistent across all camera folders in all three directories
- All 7 camera views must have corresponding subfolders and files

Camera view to View Index mapping for MADS:
- camera_front_wide_120fov: 0
- camera_cross_right_120fov: 1
- camera_rear_right_70fov: 2
- camera_rear_tele_30fov: 3
- camera_rear_left_70fov: 4
- camera_cross_left_120fov: 5
- camera_front_tele_30fov: 6


# agibot

Expected directory structure for Agibot dataset:
```
input_root/
├── videos/                                    # Input video folder
│   ├── head_color/                           # Head-mounted camera view
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── hand_left/                            # Left hand camera view
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   └── hand_right/                           # Right hand camera view
│       ├── video_id_1.mp4
│       ├── video_id_2.mp4
│       └── ...
│
├── [control_folder]/                          # Control video folder (e.g., depth, seg, edge, vis)
│   ├── head_color/                           # Head-mounted camera control
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   ├── hand_left/                            # Left hand camera control
│   │   ├── video_id_1.mp4
│   │   ├── video_id_2.mp4
│   │   └── ...
│   └── hand_right/                           # Right hand camera control
│       ├── video_id_1.mp4
│       ├── video_id_2.mp4
│       └── ...
│
└── captions/                                  # Caption folder (recommended, uses default prompt if not present)
    ├── head_color/                           # Head-mounted camera captions
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    ├── hand_left/                            # Left hand camera captions
    │   ├── video_id_1.txt
    │   ├── video_id_2.txt
    │   └── ...
    └── hand_right/                           # Right hand camera captions
        ├── video_id_1.txt
        ├── video_id_2.txt
        └── ...

Notes for Agibot:
- The videos/ folder is required (input videos)
- The control folder is required (control signal videos) - the folder name varies based on the control type
- The captions/ folder is optional; if not present, a default prompt is used
- video_id must be consistent across all camera folders in all three directories
- All 3 camera views must have corresponding subfolders and files
- Unlike MADS, Agibot does not use the "ftheta_" prefix for camera names

Camera view to View Index mapping for Agibot:
- head_color: 0
- hand_left: 1
- hand_right: 2
```

Usage:

# mads
```bash
EXP=transfer2p5_2b_mv_7train7_res480p_fps10_t24_frombase2p5avfinetune_mads_only_allcaption_uniform_nofps_wm_condition_i2v_and_t2v
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv_7train7_res480p_fps10_t24_frombase2p5avfinetune_mads_only_allcaption_uniform_nofps_wm_condition_i2v_and_t2v-0/checkpoints/iter_000010000/

PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/yiflu/project_official_i4/condition_assets/multiview-inference-assets-1203 \
    --num_conditional_frames 1 \
    --guidance 3.0 \
    --save_root results/transfer2_multiview_480p_i2v_grid_eachview/ \
    --max_samples 5 --target_height 480 --target_width 832 \
    --stack_mode grid  \
    --save_each_view \
    model.config.base_load_from=null

# auto-regressive
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/yiflu/project_official_i4/condition_assets/multiview-inference-assets-1204/normal-200frame-10fps \
    --num_conditional_frames 1 \
    --guidance 5.0 \
    --save_root results/transfer2_multiview_480p_i2v_long_grid_eachview/ \
    --max_samples 50 --target_height 480 --target_width 832 \
    --use_autoregressive --target_frames 277 \
    --stack_mode grid \
    --save_each_view \
    model.config.base_load_from=null

# agibot
# depth control
EXP=transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth-0/checkpoints/iter_000038000/

# edge control
EXP=transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge-0/checkpoints/iter_000039000/

# vis control
EXP=transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis-0/checkpoints/iter_000029000/

# seg control
EXP=transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg
ckpt_path=s3://bucket/cosmos_transfer2_multiview/cosmos2p5_mv/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg-0/checkpoints/iter_000026000/

# image to world
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/fangyinw/data/agibot/qa/ \
    --num_conditional_frames 1 \
    --guidance 3.0 \
    --save_root results/transfer2_multiview_720p_i2v/ \
    --max_samples 1 --target_height 720 --target_width 1280 \
    --stack_mode width  \
    --dataset agibot \
    --add_camera_prefix \
    model.config.base_load_from=null

# text to world
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_transfer2._src.transfer2_multiview.inference.inference_cli \
    --experiment ${EXP} \
    --ckpt_path ${ckpt_path} \
    --context_parallel_size 8 \
    --input_root /project/cosmos/fangyinw/data/agibot/qa/ \
    --num_conditional_frames 0 \
    --guidance 3.0 \
    --save_root results/transfer2_multiview_720p_t2v/ \
    --max_samples 1 --target_height 720 --target_width 1280 \
    --stack_mode width  \
    --dataset agibot \
    --add_camera_prefix \
    model.config.base_load_from=null
```
"""

import argparse
import os
from pathlib import Path

import torch as th
import torchvision

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2_multiview.scripts.mv_visualize_helper import (
    arrange_video_visualization,
    save_each_view_separately,
)
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputBlur, AddControlInputEdge
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference

NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"

# Camera name to view index mapping
CAMERA_TO_VIEW_INDEX_MADS = {
    "camera_front_wide_120fov": 0,
    "camera_cross_right_120fov": 1,
    "camera_rear_right_70fov": 2,
    "camera_rear_tele_30fov": 3,
    "camera_rear_left_70fov": 4,
    "camera_cross_left_120fov": 5,
    "camera_front_tele_30fov": 6,
}

CAMERA_TO_VIEW_INDEX_AGIBOT = {
    "head_color": 0,
    "hand_left": 1,
    "hand_right": 2,
}

CAMERA_TO_VIEW_INDEX = {
    "mads": CAMERA_TO_VIEW_INDEX_MADS,
    "agibot": CAMERA_TO_VIEW_INDEX_AGIBOT,
}


# Camera-specific caption prefixes describing camera position and orientation
CAMERA_TO_CAPTION_PREFIX_MADS = {
    "camera_front_wide_120fov": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "camera_cross_right_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "camera_rear_right_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "camera_rear_tele_30fov": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "camera_rear_left_70fov": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "camera_cross_left_120fov": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "camera_front_tele_30fov": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}

CAMERA_TO_CAPTION_PREFIX_AGIBOT = {
    "hand_left": "The video is captured from a camera mounted on the left hand of the subject.",
    "hand_right": "The video is captured from a camera mounted on the right hand of the subject.",
    "head_color": "The video is captured from a camera mounted on the head of the subject, facing forward.",
}
CAMERA_TO_CAPTION_PREFIX = {
    "mads": CAMERA_TO_CAPTION_PREFIX_MADS,
    "agibot": CAMERA_TO_CAPTION_PREFIX_AGIBOT,
}

DEFAULT_DRIVING_SCENE_PROMPT = """
A clear daytime driving scene on an open road. The weather is sunny with bright natural lighting and good visibility.
The sky is partly cloudy with scattered white clouds. The road surface is dry and well-maintained.
The overall atmosphere is calm and peaceful with moderate traffic conditions. The lighting creates clear
shadows and provides excellent contrast for safe navigation."""


def load_video(video_path: str, target_frames: int = 93, target_size: tuple[int, int] = (720, 1280)) -> th.Tensor:
    """
    Load video and process it to target size and frame count.

    Args:
        video_path: Path to video file
        target_frames: Target number of frames
        target_size: Target resolution (H, W)

    Returns:
        Video tensor with shape (C, T, H, W), dtype uint8
    """
    try:
        # Load video using easy_io
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
    video_tensor = th.from_numpy(video_frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    C, T, H, W = video_tensor.shape

    # Adjust frame count: if video is too long, take first target_frames; if too short, pad with last frame
    if T > target_frames:
        video_tensor = video_tensor[:, :target_frames, :, :]
    elif T < target_frames:
        # Pad with last frame
        last_frame = video_tensor[:, -1:, :, :]
        padding_frames = target_frames - T
        last_frame_repeated = last_frame.repeat(1, padding_frames, 1, 1)
        video_tensor = th.cat([video_tensor, last_frame_repeated], dim=1)

    # Convert to uint8: (C, T, H, W) -> (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = (video_tensor * 255.0).to(th.uint8)

    # Adjust resolution
    target_h, target_w = target_size
    if H != target_h or W != target_w:
        # Use resize and center crop
        video_tensor = resize_and_crop(video_tensor, target_size)

    # Convert back to (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)

    return video_tensor


def resize_and_crop(video: th.Tensor, target_size: tuple[int, int]) -> th.Tensor:
    """
    Resize video and center crop.

    Args:
        video: Input video with shape (T, C, H, W)
        target_size: Target resolution (H, W)

    Returns:
        Resized video with shape (T, C, target_H, target_W)
    """
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = target_size

    # Calculate scaling ratio to match the smaller dimension to target
    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(scaling_ratio * orig_h), int(scaling_ratio * orig_w))

    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, target_size)

    return video_cropped


def load_multiview_videos(
    input_root: Path,
    video_id: str,
    camera_order: list[str],
    target_frames: int = 93,
    target_size: tuple[int, int] = (720, 1280),
    folder_name: str = "videos",
    args: argparse.Namespace | None = None,
) -> th.Tensor:
    f"""
    Load multi-view videos from a specified folder.

    Args:
        input_root: Input root directory
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order
        target_frames: Target number of frames per view
        target_size: Target resolution (H, W)
        folder_name: Name of the folder containing videos (e.g., "videos" or control_folder_name)
        args: Arguments namespace
    Returns:
        Multi-view video tensor with shape (C, V*T, H, W)
    """
    if folder_name == "edge":
        add_control_input = AddControlInputEdge(
            input_keys=["video"],
            output_keys=["control_input_edge"],
            use_random=False,
            preset_strength=args.preset_edge_threshold,
        )
        folder_name = "videos"
    elif folder_name == "vis":
        add_control_input = AddControlInputBlur(
            input_keys=["video"],
            output_keys=["control_input_vis"],
            use_random=False,
            downup_preset=args.preset_blur_strength,
        )
        folder_name = "videos"
    else:
        add_control_input = None
    videos_dir = input_root / folder_name
    video_tensors = []

    for camera in camera_order:
        if (videos_dir / f"ftheta_{camera}").exists():
            sub_dir = f"ftheta_{camera}"
        elif (videos_dir / camera).exists():
            sub_dir = camera
        else:
            raise FileNotFoundError(f"Folder not found: {videos_dir / f'ftheta_{camera}'} or {videos_dir / camera}")

        video_path = videos_dir / sub_dir / f"{video_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load single view video: (C, T, H, W)
        video_tensor = load_video(str(video_path), target_frames, target_size)
        # compute on the fly for edge and blur (vis)
        if add_control_input is not None:
            video_tensor = add_control_input({"video": video_tensor})[add_control_input.output_keys[0]]
        video_tensors.append(video_tensor)

    # Concatenate all views: (C, V*T, H, W)
    multiview_video = th.cat(video_tensors, dim=1)

    return multiview_video


def load_multiview_captions(
    input_root: Path,
    video_id: str,
    camera_order: list[str],
    add_camera_prefix: bool = True,
    camera_to_caption_prefix: dict[str, str] = CAMERA_TO_CAPTION_PREFIX_MADS,
) -> list[str]:
    """
    Load multi-view captions. Uses default prompt if captions directory does not exist.

    Note: Captions are strongly recommended for best results as the model was trained with
    meaningful text descriptions. If not provided, a generic fallback is used.

    Args:
        input_root: Input root directory
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order
        add_camera_prefix: Whether to add camera-specific prefix to captions
        camera_to_caption_prefix: Dictionary mapping camera names to caption prefixes
    Returns:
        List of captions, one per view
    """
    captions_dir = input_root / "captions"

    # If captions directory does not exist, use default prompt (not recommended for best quality)
    if not captions_dir.exists():
        log.warning(
            f"Captions directory not found: {captions_dir}. Using default driving scene prompt for all cameras. "
            f"For best results, provide meaningful captions."
        )
        return [DEFAULT_DRIVING_SCENE_PROMPT] * len(camera_order)

    captions = []

    for camera in camera_order:
        if (captions_dir / f"ftheta_{camera}").exists():
            sub_dir = f"ftheta_{camera}"
        elif (captions_dir / camera).exists():
            sub_dir = camera
        else:
            raise FileNotFoundError(f"Folder not found: {captions_dir / f'ftheta_{camera}'} or {captions_dir / camera}")

        caption_filename = f"{sub_dir}/{video_id}.txt"
        caption_path = captions_dir / caption_filename

        if not caption_path.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_path}")

        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Add camera-specific prefix if enabled
        if add_camera_prefix and camera in camera_to_caption_prefix:
            caption = f"{camera_to_caption_prefix[camera]} {caption}"

        captions.append(caption)

    return captions


def construct_data_batch(
    multiview_video: th.Tensor,
    control_video: th.Tensor,
    captions: list[str],
    camera_order: list[str],
    num_conditional_frames: int = 1,
    fps: float = 10.0,
    target_frames_per_view: int = 93,
    camera_to_view_index: dict[str, int] = CAMERA_TO_VIEW_INDEX_MADS,
    hint_keys: str = "hdmap_bbox",
) -> dict:
    """
    Construct data_batch for model inference.

    Args:
        multiview_video: Multi-view input video tensor with shape (C, V*T, H, W)
        control_video: Multi-view control video tensor with shape (C, V*T, H, W)
        captions: List of captions
        camera_order: List of camera names in order
        num_conditional_frames: Number of conditional frames
        fps: Frames per second
        target_frames_per_view: Number of frames per view
        camera_to_view_index: Dictionary mapping camera names to view indices
        hint_keys: Keys for the control input
    Returns:
        data_batch dictionary
    """
    C, VT, H, W = multiview_video.shape
    n_views = len(camera_order)
    T = VT // n_views

    # Add batch dimension: (C, V*T, H, W) -> (1, C, V*T, H, W)
    multiview_video = multiview_video.unsqueeze(0)
    control_video = control_video.unsqueeze(0)

    # Construct correct view_indices based on camera order
    # Each view's T frames all use that view's corresponding view index
    view_indices_list = []
    for camera in camera_order:
        view_idx = camera_to_view_index[camera]
        view_indices_list.extend([view_idx] * T)
    view_indices = th.tensor(view_indices_list, dtype=th.int64).unsqueeze(0)  # (1, V*T)

    # Construct view_indices_selection: view indices of cameras in camera_order
    view_indices_selection = th.tensor(
        [camera_to_view_index[camera] for camera in camera_order], dtype=th.int64
    ).unsqueeze(0)  # (1, n_views)

    # Find position of front_wide_120fov in camera_order as ref_cam_view_idx_sample_position
    ref_cam_position = (
        camera_order.index("camera_front_wide_120fov") if "camera_front_wide_120fov" in camera_order else 0
    )

    # Construct data_batch
    data_batch = {
        "video": multiview_video,
        f"control_input_{hint_keys}": control_video,
        "ai_caption": [captions],
        "view_indices": view_indices,  # (1, V*T), using correct view index
        "fps": th.tensor([fps], dtype=th.float64),
        "chunk_index": th.tensor([0], dtype=th.int64),
        "frame_indices": th.arange(target_frames_per_view).unsqueeze(0),  # (1, T)
        "num_video_frames_per_view": th.tensor([target_frames_per_view], dtype=th.int64),
        "view_indices_selection": view_indices_selection,  # (1, n_views), using correct view index
        "camera_keys_selection": [camera_order],
        "sample_n_views": th.tensor([n_views], dtype=th.int64),
        "padding_mask": th.zeros(1, 1, H, W, dtype=th.float32),
        "ref_cam_view_idx_sample_position": th.tensor([ref_cam_position], dtype=th.int64),
        "front_cam_view_idx_sample_position": [None],
        "original_hw": th.tensor([[[H, W]] * n_views], dtype=th.int64),  # (1, n_views, 2)
        NUM_CONDITIONAL_FRAMES_KEY: num_conditional_frames,
    }

    return data_batch


# New helper function to load multiview videos from direct camera paths
def load_multiview_videos_from_paths(
    camera_paths: dict[str, Path],
    camera_order: list[str],
    target_frames: int = 93,
    target_size: tuple[int, int] = (720, 1280),
    is_control: bool = False,
    control_type: str | None = None,
    args: object | None = None,
) -> th.Tensor:
    """
    Load multi-view videos from direct camera paths.

    Args:
        camera_paths: Dictionary mapping camera names to video file paths
        camera_order: List of camera names in order
        target_frames: Target number of frames per view
        target_size: Target resolution (H, W)
        is_control: Whether these are control videos
        control_type: Type of control (edge, vis, depth, seg) - only used if is_control=True
        args: Arguments namespace for control generation (edge/vis)

    Returns:
        Multi-view video tensor with shape (C, V*T, H, W)
    """
    # Setup control generators if needed
    add_control_input = None
    if is_control and control_type in ["edge", "vis"]:
        if control_type == "edge":
            from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputEdge

            add_control_input = AddControlInputEdge(
                input_keys=["video"],
                output_keys=["control_input_edge"],
                use_random=False,
                preset_strength=args.preset_edge_threshold if args else "medium",
            )
        elif control_type == "vis":
            from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputBlur

            add_control_input = AddControlInputBlur(
                input_keys=["video"],
                output_keys=["control_input_vis"],
                use_random=False,
                downup_preset=args.preset_blur_strength if args else "medium",
            )

    video_tensors = []
    for camera in camera_order:
        video_path = camera_paths[camera]

        if video_path is None:
            if is_control and control_type in ["edge", "vis"]:
                # Will be computed from input video
                continue
            else:
                raise ValueError(f"Missing video path for camera: {camera}")

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load single view video: (C, T, H, W)
        video_tensor = load_video(str(video_path), target_frames, target_size)

        # Compute control on-the-fly for edge/vis
        if add_control_input is not None:
            result = add_control_input({"video": video_tensor})
            if control_type == "edge":
                video_tensor = result["control_input_edge"]
            elif control_type == "vis":
                video_tensor = result["control_input_vis"]

        video_tensors.append(video_tensor)

    # Stack all views: (C, V*T, H, W)
    multiview_video = th.cat(video_tensors, dim=1)
    return multiview_video


def generate_multiview_control_video(
    vid2world_cli: "ControlVideo2WorldInference",
    camera_order: list[str],
    camera_to_view_index: dict[str, int],
    camera_to_caption_prefix: dict[str, str],
    control_type: str,
    target_frames: int,
    target_size: tuple[int, int],
    num_conditional_frames: int,
    fps: float,
    preset_edge_threshold: str,
    preset_blur_strength: str,
    add_camera_prefix: bool,
    guidance: float,
    seed: int,
    num_steps: int,
    use_negative_prompt: bool,
    control_weight: float = 1.0,
    enable_autoregressive: bool = False,
    chunk_overlap: int = 1,
    # New parameters for direct path-based loading (for agibot)
    camera_input_paths: dict[str, Path | None] | None = None,
    camera_control_paths: dict[str, Path | None] | None = None,
    # Old parameters for video_id-based loading (backward compatibility)
    input_root: Path | None = None,
    video_id: str | None = None,
    # Caption loading parameters
    prompt_override: str | None = None,
    input_root_for_captions: Path | None = None,
) -> tuple[th.Tensor, dict]:
    """
    High-level API to generate control-conditioned multiview video.

    This function wraps the complete inference pipeline into a single call:
    1. Load multiview input videos
    2. Load multiview control videos (or generate on-the-fly for edge/vis)
    3. Load multiview captions
    4. Construct data batch
    5. Run inference

    This provides a clean interface similar to Video2WorldInference.generate_vid2world()
    for control-conditioned models.

    Args:
        vid2world_cli: ControlVideo2WorldInference instance
        input_root: Root directory containing videos/, captions/ folders
        video_id: Video ID (filename without extension)
        camera_order: List of camera names in order
        camera_to_view_index: Dictionary mapping camera names to view indices
        camera_to_caption_prefix: Dictionary mapping camera names to caption prefixes
        control_type: Control type (depth, edge, vis, seg)
        target_frames: Number of frames per view (chunk size for autoregressive mode)
        target_size: (height, width) tuple
        num_conditional_frames: Number of conditional frames
        fps: Frames per second
        preset_edge_threshold: Edge detection threshold preset ("very_low", "low", "medium", "high", "very_high")
        preset_blur_strength: Blur strength preset ("very_low", "low", "medium", "high", "very_high")
        add_camera_prefix: Whether to add camera-specific prefix to captions
        guidance: Classifier-free guidance scale
        seed: Random seed
        num_steps: Number of diffusion steps
        use_negative_prompt: Whether to use negative prompt
        control_weight: Control signal weight (default 1.0)
        enable_autoregressive: Enable autoregressive generation for longer videos (default False)
        chunk_overlap: Number of overlapping frames between chunks in autoregressive mode (default 1)

    Returns:
        Tuple of (generated_video, data_batch)
        - generated_video: Generated video tensor on CPU with shape (1, C, V*T, H, W)
        - data_batch: Data batch dictionary used for inference
    """

    # Create args namespace for control generators (edge/vis)
    class Args:
        pass

    args = Args()
    args.preset_edge_threshold = preset_edge_threshold
    args.preset_blur_strength = preset_blur_strength

    # Determine which loading mode to use
    if camera_input_paths is not None:
        # New path-based loading mode (for agibot)

        # For edge/vis controls, pass input video paths (control computed on-the-fly)
        # For depth/seg controls, use the provided control paths
        if control_type in ["edge", "vis"]:
            control_paths_to_use = camera_input_paths  # Will generate control from input
        else:
            if camera_control_paths is None:
                raise ValueError(f"camera_control_paths required for control_type='{control_type}'")
            control_paths_to_use = camera_control_paths

        # T2V optimization: if num_conditional_frames=0 and control type is depth/seg,
        # input videos are optional - use control videos as mock input
        if num_conditional_frames == 0 and control_type in ["depth", "seg"]:
            # Check if input paths are actually provided
            has_input_paths = all(
                camera_input_paths.get(cam) is not None and Path(camera_input_paths[cam]).exists()
                for cam in camera_order
            )
            if not has_input_paths:
                log.info(
                    f"T2V mode with {control_type} control: using control videos as mock input "
                    f"(input videos not provided or not found)"
                )
                camera_input_paths = control_paths_to_use

        multiview_video = load_multiview_videos_from_paths(
            camera_paths=camera_input_paths,
            camera_order=camera_order,
            target_frames=target_frames,
            target_size=target_size,
            is_control=False,
            control_type=None,
            args=args,
        )

        control_video = load_multiview_videos_from_paths(
            camera_paths=control_paths_to_use,
            camera_order=camera_order,
            target_frames=target_frames,
            target_size=target_size,
            is_control=True,
            control_type=control_type,
            args=args,
        )

        # For path-based mode, handle captions with priority:
        # 1. prompt_override (if provided)
        # 2. caption files (if input_root_for_captions provided)
        # 3. camera prefix (fallback)
        if prompt_override:
            # Use the override prompt for all cameras
            captions = [prompt_override] * len(camera_order)
        elif input_root_for_captions is not None:
            # Try to load captions from files
            # Derive video_id from the first input path (assume all cameras have same base name)
            first_input_path = camera_input_paths[camera_order[0]]
            # Extract video_id: remove camera suffix and file extension
            # e.g., "296_656371_chunk0_head_color_rgb.mp4" -> "296_656371_chunk0"
            video_id_from_path = first_input_path.stem
            for cam in camera_order:
                # Remove camera-specific suffix (e.g., "_head_color_rgb", "_hand_left_rgb")
                video_id_from_path = video_id_from_path.replace(f"_{cam}_rgb", "").replace(f"_{cam}", "")

            captions_dir = input_root_for_captions / "captions"
            if captions_dir.exists():
                captions = []
                for camera in camera_order:
                    caption_path = captions_dir / f"{video_id_from_path}_{camera}.txt"
                    if caption_path.exists():
                        with open(caption_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                        # Add camera-specific prefix if enabled
                        if add_camera_prefix and camera in camera_to_caption_prefix:
                            caption = f"{camera_to_caption_prefix[camera]} {caption}"
                        captions.append(caption)
                    else:
                        # Caption file not found, use camera prefix as fallback
                        if add_camera_prefix:
                            captions.append(f"{camera_to_caption_prefix[camera]} ")
                        else:
                            captions.append("")
                        log.warning(f"Caption file not found: {caption_path}. Using camera prefix as fallback.")
            else:
                # Captions directory doesn't exist, use camera prefix
                log.warning(
                    f"Captions directory not found: {captions_dir}. Using camera prefix as fallback. "
                    f"For best results, provide captions."
                )
                if add_camera_prefix:
                    captions = [f"{camera_to_caption_prefix[cam]} " for cam in camera_order]
                else:
                    captions = [""] * len(camera_order)
        else:
            # No prompt override or caption loading - use camera prefix only
            if add_camera_prefix:
                captions = [f"{camera_to_caption_prefix[cam]} " for cam in camera_order]
            else:
                captions = [""] * len(camera_order)
    else:
        # Old video_id-based loading mode (backward compatibility)
        if input_root is None or video_id is None:
            raise ValueError(
                "Either (camera_input_paths, camera_control_paths) or (input_root, video_id) must be provided"
            )

        # Load multiview input videos (C, V*T, H, W)
        multiview_video = load_multiview_videos(
            input_root=input_root,
            video_id=video_id,
            camera_order=camera_order,
            target_frames=target_frames,
            target_size=target_size,
            folder_name="videos",
            args=args,
        )

        # Load multiview control videos (C, V*T, H, W)
        # For edge/vis: generated on-the-fly from input videos
        # For depth/seg: loaded from disk
        control_video = load_multiview_videos(
            input_root=input_root,
            video_id=video_id,
            camera_order=camera_order,
            target_frames=target_frames,
            target_size=target_size,
            folder_name=control_type,
            args=args,
        )

        # Load multiview captions
        captions = load_multiview_captions(
            input_root=input_root,
            video_id=video_id,
            camera_order=camera_order,
            add_camera_prefix=add_camera_prefix,
            camera_to_caption_prefix=camera_to_caption_prefix,
        )

    # Construct data batch
    data_batch = construct_data_batch(
        multiview_video=multiview_video,
        control_video=control_video,
        captions=captions,
        camera_order=camera_order,
        num_conditional_frames=num_conditional_frames,
        fps=fps,
        target_frames_per_view=target_frames,
        camera_to_view_index=camera_to_view_index,
        hint_keys=control_type,
    )

    # Add control weight
    data_batch["control_weight"] = control_weight

    # Run inference
    if enable_autoregressive:
        # Get chunk size from model config
        chunk_size = vid2world_cli.model.tokenizer.get_pixel_num_frames(vid2world_cli.model.config.state_t)
        n_views = len(camera_order)

        # Get hint_keys (control type) from model config or use control_type
        hint_keys = getattr(vid2world_cli.config.model.config, "hint_keys", control_type)

        # Remove num_conditional_frames from batch - it should be passed as an argument
        # to generate_autoregressive_from_batch, not included in the batch
        if "num_conditional_frames" in data_batch:
            del data_batch["num_conditional_frames"]

        video, control = vid2world_cli.generate_autoregressive_from_batch(
            data_batch,
            n_views=n_views,
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
            guidance=guidance,
            seed=seed,
            num_conditional_frames=num_conditional_frames,
            num_steps=num_steps,
            use_negative_prompt=use_negative_prompt,
            hint_keys=hint_keys,
        )
        # Add batch dimension for consistency with non-autoregressive path
        video = video.unsqueeze(0).cpu()
    else:
        video = vid2world_cli.generate_from_batch(
            data_batch,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            use_negative_prompt=use_negative_prompt,
        ).cpu()

    return video, data_batch


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transfer2 Multiview inference from videos, control videos, and captions"
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # Generation parameters
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance value")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Number of conditional frames")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of diffusion steps")
    parser.add_argument(
        "--use_negative_prompt",
        action="store_true",
        default=True,
        help="Use default negative prompt for additional guidance.",
    )
    parser.add_argument("--control_weight", type=float, default=1.0, help="Control weight")
    # Input/output
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Input root directory containing videos/, {control_folder_name}/, and captions/ subdirectories",
    )
    parser.add_argument("--save_root", type=str, default="results/transfer2_multiview_cli/", help="Save root")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples to generate")
    parser.add_argument(
        "--stack_mode",
        type=str,
        default="width",
        choices=["height", "width", "time", "grid"],
        help="Video stacking mode for visualization. grid will create a 3x3 grid of views.",
    )
    # Video parameters
    parser.add_argument("--target_frames", type=int, default=93, help="Target number of frames per view")
    parser.add_argument("--target_height", type=int, default=720, help="Target video height")
    parser.add_argument("--target_width", type=int, default=1280, help="Target video width")
    # Caption parameters
    parser.add_argument(
        "--add_camera_prefix",
        action="store_true",
        default=True,
        help="Add camera-specific position/orientation prefix to captions",
    )
    parser.add_argument(
        "--no_camera_prefix",
        action="store_false",
        dest="add_camera_prefix",
        help="Do not add camera-specific prefix to captions",
    )
    # Save options
    parser.add_argument(
        "--save_each_view",
        action="store_true",
        help="Save each camera view as a separate video file",
    )
    # Autoregressive generation
    parser.add_argument(
        "--use_autoregressive",
        action="store_true",
        help="Use autoregressive generation for long videos",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=1,
        help="Number of overlapping frames between chunks in autoregressive generation",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the inference.")
    parser.add_argument("--hierarchical_cp", action="store_true", help="Use hierarchical CP algorithm (a2a + p2p)")
    parser.add_argument("--dataset", type=str, default="mads", choices=["mads", "agibot"], help="Dataset")
    parser.add_argument(
        "--preset_edge_threshold",
        type=str,
        default="medium",
        choices=["very_low", "low", "medium", "high", "very_high"],
        help="Preset strength for the canny edge detection",
    )
    # Experiment options
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pairs.
        For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


def main():
    os.environ["NVTE_FUSED_ATTN"] = "0"
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.enable_grad(False)

    args = parse_arguments()

    camera_to_view_index = CAMERA_TO_VIEW_INDEX[args.dataset]
    DEFAULT_CAMERA_ORDER = list(camera_to_view_index.keys())
    # Prepare experiment options
    experiment_opts = list(args.opts) if args.opts else []
    if args.use_cuda_graphs:
        experiment_opts.append("model.config.net.use_cuda_graphs=True")
    if args.hierarchical_cp:
        experiment_opts.append("model.config.net.atten_backend='transformer_engine'")

    # Initialize inference handler
    vid2world_cli = ControlVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
        hierarchical_cp=args.hierarchical_cp,
        experiment_opts=experiment_opts,
    )
    mem_bytes = th.cuda.memory_allocated(device=th.device("cuda" if th.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    input_root = Path(args.input_root)
    videos_dir = input_root / "videos"
    hint_keys = vid2world_cli.config.model.config.hint_keys
    control_folder_name = "world_scenario" if args.dataset == "mads" else hint_keys

    # Create output directory
    save_root = f"{args.save_root}/{hint_keys}"
    os.makedirs(save_root, exist_ok=True)

    # Verify required directories exist
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    # Get all video IDs (from first camera directory)
    if (videos_dir / f"ftheta_{DEFAULT_CAMERA_ORDER[0]}").exists():
        first_camera_dir = videos_dir / f"ftheta_{DEFAULT_CAMERA_ORDER[0]}"
    else:
        first_camera_dir = videos_dir / DEFAULT_CAMERA_ORDER[0]

    video_files = sorted(first_camera_dir.glob("*.mp4"))
    video_ids = [f.stem for f in video_files[: args.max_samples]]

    log.info(f"Found {len(video_ids)} video IDs, processing {min(len(video_ids), args.max_samples)} samples")

    for i, video_id in enumerate(video_ids):
        if rank0:
            log.info(f"Processing sample {i + 1}/{len(video_ids)}: {video_id}")

        try:
            # Load multi-view input videos
            multiview_video = load_multiview_videos(
                input_root,
                video_id,
                DEFAULT_CAMERA_ORDER,
                target_frames=args.target_frames,
                target_size=(args.target_height, args.target_width),
                folder_name="videos",
                args=args,
            )
            if rank0:
                log.info(f"Loaded input multiview video: {multiview_video.shape}")

            # Load multi-view control videos ({control_folder_name})
            control_video = load_multiview_videos(
                input_root,
                video_id,
                DEFAULT_CAMERA_ORDER,
                target_frames=args.target_frames,
                target_size=(args.target_height, args.target_width),
                folder_name=control_folder_name,
                args=args,
            )
            if rank0:
                log.info(f"Loaded control video ({control_folder_name}): {control_video.shape}")

            # Load multi-view captions
            captions = load_multiview_captions(
                input_root,
                video_id,
                DEFAULT_CAMERA_ORDER,
                add_camera_prefix=args.add_camera_prefix,
                camera_to_caption_prefix=CAMERA_TO_CAPTION_PREFIX[args.dataset],
            )

            if rank0:
                log.info(f"Loaded {len(captions)} captions")
                log.info(f"First caption preview: {captions[0][:100]}...")

            # Construct data_batch
            data_batch = construct_data_batch(
                multiview_video,
                control_video,
                captions,
                DEFAULT_CAMERA_ORDER,
                num_conditional_frames=args.num_conditional_frames,
                fps=args.fps,
                target_frames_per_view=args.target_frames,
                camera_to_view_index=camera_to_view_index,
                hint_keys=hint_keys,
            )

            # Add control weight
            data_batch["control_weight"] = args.control_weight

            # Run inference
            if args.use_autoregressive:
                # Use autoregressive generation
                import time

                th.cuda.synchronize()
                start_time = time.time()

                video, control = vid2world_cli.generate_autoregressive_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + i,
                    num_conditional_frames=args.num_conditional_frames,
                    num_steps=args.num_steps,
                    n_views=len(DEFAULT_CAMERA_ORDER),
                    chunk_size=vid2world_cli.model.tokenizer.get_pixel_num_frames(vid2world_cli.model.config.state_t),
                    chunk_overlap=args.chunk_overlap,
                    use_negative_prompt=args.use_negative_prompt,
                )

                th.cuda.synchronize()
                end_time = time.time()
                if rank0:
                    log.info(f"Time taken for autoregressive generation: {end_time - start_time:.2f} seconds")

                # Add batch dimension for saving
                video = video.unsqueeze(0)
                control = control.unsqueeze(0)

                # Apply visualization layout
                video_arranged = arrange_video_visualization(video, data_batch, method=args.stack_mode)
                control_arranged = arrange_video_visualization(control, data_batch, method=args.stack_mode)

                if rank0:
                    video_path = f"{save_root}/inference_{video_id}_video"
                    save_img_or_video(video_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved video to {video_path}")

                    video_path = f"{save_root}/inference_{video_id}_control"
                    save_img_or_video(control_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved control to {video_path}")

                    # Save each view separately if requested (only generated video, not control)
                    if args.save_each_view:
                        save_dir = f"{save_root}/inference_{video_id}"
                        save_each_view_separately(
                            mv_video=video[0],
                            data_batch=data_batch,
                            save_dir=save_dir,
                            fps=args.fps,
                        )

            else:
                # Extract control video from data_batch for saving
                control_video = (
                    data_batch[f"control_input_{hint_keys}"].float() / 255.0
                ).cpu()  # (1, 3, V*T, H, W), convert to [0,1]

                # Use single-shot generation
                video = vid2world_cli.generate_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + i,
                    num_steps=args.num_steps,
                    use_negative_prompt=args.use_negative_prompt,
                ).cpu()

                # Apply visualization layout
                video_arranged = arrange_video_visualization(
                    video, data_batch, method=args.stack_mode, dataset=args.dataset
                )
                control_arranged = arrange_video_visualization(
                    control_video, data_batch, method=args.stack_mode, dataset=args.dataset
                )

                # Save results
                if rank0:
                    video_path = f"{save_root}/inference_{video_id}_video"
                    save_img_or_video(video_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved video to {video_path}")

                    video_path = f"{save_root}/inference_{video_id}_control"
                    save_img_or_video(control_arranged[0], video_path, fps=args.fps)
                    log.info(f"Saved control to {video_path}")

                    # Save each view separately if requested (only generated video, not control)
                    if args.save_each_view:
                        save_dir = f"{save_root}/inference_{video_id}"
                        save_each_view_separately(
                            mv_video=video[0],
                            data_batch=data_batch,
                            save_dir=save_dir,
                            fps=args.fps,
                        )

        except Exception as e:
            log.error(f"Error processing {video_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Synchronize all processes
    if args.context_parallel_size > 1:
        th.distributed.barrier()

    # Cleanup distributed resources
    vid2world_cli.cleanup()


if __name__ == "__main__":
    main()
