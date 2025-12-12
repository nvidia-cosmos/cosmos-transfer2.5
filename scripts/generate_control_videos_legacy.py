#!/usr/bin/env python3
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
Standalone script to generate control videos for Transfer2 from ClipGT data.

Usage:
    python generate_control_videos.py input_rootectory save_rootectory [options]

Example:
    python generate_control_videos.py /path/to/clip_data/ ./output_videos/
"""

import sys
from pathlib import Path

import click
import imageio
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from av_utils.data_model import SceneFiles
from av_utils.render_config import CAMERA_NAME_MAPPING, SETTINGS
from render_hd_map import load_and_interpolate, render_hdmap


def save_control_video(
    rendered_frames: np.ndarray, save_root: Path, clip_id: str, camera_name: str, fps: int = 30
) -> Path:
    """Save rendered frames as MP4 video with proper naming convention."""
    # Create clip directory
    clip_save_root = save_root / clip_id
    clip_save_root.mkdir(parents=True, exist_ok=True)

    # Convert camera name format: camera:front:wide:120fov -> camera_front_wide_120fov
    camera_alias = camera_name.replace(":", "_")

    # Generate filename: clip_id.camera_alias.mp4
    output_file = clip_save_root / f"{clip_id}.{camera_alias}.mp4"

    # Ensure video dimensions are even (required for x264)
    height, width = rendered_frames.shape[1:3]
    if height % 2 != 0:
        rendered_frames = rendered_frames[:, :-1, :]
    if width % 2 != 0:
        rendered_frames = rendered_frames[:, :, :-1]

    # Save video
    writer = imageio.get_writer(
        str(output_file),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
    )

    for frame in rendered_frames:
        writer.append_data(frame)
    writer.close()

    return output_file


@click.command()
@click.argument("input_root", type=click.Path(exists=True, path_type=Path))
@click.argument("save_root", type=click.Path(path_type=Path))
@click.option("--cameras", default="all", help='Comma-separated camera names or "all" for all cameras (default: all)')
def main(input_root: Path, save_root: Path, cameras: str):
    """Generate control videos from local ClipGT data.

    INPUT_ROOT: Directory containing ClipGT parquet files
    SAVE_ROOT: Directory to save generated control videos

    Required files:
    - {clip_id}.obstacle.parquet
    - {clip_id}.calibration_estimate.parquet
    - {clip_id}.egomotion_estimate.parquet

    Optional files enhance rendering quality:
    - {clip_id}.lane.parquet, {clip_id}.lane_line.parquet, etc.
    """

    # Parse and validate camera names
    if cameras.strip().lower() == "all":
        camera_names = list(CAMERA_NAME_MAPPING.keys())
        logger.info("Processing all available cameras")
    else:
        camera_names = [c.strip() for c in cameras.split(",")]

    valid_cameras = list(CAMERA_NAME_MAPPING.keys())
    for camera in camera_names:
        if camera not in valid_cameras:
            logger.error(f"Invalid camera: {camera}")
            logger.info(f"Valid options: {valid_cameras} or 'all'")
            raise click.BadParameter(f"Invalid camera: {camera}")

    save_root.mkdir(parents=True, exist_ok=True)

    # Detect clip ID from directory name or parquet files
    clip_id = input_root.name
    parquet_files = list(input_root.glob("*.parquet"))
    if parquet_files:
        # Look for required parquet files to determine clip_id
        def get_clip_id(files, default_clip_id):
            required_suffix = [".obstacle.parquet", ".calibration_estimate.parquet", ".egomotion_estimate.parquet"]
            for file in files:
                for suffix in required_suffix:
                    if file.name.endswith(suffix):
                        return file.name.replace(suffix, "")
            return default_clip_id

        clip_id = get_clip_id(parquet_files, clip_id)

    logger.info(f"Processing clip: {clip_id}")
    logger.info(f"Cameras: {camera_names}")

    # Validate required files
    required_files = {
        "obstacle_file": input_root / f"{clip_id}.obstacle.parquet",
        "calibration_estimate_file": input_root / f"{clip_id}.calibration_estimate.parquet",
        "egomotion_estimate_file": input_root / f"{clip_id}.egomotion_estimate.parquet",
    }

    missing_files = [str(path) for name, path in required_files.items() if not path.exists()]
    if missing_files:
        logger.error("Required files missing:")
        for f in missing_files:
            logger.error(f"  - {f}")
        raise click.FileError("Required parquet files not found")

    # Add optional files if they exist
    optional_files = [
        "lane_file",
        "lane_line_file",
        "road_boundary_file",
        "crosswalk_file",
        "pole_file",
        "road_marking_file",
        "wait_line_file",
        "traffic_light_file",
        "traffic_sign_file",
    ]

    scene_files_dict = required_files.copy()
    for file_type in optional_files:
        file_name = file_type.replace("_file", "")
        file_path = input_root / f"{clip_id}.{file_name}.parquet"
        scene_files_dict[file_type] = file_path if file_path.exists() else None

    scene_files = SceneFiles(**scene_files_dict)

    # Log discovered files
    logger.info("Found files:")
    for name, path in scene_files_dict.items():
        if path and path.exists():
            logger.info(f"  ✓ {name}: {path.name}")
        else:
            logger.info(f"  ✗ {name}: not found")

    try:
        logger.info("Loading and interpolating data...")
        camera_models, camera_poses, all_object_info, map_data = load_and_interpolate(
            scene_files, camera_names, SETTINGS["RESIZE_RESOLUTION"], SETTINGS["INPUT_POSE_FPS"]
        )
        logger.info(f"Loaded {len(camera_models)} camera models")
        logger.info(f"Processing {len(all_object_info)} frames of object data")

        # Render each camera
        for i, camera_name in enumerate(camera_names):
            logger.info(f"Rendering camera {i + 1}/{len(camera_names)}: {camera_name}")

            if camera_name not in camera_models:
                logger.warning(f"Camera {camera_name} not found in calibration data, skipping")
                continue

            rendered_frames = render_hdmap(
                camera_poses[camera_name],
                all_object_info,
                map_data,
                camera_models[camera_name],
                render_version="v3",
            )

            logger.info(f"Rendered {len(rendered_frames)} frames for {camera_name}")

            # Save video with new format
            output_file = save_control_video(
                rendered_frames, save_root, clip_id, camera_name, fps=SETTINGS["TARGET_RENDER_FPS"]
            )
            logger.info(f"Saved: {output_file}")

        logger.info("Processing completed successfully!")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.error("This might be due to:")
        logger.error("  - Missing or corrupted parquet files")
        logger.error("  - Incompatible data format")
        logger.error("  - Missing graphics dependencies (ModernGL, OpenGL)")
        raise


if __name__ == "__main__":
    main()
