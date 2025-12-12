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


from pathlib import Path

import click
from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import load_scene
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.config import DEFAULT_CAMERA_NAMES, SETTINGS
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local import (
    convert_scene_data_for_rendering,
    render_multi_camera_tiled,
)


@click.command()
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing scene data (ClipGT or RDS-HQ format)",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save generated control videos",
)
@click.option(
    "--cameras",
    default="all",
    help='Comma-separated camera names or "all" for all cameras (default: all)',
)
@click.option(
    "--clip-id",
    default=None,
    type=str,
    help="Specific clip ID to process (for RDS-HQ format with multiple clips)",
)
def main(input_dir: Path, output_dir: Path, cameras: str, clip_id: str | None) -> None:
    """Render world scenario videos for all cameras in the scene."""
    logger.info(f"Loading data from: {input_dir}")
    data_path = input_dir

    # Parse camera names
    if cameras.strip().lower() == "all":
        camera_names = list(DEFAULT_CAMERA_NAMES)
    else:
        camera_names = [c.strip() for c in cameras.split(",")]

    logger.info(f"Rendering {len(camera_names)} camera(s): {', '.join(camera_names)}")

    scene_data = load_scene(
        data_path,
        camera_names=camera_names,
        max_frames=-1,
        input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
        resize_resolution_hw=SETTINGS["RESIZE_RESOLUTION"],
        clip_id=clip_id,
    )

    logger.info(f"Loaded scene {scene_data.scene_id}:")
    logger.info(f"  - Frames: {scene_data.num_frames}")
    logger.info(f"  - Dynamic objects: {len(scene_data.dynamic_objects)}")
    logger.info("  - Map elements loaded")
    logger.info(f"  - Cameras: {', '.join(camera_names)}")

    # Convert to rendering format
    all_camera_models, all_camera_poses = convert_scene_data_for_rendering(
        scene_data,
        camera_names,
        SETTINGS["RESIZE_RESOLUTION"],
    )

    # Render using simplified output structure
    render_multi_camera_tiled(
        all_camera_models,
        all_camera_poses,
        scene_data,
        camera_names,
        str(output_dir),
        scene_data.scene_id,
        max_frames=-1,
        chunk_output=False,
        overlay_camera=False,
        alpha=0.5,
        clipgt_path=data_path,
        use_persistent_vbos=True,
        multi_sample=4,
        simplified_output=True,
    )


if __name__ == "__main__":
    main()
