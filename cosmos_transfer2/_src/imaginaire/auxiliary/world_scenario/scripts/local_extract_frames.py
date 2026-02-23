# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
HD Map Rendering and Frame Extraction for Autonomous Vehicle Data.

This module provides functions for rendering HD maps from ClipGT or RDS-HQ format data,
extracting frames with 3D bounding box annotations, and overlaying camera views on
rendered maps. It supports both single and multi-camera rendering with optional video
overlays and chunked output.

Features:
    - HD map rendering with 3D bounding box visualization
    - Multi-camera tiled rendering for performance
    - Frame extraction with 3D annotations (JSON format)
    - Camera video overlay on rendered maps
    - Chunked video output for long sequences
    - Support for novel camera poses from tar files
    - Frustum culling for efficient annotation extraction

Prerequisites:
    - OpenGL-capable GPU
    - FFmpeg installed and available in PATH
    - Python packages: numpy, imageio, click, loguru, scipy

Usage Examples:
    # Render HD map video for a single camera
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /path/to/data --camera-names camera_front_wide_120fov

    # Extract frames with 3D annotations (every 10th frame)
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /path/to/data \\
        --camera-names camera_front_wide_120fov \\
        --extract-frames \\
        --skip-frames 10

    # Render with camera overlay (50% alpha)
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /path/to/data \\
        --camera-names camera_front_wide_120fov \\
        --overlay-camera \\
        --alpha 0.5

    # Render chunked videos (300 frames per chunk)
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /path/to/data \\
        --camera-names camera_front_wide_120fov \\
        --chunk-output

    # Use novel camera poses from tar file
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /path/to/data \\
        --camera-names camera_front_wide_120fov \\
        --novel-pose-tar /path/to/poses.tar

    # Render all available cameras
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /path/to/data --camera-names all

    # Download from S3, process, and upload to S3 (uses default /tmp/local_data)
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \\
        --s3-input s3://bucket/input/path \\
        --s3-output s3://bucket/output/path \\
        --extract-frames

    # Different profiles for input and output (defaults already set)
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \\
        --s3-input s3://bucket1/input \\
        --s3-output s3://bucket2/output \\
        --extract-frames

    # Use custom local data directory
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /custom/path/to/data \\
        --extract-frames

    # Process with S3 input/output and custom endpoints
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /tmp/local_data \\
        --s3-input s3://bucket/input \\
        --s3-output s3://bucket/output \\
        --s3-endpoint-url https://custom-s3.example.com \\
        --no-keep-local

    # Different endpoints for input and output
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py /tmp/local_data \\
        --s3-input s3://bucket1/input \\
        --s3-output s3://bucket2/output \\
        --s3-input-endpoint-url https://input-s3.example.com \\
        --s3-output-endpoint-url https://output-s3.example.com \\
        --s3-input-profile input-profile \\
        --s3-output-profile output-profile

Output Formats:
    - Video rendering: Creates MP4 files with HD map visualization
    - Frame extraction: Creates JPG images and JSON annotation files
    - Chunked output: Multiple video files for long sequences
    - Annotation format: JSON with frame_id, camera, and bbox_3d arrays

Data Format Support:
    - ClipGT format: Original format with video files at root
    - RDS-HQ/MADS format: Videos in ftheta_<camera_name>/ subdirectories

S3 Support:
    - Input: Use --s3-input to download data from S3 before processing
    - Output: Use --s3-output to upload results to S3 after processing
    - Profile: Use --aws-profile for default profile (applies to both if not overridden)
    - Separate Profiles: Use --s3-input-profile and --s3-output-profile for different credentials
    - Endpoint: Use --s3-endpoint-url for default endpoint (applies to both if not overridden)
    - Separate Endpoints: Use --s3-input-endpoint-url and --s3-output-endpoint-url for different endpoints
    - Cleanup: Use --no-keep-local to remove temporary files after upload

Batch Processing:
    - Sequence IDs File: Use --sequence-ids-file to process multiple sequences from a file
    - Base Path: Use --s3-input-base-path to specify the base S3 path (sequence IDs are appended)
    - Consolidation: Outputs from all sequences are automatically consolidated into a single output directory
    - File Format: One UUID per line in the sequence IDs file (e.g., 000a3258-1d6e-11ed-a342-00044bf65f77)


Example Usage:
    # Single sequence processing
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    --s3-input s3://debug/MADS/control_annotation_10/000a3258-1d6e-11ed-a342-00044bf65f77/ \
    --s3-output s3://cosmos_understanding/benchmark/3d_grounding_av/v02_benchmark/ \
    --s3-input-profile team-cosmos-benchmark \
    --s3-output-profile team-cosmos \
    --s3-output-endpoint-url https://pdx.s8k.io \
    --skip-frames 30 \
    --extract-frames

    # Process multiple sequences from file and consolidate outputs
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    --sequence-ids-file sequence_ids.txt \
    --s3-input-base-path s3://debug/MADS/v1/control_annotation_10/ \
    --s3-output s3://cosmos_understanding/benchmark/3d_grounding_av/v02_benchmark/ \
    --s3-input-profile team-cosmos-benchmark \
    --s3-output-profile team-cosmos \
    --s3-output-endpoint-url https://pdx.s8k.io \
    --skip-frames 30 \
    --extract-frames 
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import click
import imageio
import numpy as np
from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import load_scene
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import ObjectType, SceneData
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.config import SETTINGS
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.tiled_multi_camera_renderer import (
    TiledMultiCameraRenderer,
)

# Import common functions from local.py to avoid duplication
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local import (
    convert_scene_data_for_rendering,
    override_camera_poses_with_tar,
    read_video_simple,
)

# Import overlay rendering functionality
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local_render_frames import (
    OverlayRenderer,
    create_overlay_renderer,
    load_camera_videos,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera


def load_sequence_ids_from_file(file_path: str) -> list[str]:
    """
    Load sequence IDs from a text file (one UUID per line).

    Args:
        file_path: Path to the file containing UUIDs, one per line

    Returns:
        List of sequence IDs (UUIDs) found in the file, normalized to lowercase
    """
    sequence_ids: list[str] = []
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Strip whitespace and skip empty lines
                uuid = line.strip()
                if uuid:
                    # Validate UUID format
                    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", uuid, re.IGNORECASE):
                        sequence_ids.append(uuid.lower())
                    else:
                        logger.warning(f"Invalid UUID format at line {line_num} in {file_path}: {uuid}")
        return sequence_ids
    except FileNotFoundError:
        logger.error(f"Sequence IDs file not found: {file_path}")
        raise


def consolidate_outputs(output_dir: str, sequence_outputs: list[str]) -> None:
    """
    Consolidate outputs from multiple sequence processing runs.

    Merges images/, text/, and meta.json from multiple sequence outputs into
    a single consolidated output directory.

    Args:
        output_dir: Final consolidated output directory
        sequence_outputs: List of output directories from individual sequences
    """
    consolidated_path = Path(output_dir)
    consolidated_path.mkdir(parents=True, exist_ok=True)

    consolidated_images_dir = consolidated_path / "images"
    consolidated_text_dir = consolidated_path / "text"
    consolidated_images_dir.mkdir(parents=True, exist_ok=True)
    consolidated_text_dir.mkdir(parents=True, exist_ok=True)

    all_meta_entries = []
    total_images = 0
    total_text = 0

    for seq_output in sequence_outputs:
        seq_path = Path(seq_output)
        if not seq_path.exists():
            logger.warning(f"Sequence output directory does not exist: {seq_output}")
            continue

        # Copy images
        seq_images_dir = seq_path / "images"
        if seq_images_dir.exists():
            for img_file in seq_images_dir.glob("*.jpg"):
                dest_img = consolidated_images_dir / img_file.name
                if not dest_img.exists():  # Avoid overwriting if same filename
                    shutil.copy2(img_file, dest_img)
                    total_images += 1
                else:
                    logger.warning(f"Duplicate image filename, skipping: {img_file.name}")

        # Copy text files
        seq_text_dir = seq_path / "text"
        if seq_text_dir.exists():
            for text_file in seq_text_dir.glob("*.json"):
                dest_text = consolidated_text_dir / text_file.name
                if not dest_text.exists():  # Avoid overwriting if same filename
                    shutil.copy2(text_file, dest_text)
                    total_text += 1
                else:
                    logger.warning(f"Duplicate text filename, skipping: {text_file.name}")

        # Merge meta.json entries
        seq_meta_file = seq_path / "meta.json"
        if seq_meta_file.exists():
            try:
                with open(seq_meta_file, "r") as f:
                    seq_meta_entries = json.load(f)
                    if isinstance(seq_meta_entries, list):
                        all_meta_entries.extend(seq_meta_entries)
            except Exception as e:
                logger.warning(f"Failed to load meta.json from {seq_output}: {e}")

    # Write consolidated meta.json
    consolidated_meta_path = consolidated_path / "meta.json"
    with open(consolidated_meta_path, "w") as f:
        json.dump(all_meta_entries, f, indent=2)

    logger.info(f"Consolidated outputs:")
    logger.info(f"  - Total images: {total_images}")
    logger.info(f"  - Total text files: {total_text}")
    logger.info(f"  - Total meta entries: {len(all_meta_entries)}")
    logger.info(f"  - Consolidated directory: {consolidated_path}")


def process_multiple_sequences(
    sequence_ids: list[str],
    s3_input_base_path: str,
    camera_names: list[str],
    output_dir: str,
    max_frames: int,
    chunk_output: bool,
    overlay_camera: bool,
    alpha: float,
    use_persistent_vbos: bool,
    multi_sample: int,
    novel_pose_tar: Optional[str],
    extract_frames: bool,
    skip_frames: int,
    s3_output: Optional[str],
    aws_profile: Optional[str],
    s3_input_profile: Optional[str],
    s3_output_profile: Optional[str],
    s3_endpoint_url: Optional[str],
    s3_input_endpoint_url: Optional[str],
    s3_output_endpoint_url: Optional[str],
    keep_local: bool,
) -> None:
    """
    Process multiple sequences from a list of sequence IDs.

    Args:
        sequence_ids: List of sequence IDs (UUIDs) to process
        s3_input_base_path: Base S3 path (sequence IDs will be appended)
        All other parameters are the same as main() function
    """
    # Determine profiles and endpoints
    input_profile = s3_input_profile if s3_input_profile is not None else aws_profile
    output_profile = s3_output_profile if s3_output_profile is not None else aws_profile

    def get_endpoint(input_specific: str | None, default: str | None) -> str | None:
        """Get endpoint URL, only if explicitly provided."""
        if input_specific and input_specific.strip():
            return input_specific.strip()
        if default and default.strip():
            return default.strip()
        return None

    input_endpoint = get_endpoint(s3_input_endpoint_url, s3_endpoint_url)
    output_endpoint = get_endpoint(s3_output_endpoint_url, s3_endpoint_url)

    # Ensure base path ends with /
    base_path = s3_input_base_path.rstrip("/") + "/"

    # Create temporary directory for consolidated processing
    consolidated_output = Path(output_dir)
    consolidated_output.mkdir(parents=True, exist_ok=True)

    sequence_outputs: list[str] = []
    successful_sequences = 0
    failed_sequences = 0

    logger.info(f"Processing {len(sequence_ids)} sequence(s)...")

    for idx, sequence_id in enumerate(sequence_ids, 1):
        logger.info(f"{'=' * 80}")
        logger.info(f"Processing sequence {idx}/{len(sequence_ids)}: {sequence_id}")
        logger.info(f"{'=' * 80}")

        # Construct S3 input path for this sequence
        sequence_s3_input = base_path + sequence_id + "/"

        # Create temporary output directory for this sequence
        seq_output_dir = str(consolidated_output / f"seq_{sequence_id}")

        try:
            # Process this sequence by calling main logic
            process_single_sequence(
                s3_input=sequence_s3_input,
                data_dir=None,
                camera_names=camera_names,
                output_dir=seq_output_dir,
                max_frames=max_frames,
                chunk_output=chunk_output,
                overlay_camera=overlay_camera,
                alpha=alpha,
                use_persistent_vbos=use_persistent_vbos,
                multi_sample=multi_sample,
                novel_pose_tar=novel_pose_tar,
                extract_frames=extract_frames,
                skip_frames=skip_frames,
                input_profile=input_profile,
                input_endpoint=input_endpoint,
                keep_local=True,  # Keep individual sequence outputs for consolidation
            )

            sequence_outputs.append(seq_output_dir)
            successful_sequences += 1
            logger.info(f"Successfully processed sequence {sequence_id}")

        except Exception as e:
            failed_sequences += 1
            logger.error(f"Failed to process sequence {sequence_id}: {e}")
            continue

    # Consolidate all outputs
    if sequence_outputs:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Consolidating outputs from {len(sequence_outputs)} sequence(s)...")
        logger.info(f"{'=' * 80}")
        consolidate_outputs(output_dir, sequence_outputs)

        # Cleanup individual sequence outputs before S3 upload if not keeping local
        if not keep_local:
            logger.info("Cleaning up individual sequence output directories before S3 upload...")
            for seq_output in sequence_outputs:
                if Path(seq_output).exists():
                    shutil.rmtree(seq_output)
                    logger.debug(f"Removed sequence output directory: {seq_output}")

        # Upload consolidated output to S3 if specified
        if s3_output:
            logger.info(f"Uploading consolidated output to S3: {s3_output}")
            try:
                upload_to_s3(output_dir, s3_output, output_profile, output_endpoint)
                logger.info(f"Successfully uploaded consolidated output to {s3_output}")
            except Exception as e:
                logger.error(f"Failed to upload consolidated output to S3: {e}")
                raise

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing complete:")
    logger.info(f"  - Successful: {successful_sequences}/{len(sequence_ids)}")
    logger.info(f"  - Failed: {failed_sequences}/{len(sequence_ids)}")
    logger.info(f"{'=' * 80}")


def process_single_sequence(
    s3_input: Optional[str],
    data_dir: Optional[str],
    camera_names: list[str],
    output_dir: str,
    max_frames: int,
    chunk_output: bool,
    overlay_camera: bool,
    alpha: float,
    use_persistent_vbos: bool,
    multi_sample: int,
    novel_pose_tar: Optional[str],
    extract_frames: bool,
    skip_frames: int,
    input_profile: Optional[str],
    input_endpoint: Optional[str],
    keep_local: bool,
) -> None:
    """
    Process a single sequence (extracted from main() for reuse).

    This function contains the core processing logic from main() for a single sequence.
    """
    # Handle S3 input download if specified
    temp_input_dir = None
    if s3_input:
        logger.info(f"Downloading input data from S3: {s3_input}")
        if input_profile:
            logger.info(f"Using AWS profile for input: {input_profile}")
        if input_endpoint:
            logger.info(f"Using S3 endpoint for input: {input_endpoint}")
        # Create temporary directory for download
        temp_input_dir = tempfile.mkdtemp(prefix="s3_input_")
        try:
            download_from_s3(s3_input, temp_input_dir, input_profile, input_endpoint)
            data_dir = temp_input_dir
            logger.info(f"Downloaded input data to temporary directory: {temp_input_dir}")
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            if temp_input_dir and Path(temp_input_dir).exists():
                shutil.rmtree(temp_input_dir)
            raise
    elif not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}. Use --s3-input to download from S3.")

    logger.info(f"Loading data from: {data_dir}")
    data_path = Path(data_dir)

    # Load scene data
    scene_data = load_scene(
        data_path,
        camera_names=None,
        max_frames=max_frames,
        input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
        resize_resolution_hw=SETTINGS["RESIZE_RESOLUTION"],
    )

    # Parse camera names
    if "all" in camera_names:
        available_cameras = []
        for cam in scene_data.camera_models.keys():
            mapped_name = cam.replace(":", "_")
            video_path_patterns = [
                data_path / f"{scene_data.scene_id}.{mapped_name}.mp4",
                data_path / f"ftheta_{mapped_name}" / f"{scene_data.scene_id}.mp4",
            ]
            video_dir = data_path / f"ftheta_{mapped_name}"
            if video_dir.exists():
                matching_videos = list(video_dir.glob(f"{scene_data.scene_id}*.mp4"))
                video_exists = bool(matching_videos) or any(path.exists() for path in video_path_patterns)
            else:
                video_exists = any(path.exists() for path in video_path_patterns)
            if video_exists:
                available_cameras.append(cam)
        camera_names = available_cameras
        logger.info(f"Found {len(camera_names)} cameras with video files")

        if len(camera_names) == 0:
            logger.error("No cameras with video files found.")
            raise ValueError("No cameras with video files found")

    logger.info(f"Loaded scene {scene_data.scene_id}:")
    logger.info(f"  - Frames: {scene_data.num_frames}")
    logger.info(f"  - Dynamic objects: {len(scene_data.dynamic_objects)}")
    logger.info(f"  - Cameras: {', '.join(camera_names)}")

    # Convert to rendering format
    all_camera_models, all_camera_poses = convert_scene_data_for_rendering(
        scene_data,
        camera_names,
        SETTINGS["RESIZE_RESOLUTION"],
    )

    # Optionally override camera poses
    if novel_pose_tar is not None:
        all_camera_poses = override_camera_poses_with_tar(all_camera_poses, camera_names, novel_pose_tar)

    # Process frames or render video
    if extract_frames:
        logger.info(f"Extracting frames with 3D annotations (skip_frames={skip_frames})")
        extract_frames_with_annotations(
            scene_data,
            camera_names,
            output_dir,
            data_path,
            max_frames,
            skip_frames,
            overlay_camera,
            alpha,
            all_camera_models,
            all_camera_poses,
            use_persistent_vbos,
            multi_sample,
        )
    else:
        logger.info(f"Using tiled multi-camera renderer for {len(camera_names)} camera(s)")
        render_multi_camera_tiled(
            all_camera_models,
            all_camera_poses,
            scene_data,
            camera_names,
            output_dir,
            scene_data.scene_id,
            max_frames,
            chunk_output,
            overlay_camera,
            alpha,
            data_path,
            use_persistent_vbos,
            multi_sample,
        )

    # Cleanup temporary input directory if created
    if temp_input_dir and Path(temp_input_dir).exists():
        logger.debug(f"Cleaning up temporary input directory: {temp_input_dir}")
        shutil.rmtree(temp_input_dir)


def download_from_s3(
    s3_path: str, local_path: str, profile: str | None = None, endpoint_url: str | None = None
) -> None:
    """
    Download directory or file from S3 to local path.

    Args:
        s3_path: S3 path (s3://bucket/path)
        local_path: Local destination path
        profile: Optional AWS profile name
        endpoint_url: Optional S3 endpoint URL (only used if provided and not empty)
    """
    # Build command with proper argument order: aws [--profile X] [--endpoint-url Y] s3 sync ...
    cmd_parts: list[str] = ["aws"]

    if profile:
        cmd_parts.extend(["--profile", profile])

    # Only add --endpoint-url if explicitly provided (not None and not empty)
    if endpoint_url and endpoint_url.strip():
        cmd_parts.extend(["--endpoint-url", endpoint_url])

    cmd_parts.extend(["s3", "sync", s3_path, local_path, "--no-progress"])

    logger.info(f"Downloading from S3: {s3_path} -> {local_path}")
    if profile:
        logger.info(f"  Using AWS profile: {profile}")
    if endpoint_url:
        logger.info(f"  Using endpoint URL: {endpoint_url}")
    logger.debug(f"Command: {' '.join(cmd_parts)}")
    try:
        result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
        logger.info(f"Successfully downloaded from {s3_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download from S3: {e.stderr}")
        logger.error(f"Command was: {' '.join(cmd_parts)}")
        if "SignatureDoesNotMatch" in str(e.stderr):
            logger.error("Signature mismatch error - check that:")
            logger.error("  1. AWS profile credentials match the endpoint URL")
            logger.error("  2. Endpoint URL is correct for the S3 service")
            logger.error("  3. Profile configuration doesn't conflict with endpoint URL")
        raise RuntimeError(f"Failed to download from {s3_path}: {e.stderr}") from e


def upload_to_s3(local_path: str, s3_path: str, profile: str | None = None, endpoint_url: str | None = None) -> None:
    """
    Upload directory or file from local path to S3.

    Args:
        local_path: Local source path
        s3_path: S3 destination path (s3://bucket/path)
        profile: Optional AWS profile name
        endpoint_url: Optional S3 endpoint URL (only used if provided and not empty)
    """
    # Build command with proper argument order: aws [--profile X] [--endpoint-url Y] s3 sync ...
    cmd_parts: list[str] = ["aws"]

    if profile:
        cmd_parts.extend(["--profile", profile])

    # Only add --endpoint-url if explicitly provided (not None and not empty)
    if endpoint_url and endpoint_url.strip():
        cmd_parts.extend(["--endpoint-url", endpoint_url])

    cmd_parts.extend(["s3", "sync", local_path, s3_path, "--no-progress"])

    logger.info(f"Uploading to S3: {local_path} -> {s3_path}")
    if profile:
        logger.info(f"  Using AWS profile: {profile}")
    if endpoint_url:
        logger.info(f"  Using endpoint URL: {endpoint_url}")
    logger.debug(f"Command: {' '.join(cmd_parts)}")
    try:
        result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
        logger.info(f"Successfully uploaded to {s3_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload to S3: {e.stderr}")
        logger.error(f"Command was: {' '.join(cmd_parts)}")
        if "SignatureDoesNotMatch" in str(e.stderr):
            logger.error("Signature mismatch error - check that:")
            logger.error("  1. AWS profile credentials match the endpoint URL")
            logger.error("  2. Endpoint URL is correct for the S3 service")
            logger.error("  3. Profile configuration doesn't conflict with endpoint URL")
        raise RuntimeError(f"Failed to upload to {s3_path}: {e.stderr}") from e


def is_bbox_in_camera_view(
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    camera_model: FThetaCamera,
    camera_pose: np.ndarray,
    bbox_orientation: np.ndarray | None = None,
) -> bool:
    """
    Check if a 3D bounding box is visible in the camera view with strict filtering.

    Filters out objects that are:
    - Behind camera (negative z in camera frame)
    - Too far away (beyond 100m)
    - Outside the camera frame (strict FOV check)
    - Not sufficiently visible (less than 25% of corners visible)

    Args:
        bbox_center: (3,) array of [x, y, z] in world coordinates
        bbox_dims: (3,) array of [length, width, height]
        camera_model: Camera model
        camera_pose: (4, 4) camera to world transformation matrix
        bbox_orientation: Optional (4,) quaternion [x, y, z, w] for bbox rotation in FLU convention

    Returns:
        True if the bbox is sufficiently visible within the camera frame and within distance limit
    """
    # Transform bbox center to camera coordinates
    world_to_camera = np.linalg.inv(camera_pose)
    bbox_center_homogeneous = np.hstack([bbox_center, 1.0])
    bbox_center_cam = (world_to_camera @ bbox_center_homogeneous)[:3]
    # Check if object is behind camera (negative z in camera frame)
    if bbox_center_cam[2] <= 0:
        return False

    # Check distance from camera using depth (z-coordinate in camera frame)
    # Filter out objects beyond max_distance_from_camera meters
    # In RDF camera coordinates, Z is forward, so bbox_center_cam[2] is the forward distance
    forward_distance = float(bbox_center_cam[2])
    max_distance_from_camera = 100.0  # 100 meters
    if forward_distance > max_distance_from_camera:
        return False

    # Get bbox corners in world coordinates
    # Create 8 corners of the bounding box in local space (centered at origin)
    l, w, h = bbox_dims
    corners_local = np.array(
        [
            [-l / 2, -w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [l / 2, w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
        ]
    )

    # Apply rotation if orientation is provided
    if bbox_orientation is not None:
        from scipy.spatial.transform import Rotation

        rot_matrix = Rotation.from_quat(bbox_orientation).as_matrix()
        # Rotate corners: corners_rotated = corners_local @ rot_matrix
        # rot_matrix rotates from local to world coordinates
        corners_local = corners_local @ rot_matrix

    # Transform corners to world coordinates (rotate then translate)
    corners_world = corners_local + bbox_center

    # Transform to camera coordinates
    corners_cam_homogeneous = (world_to_camera @ np.hstack([corners_world, np.ones((8, 1))]).T).T
    corners_cam = corners_cam_homogeneous[:, :3]

    # Check if any corner is in front of the camera (positive z)
    if np.all(corners_cam[:, 2] <= 0):
        return False

    # For corners in front of camera, project to pixels
    valid_corners = corners_cam[corners_cam[:, 2] > 0]
    if len(valid_corners) == 0:
        return False

    # Project valid corners to pixel coordinates using standardized FoV
    pixels = camera_model.ray2pixel_np(valid_corners)

    # Check if pixels are within strict image bounds (no margin - must be actually visible)
    x_in_bounds = np.logical_and(pixels[:, 0] >= 0, pixels[:, 0] < camera_model.width)
    y_in_bounds = np.logical_and(pixels[:, 1] >= 0, pixels[:, 1] < camera_model.height)
    in_bounds = np.logical_and(x_in_bounds, y_in_bounds)

    # Require at least 10% of corners to be within bounds
    visibility_ratio_threshold = 0.10
    visible_corners = np.sum(in_bounds)
    total_valid_corners = len(valid_corners)
    visible_ratio = visible_corners / total_valid_corners if total_valid_corners > 0 else 0.0

    # Also check that at least some portion of the bbox center projects within bounds
    center_pixel = camera_model.ray2pixel_np(bbox_center_cam.reshape(1, -1))[0]
    center_in_bounds = 0 <= center_pixel[0] < camera_model.width and 0 <= center_pixel[1] < camera_model.height

    # Object is visible if center is in bounds AND sufficient corners are visible
    return center_in_bounds and visible_ratio >= visibility_ratio_threshold


def world_to_camera_coordinates(
    bbox_3d_world: list[float],
    camera_pose: np.ndarray,
) -> list[float]:
    """
    Transform 3D bounding box from FLU world coordinates to RDF camera coordinates.

    According to world_scenario_parquet.md:
    - World Coordinate System: FLU convention (Forward=X, Left=Y, Up=Z)
      - Origin: Ego vehicle's starting position (first frame) defines world origin (0,0,0)
    - Camera Coordinate System: RDF convention (Right=X, Down=Y, Forward=Z)
      - X: Right (positive X is to the right of camera)
      - Y: Down (positive Y is downward)
      - Z: Forward (positive Z is forward, in front of camera)

    Transformation chain: world → camera (via camera_pose inverse)
    - camera_pose: (4, 4) camera-to-world transformation matrix
    - world_to_camera = inv(camera_pose)

    Args:
        bbox_3d_world: [x, y, z, x_size, y_size, z_size, roll, pitch, yaw] in FLU world coordinates
                       - x, y, z: Center position in world coordinate frame (meters)
                       - x_size, y_size, z_size: Bounding box dimensions (meters)
                       - roll, pitch, yaw: XYZ Euler angles in radians (rotation in world frame)
        camera_pose: (4, 4) camera-to-world transformation matrix

    Returns:
        [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw] in RDF camera coordinates
        - x_center, y_center, z_center: Center position in camera coordinate frame (meters)
        - x_size, y_size, z_size: Bounding box dimensions (same as input, sizes don't change)
        - roll, pitch, yaw: XYZ Euler angles in radians, transformed to camera frame
    """
    if len(bbox_3d_world) < 6:
        return bbox_3d_world

    # Extract center and size
    center_world = np.array(bbox_3d_world[:3], dtype=np.float32)
    size = bbox_3d_world[3:6]

    # Get world-to-camera transformation (inverse of camera-to-world)
    world_to_camera = np.linalg.inv(camera_pose)

    # Transform center: camera_coords = world_to_camera @ [world_coords, 1]
    center_world_homogeneous = np.hstack([center_world, 1.0])
    center_camera_homogeneous = world_to_camera @ center_world_homogeneous
    center_camera = center_camera_homogeneous[:3]

    # Transform rotation angles
    # Input format: [roll, pitch, yaw] (XYZ Euler order) in world FLU frame
    # We need to convert to rotation matrix, transform, then back to [roll, pitch, yaw]
    if len(bbox_3d_world) >= 9:
        roll_world, pitch_world, yaw_world = bbox_3d_world[6], bbox_3d_world[7], bbox_3d_world[8]

        # Convert Euler angles to rotation matrix (XYZ order: roll around X, pitch around Y, yaw around Z)
        from scipy.spatial.transform import Rotation

        R_bbox_world = Rotation.from_euler("xyz", [roll_world, pitch_world, yaw_world], degrees=False).as_matrix()

        # Transform rotation matrix: R_camera = R_world_to_camera @ R_world @ R_camera_to_world
        # Since world_to_camera[:3, :3] is the rotation part
        R_world_to_camera = world_to_camera[:3, :3]
        R_camera_to_world = camera_pose[:3, :3]
        R_bbox_camera = R_world_to_camera @ R_bbox_world @ R_camera_to_world

        # Convert back to Euler angles (XYZ order: roll, pitch, yaw)
        rot = Rotation.from_matrix(R_bbox_camera)
        euler_camera = rot.as_euler("xyz", degrees=False)
        roll_camera, pitch_camera, yaw_camera = euler_camera[0], euler_camera[1], euler_camera[2]
    else:
        roll_camera, pitch_camera, yaw_camera = 0.0, 0.0, 0.0

    # Output format: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
    # All coordinates are in camera coordinate system (RDF convention: Right, Down, Forward)
    # - x_center, y_center, z_center: Object center position in camera coordinates
    #   - x_center: Right (positive X is to the right of camera)
    #   - y_center: Down (positive Y is downward)
    #   - z_center: Forward (positive Z is forward, in front of camera)
    # - x_size, y_size, z_size: Bounding box dimensions (same as input, sizes don't change)
    # - roll, pitch, yaw: Rotation angles in camera coordinate frame (XYZ Euler order, radians)
    bbox_3d_camera = [
        float(center_camera[0]),  # x_center in camera coordinates (Right)
        float(center_camera[1]),  # y_center in camera coordinates (Down)
        float(center_camera[2]),  # z_center in camera coordinates (Forward)
        float(size[0]),  # x_size
        float(size[1]),  # y_size
        float(size[2]),  # z_size
        float(roll_camera),
        float(pitch_camera),
        float(yaw_camera),
    ]

    return bbox_3d_camera


def extract_camera_params(camera_model: FThetaCamera) -> dict[str, float]:
    """
    Extract camera parameters (fx, fy, cx, cy) from camera model.

    These parameters are used to project camera coordinates (RDF convention: Right, Down, Forward)
    to 2D pixel coordinates using the pinhole camera model:
        x_2d = fx * (X / Z) + cx
        y_2d = fy * (Y / Z) + cy

    Where (X, Y, Z) are in camera coordinates (RDF):
        - X: Right (positive X is to the right)
        - Y: Down (positive Y is downward)
        - Z: Forward (positive Z is forward, in front of camera)

    Note: The camera model uses FTheta (fisheye) projection, but we approximate it with
    pinhole parameters computed from FOV. This approximation works well for projection
    of camera coordinates to pixels.

    Note: FOV is used internally to compute focal lengths but is not stored in output.
    FOV does not affect world-to-camera coordinate conversion, which only depends on
    camera pose (rotation and translation).

    Args:
        camera_model: FThetaCamera model (uses RDF camera coordinate convention)

    Returns:
        Dictionary with fx, fy, cx, cy parameters for pinhole projection
        - fx, fy: Focal lengths in pixels (computed from FOV, appropriate for RDF coordinates)
        - cx, cy: Principal point coordinates in pixels
    """
    # Extract cx, cy directly from intrinsics
    intrinsics = camera_model.intrinsics
    cx = float(intrinsics[0])
    cy = float(intrinsics[1])
    width = int(intrinsics[2])
    height = int(intrinsics[3])

    # Compute fx and fy from FOV if available
    # For FTheta cameras, we compute equivalent focal lengths from FOV
    # These focal lengths are appropriate for projecting RDF camera coordinates to pixels
    # Note: FOV is used here to compute focal lengths, but does not affect
    # world-to-camera coordinate conversion (which only uses camera pose)
    try:
        # Get FOV using the public property (automatically computes if needed)
        h_fov, v_fov = camera_model.fov

        # Extract scalar values (handle numpy scalars/arrays)
        h_fov_val = h_fov.item() if isinstance(h_fov, (np.ndarray, np.generic)) else float(h_fov)
        v_fov_val = v_fov.item() if isinstance(v_fov, (np.ndarray, np.generic)) else float(v_fov)

        # Compute focal lengths from FOV: fx = width / (2 * tan(h_fov / 2))
        # These are appropriate for RDF camera coordinate system (X-right, Y-down, Z-forward)
        fx_val = width / (2.0 * np.tan(h_fov_val / 2.0))
        fy_val = height / (2.0 * np.tan(v_fov_val / 2.0))
        fx = fx_val.item() if isinstance(fx_val, (np.ndarray, np.generic)) else float(fx_val)
        fy = fy_val.item() if isinstance(fy_val, (np.ndarray, np.generic)) else float(fy_val)
    except Exception as e:
        # Fallback: compute approximate focal length assuming typical FOV
        # For wide-angle cameras, approximate FOV is often around 120 degrees
        # This is a reasonable approximation if FOV is not available
        logger.debug(f"Could not compute focal length from FOV: {e}, using approximation")
        h_fov_approx = np.radians(60.0)
        fx_val = width / (2.0 * np.tan(h_fov_approx / 2.0))
        fx = fx_val.item() if isinstance(fx_val, (np.ndarray, np.generic)) else float(fx_val)
        fy = fx

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def project_bbox_to_2d(
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    camera_model: FThetaCamera,
    camera_pose: np.ndarray,
    bbox_orientation: np.ndarray | None = None,
) -> tuple[float, float, float, float, float] | None:
    """
    Project a 3D bounding box to 2D pixel coordinates and return 2D bbox bounds.

    Args:
        bbox_center: (3,) array of [x, y, z] in world coordinates
        bbox_dims: (3,) array of [length, width, height]
        camera_model: Camera model
        camera_pose: (4, 4) camera to world transformation matrix
        bbox_orientation: Optional (4,) quaternion [x, y, z, w] for bbox rotation in FLU convention

    Returns:
        Tuple of (min_x, min_y, max_x, max_y, depth) in pixel coordinates, or None if not visible
    """
    # Transform bbox center to camera coordinates
    world_to_camera = np.linalg.inv(camera_pose)
    bbox_center_homogeneous = np.hstack([bbox_center, 1.0])
    bbox_center_cam = (world_to_camera @ bbox_center_homogeneous)[:3]

    # Check if object is behind camera
    if bbox_center_cam[2] <= 0:
        return None

    depth = float(bbox_center_cam[2])

    # Get bbox corners in world coordinates
    # Create 8 corners of the bounding box in local space (centered at origin)
    l, w, h = bbox_dims
    corners_local = np.array(
        [
            [-l / 2, -w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [l / 2, w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
        ]
    )

    # Apply rotation if orientation is provided
    if bbox_orientation is not None:
        from scipy.spatial.transform import Rotation

        rot_matrix = Rotation.from_quat(bbox_orientation).as_matrix()
        # Rotate corners: corners_rotated = corners_local @ rot_matrix
        # rot_matrix rotates from local to world coordinates
        corners_local = corners_local @ rot_matrix

    # Transform corners to world coordinates (rotate then translate)
    corners_world = corners_local + bbox_center

    # Transform to camera coordinates
    corners_cam_homogeneous = (world_to_camera @ np.hstack([corners_world, np.ones((8, 1))]).T).T
    corners_cam = corners_cam_homogeneous[:, :3]

    # Filter to corners in front of camera
    valid_corners = corners_cam[corners_cam[:, 2] > 0]
    if len(valid_corners) == 0:
        return None

    # Project to pixel coordinates
    pixels = camera_model.ray2pixel_np(valid_corners)

    # Get 2D bounding box bounds
    min_x = float(np.min(pixels[:, 0]))
    min_y = float(np.min(pixels[:, 1]))
    max_x = float(np.max(pixels[:, 0]))
    max_y = float(np.max(pixels[:, 1]))

    return (min_x, min_y, max_x, max_y, depth)


def is_bbox_overlapped(
    bbox1: tuple[float, float, float, float, float],
    bbox2: tuple[float, float, float, float, float],
    overlap_threshold: float = 0.90,
) -> bool:
    """
    Check if bbox1 is fully overlapped by bbox2 (bbox2 occludes bbox1).

    Args:
        bbox1: (min_x, min_y, max_x, max_y, depth) of the potentially occluded bbox
        bbox2: (min_x, min_y, max_x, max_y, depth) of the potentially occluding bbox
        overlap_threshold: Minimum overlap ratio to consider fully overlapped (default: 0.9 = 90%)

    Returns:
        True if bbox1 is fully overlapped by bbox2
    """
    min_x1, min_y1, max_x1, max_y1, depth1 = bbox1
    min_x2, min_y2, max_x2, max_y2, depth2 = bbox2

    # bbox2 must be closer (smaller depth) to occlude bbox1
    if depth2 >= depth1:
        return False

    # Calculate intersection area
    inter_min_x = max(min_x1, min_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_x = min(max_x1, max_x2)
    inter_max_y = min(max_y1, max_y2)

    if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
        return False  # No intersection

    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    bbox1_area = (max_x1 - min_x1) * (max_y1 - min_y1)

    if bbox1_area <= 0:
        return False

    # Check if overlap ratio exceeds threshold
    overlap_ratio = inter_area / bbox1_area
    return overlap_ratio >= overlap_threshold


def extract_3d_annotations(
    scene_data: SceneData,
    frame_id: int,
    camera_model: Optional[FThetaCamera] = None,
    camera_pose: Optional[np.ndarray] = None,
    filter_occluded: bool = True,
    overlap_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """
    Extract 3D bounding box annotations for a given frame with strict visibility filtering.

    Filters out objects that are:
    - Too far away (beyond 50m) or outside camera FOV
    - Fully overlapped by other objects (occluded)
    - Behind objects that are closer to the camera

    Note: camera_pose is required. If not provided, returns empty list.

    Args:
        scene_data: Scene data containing dynamic objects
        frame_id: Frame index
        camera_model: Optional camera model for frustum culling
        camera_pose: Required (4, 4) camera to world transformation matrix
        filter_occluded: If True, filter out objects fully overlapped by closer objects (default: True)
        overlap_threshold: Minimum overlap ratio to consider fully overlapped (default: 0.9 = 90%)

    Returns:
        List of annotation dictionaries with label and bbox_3d in camera coordinates (RDF convention).
        Returns empty list if camera_pose is not provided.
    """
    # Require camera_pose for visibility filtering
    if camera_pose is None:
        logger.warning("camera_pose is required for 3D annotation extraction. Skipping frame.")
        return []

    annotations_with_metadata = []
    timestamps = scene_data.timestamps

    if frame_id >= len(timestamps):
        return []

    frame_timestamp = timestamps[frame_id]

    # First pass: collect all visible annotations with their 2D projections
    for track_id, obj in scene_data.dynamic_objects.items():
        # Filter to only include vehicles (exclude Pedestrians and Cyclists)
        if obj.object_type in (ObjectType.PEDESTRIAN, ObjectType.CYCLIST, ObjectType.OTHER):
            continue

        # Get pose data at this frame's timestamp
        pose_data = obj.get_pose_at_timestamp(frame_timestamp)
        if pose_data is None:
            continue

        center, dimensions, orientation = pose_data

        # Require camera_model for extraction (camera_pose already checked at function start)
        if camera_model is None:
            logger.warning("camera_model is required for 3D annotation extraction. Skipping object.")
            continue

        # Check if bbox is in view with strict filtering (includes distance, FOV, and visibility checks)
        if not is_bbox_in_camera_view(
            center,
            dimensions,
            camera_model,
            camera_pose,
            orientation,  # Pass orientation for proper corner rotation
        ):
            continue

        # Project to 2D for occlusion checking
        bbox_2d = project_bbox_to_2d(center, dimensions, camera_model, camera_pose, orientation)
        if bbox_2d is None:
            continue

        # Extract object label (track_id with type)
        label = f"{obj.object_type.value}_{track_id}"

        # Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw]
        # orientation is in scipy format [x, y, z, w]
        # Using XYZ Euler order to get [roll, pitch, yaw] for camera-centered coordinates
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_quat(orientation)
        euler = rot.as_euler("xyz", degrees=False)  # Returns [roll, pitch, yaw] (XYZ Euler order)
        roll, pitch, yaw = euler[0], euler[1], euler[2]

        # Build bbox_3d in FLU world coordinates: [x, y, z, x_size, y_size, z_size, roll, pitch, yaw]
        # Note: dimensions from DynamicObject are [length, width, height] in FLU convention
        # We map: length -> x_size (forward), width -> y_size (left), height -> z_size (up)
        # Rotation order: [roll, pitch, yaw] (XYZ Euler order)
        # Format: FLU world coordinates
        bbox_3d_world = [
            float(center[0]),  # x in world (FLU)
            float(center[1]),  # y in world (FLU)
            float(center[2]),  # z in world (FLU)
            float(dimensions[0]),  # x_size (length along forward/X)
            float(dimensions[1]),  # y_size (width along left/Y)
            float(dimensions[2]),  # z_size (height along up/Z)
            float(roll),  # roll (rotation around X)
            float(pitch),  # pitch (rotation around Y)
            float(yaw),  # yaw (rotation around Z)
        ]

        # Convert from FLU world coordinates to RDF camera coordinates
        # According to world_scenario_parquet.md:
        # - World origin: Ego vehicle's starting position (first frame) defines world origin (0,0,0)
        # - Camera coordinates: RDF convention (Right=X, Down=Y, Forward=Z)
        # - Transformation: world → camera via camera_pose inverse
        bbox_3d = world_to_camera_coordinates(bbox_3d_world, camera_pose)

        annotations_with_metadata.append(
            {
                "label": label,
                "bbox_3d": bbox_3d,
                "bbox_2d": bbox_2d,
            }
        )

    # Second pass: filter out occluded objects if requested
    # camera_pose is required (checked above), so we can always filter if requested
    if filter_occluded and camera_model is not None:
        # Sort by depth (closest first)
        annotations_with_metadata.sort(key=lambda x: x["bbox_2d"][4] if x["bbox_2d"] is not None else float("inf"))

        filtered_annotations = []
        for i, ann1 in enumerate(annotations_with_metadata):
            if ann1["bbox_2d"] is None:
                # If no 2D projection, keep it (shouldn't happen if camera is provided)
                filtered_annotations.append({"label": ann1["label"], "bbox_3d": ann1["bbox_3d"]})
                continue

            is_occluded = False
            # Check if this object is fully overlapped by any closer object
            for j, ann2 in enumerate(annotations_with_metadata[:i]):
                if ann2["bbox_2d"] is None:
                    continue

                if is_bbox_overlapped(ann1["bbox_2d"], ann2["bbox_2d"], overlap_threshold):
                    is_occluded = True
                    logger.debug(f"Filtered out {ann1['label']} (occluded by {ann2['label']})")
                    break

            if not is_occluded:
                filtered_annotations.append({"label": ann1["label"], "bbox_3d": ann1["bbox_3d"]})

        return filtered_annotations
    else:
        # Return annotations without occlusion filtering
        return [{"label": ann["label"], "bbox_3d": ann["bbox_3d"]} for ann in annotations_with_metadata]


# Common functions are imported from local.py to avoid duplication
# Overlay rendering functions are imported from local_render_frames.py


@click.command()
@click.argument("data_dir", type=click.Path(exists=False), required=False, nargs=1)
@click.option(
    "--camera-names",
    default=["all"],
    multiple=True,
    help="Camera sensor name (use underscores, e.g., camera_front_wide_120fov)",
)
@click.option("--output-dir", default="output", help="Output directory for rendered videos")
@click.option("--max-frames", default=-1, help="Maximum number of frames to render (-1 for all)")
@click.option(
    "--chunk-output/--no-chunk-output",
    is_flag=True,
    default=False,
    help="Save as single video instead of chunked videos",
)
@click.option(
    "--overlay-camera/--no-overlay-camera",
    default=False,
    is_flag=True,
    help="Overlay camera view on the HD map",
)
@click.option(
    "--alpha",
    default=0.5,
    help="Alpha value for camera overlay",
)
@click.option(
    "--use-persistent-vbos/--no-persistent-vbos",
    default=True,
    is_flag=True,
    help="Use persistent VBOs for static geometry",
)
@click.option(
    "--multi-sample",
    default=4,
    type=int,
    help="Number of samples for multisampling anti-aliasing (MSAA). Use 1 to disable, 4 for 4x MSAA (default).",
)
@click.option(
    "--novel-pose-tar",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional tar with novel poses (keys like 000000.pose.camera_front_wide_120fov.npy)",
)
@click.option(
    "--extract-frames/--no-extract-frames",
    default=False,
    is_flag=True,
    help="Extract individual frames with 3D bbox annotations instead of rendering video",
)
@click.option(
    "--skip-frames",
    default=30,
    type=int,
    help="Extract every Nth frame (default: 1, extract all frames)",
)
@click.option(
    "--s3-input",
    default=None,
    type=str,
    help="S3 path for input data (s3://bucket/path). If provided, downloads data before processing.",
)
@click.option(
    "--s3-output",
    default=None,
    type=str,
    help="S3 path for output data (s3://bucket/path). If provided, uploads results after processing.",
)
@click.option(
    "--aws-profile",
    default=None,
    type=str,
    help="AWS profile to use for S3 operations (applies to both input and output if not overridden)",
)
@click.option(
    "--s3-input-profile",
    default="team-cosmos",
    type=str,
    help="AWS profile to use for S3 input operations (overrides --aws-profile for input)",
)
@click.option(
    "--s3-output-profile",
    default="team-cosmos",
    type=str,
    help="AWS profile to use for S3 output operations (overrides --aws-profile for output)",
)
@click.option(
    "--s3-endpoint-url",
    default=None,
    type=str,
    help="Optional S3 endpoint URL (for custom S3-compatible services)",
)
@click.option(
    "--s3-input-endpoint-url",
    default=None,
    type=str,
    help="Optional S3 endpoint URL for input operations (overrides --s3-endpoint-url for input)",
)
@click.option(
    "--s3-output-endpoint-url",
    default=None,
    type=str,
    help="Optional S3 endpoint URL for output operations (overrides --s3-endpoint-url for output)",
)
@click.option(
    "--keep-local/--no-keep-local",
    default=False,
    is_flag=True,
    help="Keep local files after S3 upload (default: False)",
)
@click.option(
    "--sequence-ids-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to file containing sequence IDs (UUIDs) to process, one per line. If provided, processes each sequence and consolidates outputs.",
)
@click.option(
    "--s3-input-base-path",
    default=None,
    type=str,
    help="Base S3 path for input data. Sequence IDs will be appended to this path (e.g., s3://bucket/path/). Required when using --sequence-ids-file.",
)
def main(
    data_dir: Optional[str],
    camera_names: list[str],
    output_dir: str,
    max_frames: int,
    chunk_output: bool,
    overlay_camera: bool,
    alpha: float,
    use_persistent_vbos: bool,
    multi_sample: int,
    novel_pose_tar: Optional[str],
    extract_frames: bool,
    skip_frames: int,
    s3_input: Optional[str],
    s3_output: Optional[str],
    aws_profile: Optional[str],
    s3_input_profile: Optional[str],
    s3_output_profile: Optional[str],
    s3_endpoint_url: Optional[str],
    s3_input_endpoint_url: Optional[str],
    s3_output_endpoint_url: Optional[str],
    keep_local: bool,
    sequence_ids_file: Optional[str],
    s3_input_base_path: Optional[str],
) -> None:
    """
    Main entry point for HD map rendering and frame extraction.

    This function orchestrates the entire pipeline:
    1. Loads scene data from ClipGT or RDS-HQ format
    2. Converts camera models and poses for rendering
    3. Optionally overrides poses with novel trajectory
    4. Either extracts frames with annotations or renders HD map videos

    Args:
        data_dir: Path to data directory containing scene data (ClipGT or RDS-HQ format).
                 Optional, defaults to /tmp/local_data if not provided.
        camera_names: List of camera sensor names (use underscores, e.g., camera_front_wide_120fov).
                     Use "all" to process all available cameras with video files.
        output_dir: Output directory for rendered videos or extracted frames
        max_frames: Maximum number of frames to process (-1 for all frames)
        chunk_output: If True, save as chunked videos instead of single video
        overlay_camera: If True, overlay camera video on rendered HD map
        alpha: Alpha blending value for overlay (0.0 = full camera, 1.0 = full HD map)
        use_persistent_vbos: Use persistent VBOs for static geometry (improves performance)
        multi_sample: MSAA samples (1 = disabled, 4 = 4x MSAA, default)
        novel_pose_tar: Optional path to tar file with novel camera poses
                       (keys like 000000.pose.camera_front_wide_120fov.npy)
        extract_frames: If True, extract individual frames with annotations instead of rendering video
        skip_frames: Extract every Nth frame when extract_frames=True (1 = all frames)
        s3_input: Optional S3 path for input data. If provided, downloads data before processing
        s3_output: Optional S3 path for output data. If provided, uploads results after processing
        aws_profile: Optional AWS profile name for S3 operations (default for both input/output)
        s3_input_profile: Optional AWS profile for input operations (overrides aws_profile for input)
        s3_output_profile: Optional AWS profile for output operations (overrides aws_profile for output)
        s3_endpoint_url: Optional S3 endpoint URL for custom S3-compatible services (default for both)
        s3_input_endpoint_url: Optional S3 endpoint URL for input (overrides s3_endpoint_url for input)
        s3_output_endpoint_url: Optional S3 endpoint URL for output (overrides s3_endpoint_url for output)
        keep_local: If True, keep local files after S3 upload (default: False)

    Output:
        - Video rendering: Creates MP4 files in output_dir
        - Frame extraction: Creates JPG images and JSON files in output_dir
        - Each JSON contains frame_id, camera name, and list of 3D bounding box annotations

    Raises:
        FileNotFoundError: If data directory or required files are missing
        ValueError: If camera names are invalid or novel pose tar format is incorrect
    """

    # Load sequence IDs from file if provided
    sequence_ids: list[str] = []
    if sequence_ids_file:
        if not s3_input_base_path:
            raise ValueError("--s3-input-base-path is required when using --sequence-ids-file")
        sequence_ids = load_sequence_ids_from_file(sequence_ids_file)
        logger.info(f"Loaded {len(sequence_ids)} sequence ID(s) from {sequence_ids_file}")
        if not sequence_ids:
            logger.error("No valid sequence IDs found in file")
            return

    # If processing multiple sequences, handle them in a loop
    if sequence_ids:
        if not s3_input_base_path:
            raise ValueError("--s3-input-base-path is required when using --sequence-ids-file")
        process_multiple_sequences(
            sequence_ids=sequence_ids,
            s3_input_base_path=s3_input_base_path,  # type: ignore[arg-type]  # Already checked above
            camera_names=camera_names,
            output_dir=output_dir,
            max_frames=max_frames,
            chunk_output=chunk_output,
            overlay_camera=overlay_camera,
            alpha=alpha,
            use_persistent_vbos=use_persistent_vbos,
            multi_sample=multi_sample,
            novel_pose_tar=novel_pose_tar,
            extract_frames=extract_frames,
            skip_frames=skip_frames,
            s3_output=s3_output,
            aws_profile=aws_profile,
            s3_input_profile=s3_input_profile,
            s3_output_profile=s3_output_profile,
            s3_endpoint_url=s3_endpoint_url,
            s3_input_endpoint_url=s3_input_endpoint_url,
            s3_output_endpoint_url=s3_output_endpoint_url,
            keep_local=keep_local,
        )
        return

    # Use default data directory if not provided
    if data_dir is None:
        data_dir = "/tmp/local_data"

    # Determine input and output profiles/endpoints
    # Use input/output specific values if provided, otherwise fall back to defaults
    input_profile = s3_input_profile if s3_input_profile is not None else aws_profile
    output_profile = s3_output_profile if s3_output_profile is not None else aws_profile

    # For endpoints, only use if explicitly provided via command line
    # Since defaults are set in click options, we need to check if they were explicitly provided
    # We'll pass None to the functions, which will only add --endpoint-url if the value is truthy
    # This way, if defaults match what's in the profile, we don't override it
    input_endpoint = s3_input_endpoint_url if s3_input_endpoint_url else None
    output_endpoint = s3_output_endpoint_url if s3_output_endpoint_url else None

    # Only use default endpoint if input/output specific ones aren't set
    if not input_endpoint and s3_endpoint_url:
        input_endpoint = s3_endpoint_url
    if not output_endpoint and s3_endpoint_url:
        output_endpoint = s3_endpoint_url

    # If endpoint is the default value, don't pass it (let profile handle it)
    default_endpoint = "https://pbss.s8k.io"
    if input_endpoint == default_endpoint:
        input_endpoint = None
    if output_endpoint == default_endpoint:
        output_endpoint = None

    # Handle S3 input download if specified
    temp_input_dir = None
    if s3_input:
        logger.info(f"Downloading input data from S3: {s3_input}")
        if input_profile:
            logger.info(f"Using AWS profile for input: {input_profile}")
        if input_endpoint:
            logger.info(f"Using S3 endpoint for input: {input_endpoint}")
        # Create temporary directory for download
        temp_input_dir = tempfile.mkdtemp(prefix="s3_input_")
        try:
            download_from_s3(s3_input, temp_input_dir, input_profile, input_endpoint)
            data_dir = temp_input_dir
            logger.info(f"Downloaded input data to temporary directory: {temp_input_dir}")
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            if temp_input_dir and Path(temp_input_dir).exists():
                shutil.rmtree(temp_input_dir)
            raise
    elif not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. Use --s3-input to download from S3 or provide a valid --data-dir."
        )

    logger.info(f"Loading data from: {data_dir}")
    data_path = Path(data_dir)

    # Load scene data using the new loader system (auto-detects ClipGT or RDS-HQ)
    # First load with None to get all available cameras
    scene_data = load_scene(
        data_path,
        camera_names=None,  # Load all available cameras first
        max_frames=max_frames,
        input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
        resize_resolution_hw=SETTINGS["RESIZE_RESOLUTION"],
    )

    # Parse camera names after loading to see what's available
    if "all" in camera_names:
        # Only include cameras that have video files
        available_cameras = []
        for cam in scene_data.camera_models.keys():
            mapped_name = cam.replace(":", "_")
            # Try multiple possible video path patterns
            video_path_patterns = [
                data_path / f"{scene_data.scene_id}.{mapped_name}.mp4",  # Original ClipGT format
                data_path / f"ftheta_{mapped_name}" / f"{scene_data.scene_id}.mp4",  # RDS-HQ/MADS format
            ]
            # Also try glob pattern in case video files exist with matching prefix
            video_dir = data_path / f"ftheta_{mapped_name}"
            if video_dir.exists():
                # Check for any video file matching the clip ID pattern
                matching_videos = list(video_dir.glob(f"{scene_data.scene_id}*.mp4"))
                if matching_videos:
                    video_exists = True
                else:
                    video_exists = any(path.exists() for path in video_path_patterns)
            else:
                video_exists = any(path.exists() for path in video_path_patterns)
            if video_exists:
                available_cameras.append(cam)
        camera_names = available_cameras
        logger.info(f"Found {len(camera_names)} cameras with video files")

        if len(camera_names) == 0:
            logger.error("No cameras with video files found. Cannot proceed with frame extraction.")
            logger.info("Available camera models in scene data:")
            for cam in scene_data.camera_models.keys():
                logger.info(f"  - {cam}")
            logger.info("Checking for video files...")
            for cam in scene_data.camera_models.keys():
                mapped_name = cam.replace(":", "_")
                video_dir = data_path / f"ftheta_{mapped_name}"
                if video_dir.exists():
                    videos = list(video_dir.glob("*.mp4"))
                    logger.info(f"  {mapped_name}: {len(videos)} video files found")
                    if videos:
                        logger.info(f"    Sample: {videos[0].name}")
            return

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

    # Optionally override camera poses with a novel trajectory from tar
    if novel_pose_tar is not None:
        all_camera_poses = override_camera_poses_with_tar(all_camera_poses, camera_names, novel_pose_tar)

    # Choose between frame extraction or video rendering
    if extract_frames:
        logger.info(f"Extracting frames with 3D annotations (skip_frames={skip_frames})")
        extract_frames_with_annotations(
            scene_data,
            camera_names,
            output_dir,
            data_path,
            max_frames,
            skip_frames,
            overlay_camera,
            alpha,
            all_camera_models,
            all_camera_poses,
            use_persistent_vbos,
            multi_sample,
        )

    else:
        # Always use tiled multi-camera renderer (works for single camera too)
        logger.info(f"Using tiled multi-camera renderer for {len(camera_names)} camera(s)")
        render_multi_camera_tiled(
            all_camera_models,
            all_camera_poses,
            scene_data,
            camera_names,
            output_dir,
            scene_data.scene_id,
            max_frames,
            chunk_output,
            overlay_camera,
            alpha,
            data_path,
            use_persistent_vbos,
            multi_sample,
        )

    # Handle S3 output upload if specified (for both frame extraction and video rendering)
    if s3_output:
        logger.info(f"Uploading output to S3: {s3_output}")
        if output_profile:
            logger.info(f"Using AWS profile for output: {output_profile}")
        if output_endpoint:
            logger.info(f"Using S3 endpoint for output: {output_endpoint}")
        try:
            upload_to_s3(output_dir, s3_output, output_profile, output_endpoint)
            logger.info(f"Successfully uploaded output to {s3_output}")
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    # Cleanup temporary input directory if created
    if temp_input_dir and Path(temp_input_dir).exists():
        logger.info(f"Cleaning up temporary input directory: {temp_input_dir}")
        shutil.rmtree(temp_input_dir)


def extract_frames_with_annotations(
    scene_data: SceneData,
    camera_names: list[str],
    output_dir: str,
    data_path: Path,
    max_frames: int,
    skip_frames: int,
    overlay_camera: bool,
    alpha: float,
    all_camera_models: Dict[str, FThetaCamera],
    all_camera_poses: Dict[str, np.ndarray],
    use_persistent_vbos: bool,
    multi_sample: int,
) -> None:
    """
    Extract individual frames with 3D bounding box annotations.

    This function processes frames from the scene data and extracts:
    - Frame images (JPG format) with optional HD map overlay
    - JSON annotation files containing 3D bounding boxes visible in each frame

    The annotations are filtered by camera frustum with strict visibility checks:
    - Objects must be within 50m from camera
    - Objects must be within the camera frame (no margin)
    - At least 10% of the bbox must be visible

    Args:
        scene_data: Scene data containing dynamic objects and map elements
        camera_names: List of camera names to process
        output_dir: Directory to save extracted frames and annotations
        data_path: Path to data directory containing video files
        max_frames: Maximum number of frames to extract (-1 for all)
        skip_frames: Extract every Nth frame (1 = all frames)
        overlay_camera: If True, overlay camera video on rendered HD map
        alpha: Alpha blending value for overlay
        all_camera_models: Dictionary mapping camera names to FThetaCamera models
        all_camera_poses: Dictionary mapping camera names to pose arrays (N, 4, 4)
        use_persistent_vbos: Use persistent VBOs for rendering performance
        multi_sample: MSAA samples for rendering

    Output Structure:
        output_dir/
        ├── images/
        │   └── <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.jpg
        ├── text/
        │   └── <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.json
        └── meta.json

    Output Files:
        - Frame images: images/<uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.jpg
        - Annotation JSON: text/<uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.json
        - Metadata JSON: meta.json (contains array of entries with id, conversation, media paths)

    JSON Format:
        {
            "frame_id": int,
            "camera": str,
            "camera_params": {"fx": float, "fy": float, "cx": float, "cy": float},
            "annotations": [
                {
                    "label": str,  # e.g., "vehicle_123"
                    "bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
                }
            ]
        }

    Coordinate System:
        - Output: RDF camera coordinates [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
          - RDF convention: Right (+X), Down (+Y), Forward (+Z)
          - x_center, y_center, z_center: Center position in camera coordinate frame (meters)
            - x_center: Right (positive X is to the right of camera)
            - y_center: Down (positive Y is downward)
            - z_center: Forward (positive Z is forward, in front of camera)
          - x_size, y_size, z_size: Bounding box dimensions (meters)
          - roll, pitch, yaw: XYZ Euler angles in radians (rotation in camera frame)

        Note: World coordinates are transformed to camera coordinates using the camera pose.
        According to world_scenario_parquet.md:
        - World origin: Ego vehicle's starting position (first frame) defines world origin (0,0,0)
        - World convention: FLU (Forward=X, Left=Y, Up=Z)
        - Camera convention: RDF (Right=X, Down=Y, Forward=Z)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for images and text (conversation files)
    images_dir = output_path / "images"
    text_dir = output_path / "text"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    # Parse scene_id to extract uuid and timestamps
    # Expected format: <uuid>_<start_ts>_<end_ts>
    scene_id_parts = scene_data.scene_id.split("_")
    if len(scene_id_parts) >= 3:
        uuid_part = scene_id_parts[0]
        start_ts = scene_id_parts[1]
        end_ts = scene_id_parts[2]
    else:
        uuid_part = scene_data.scene_id
        start_ts = "0"
        end_ts = "0"

    # List to store metadata entries for meta.json
    meta_entries = []

    # Check if we have any cameras
    if not camera_names:
        logger.error("No cameras available for frame extraction. Exiting.")
        return

    # Check if camera poses are available (required for coordinate conversion)
    if not all_camera_poses or camera_names[0] not in all_camera_poses:
        logger.error(
            "Camera poses are not available. Frame extraction with 3D annotations requires camera poses. Exiting."
        )
        return

    # Verify all requested cameras have poses
    missing_poses = [name for name in camera_names if name not in all_camera_poses]
    if missing_poses:
        logger.error(f"Camera poses missing for cameras: {missing_poses}. Exiting.")
        return

    # Determine frames to process
    num_frames = len(all_camera_poses[camera_names[0]])
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    # Create overlay renderer if overlaying
    overlay_renderer: Optional[OverlayRenderer] = None
    video_arrays: Dict[str, np.ndarray] = {}
    if overlay_camera:
        selected_camera_models = {name: all_camera_models[name] for name in camera_names}
        overlay_renderer = create_overlay_renderer(
            camera_models=selected_camera_models,
            scene_data=scene_data,
            use_persistent_vbos=use_persistent_vbos,
            multi_sample=multi_sample,
            filter_vehicles_only=True,
        )
        # Load camera videos
        video_arrays = load_camera_videos(
            camera_names=camera_names,
            scene_id=scene_data.scene_id,
            data_path=data_path,
            camera_models=all_camera_models,
        )

    # Process frames
    logger.info(f"Extracting frames (every {skip_frames} frame(s))...")
    frames_extracted = 0

    for frame_id in range(0, num_frames, skip_frames):
        # Process each camera
        for camera_name in camera_names:
            # Extract 3D annotations for this frame, filtered by camera frustum
            camera_model = all_camera_models[camera_name]
            camera_pose = all_camera_poses[camera_name][frame_id]
            annotations = extract_3d_annotations(
                scene_data,
                frame_id,
                camera_model,
                camera_pose,
            )

            mapped_name = camera_name.replace(":", "_")

            # Generate frame filename: <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>
            frame_basename = f"{uuid_part}_{start_ts}_{end_ts}_{mapped_name}_frame_{frame_id:06d}"
            # Save to subdirectories
            frame_image_path = images_dir / f"{frame_basename}.jpg"
            frame_json_path = text_dir / f"{frame_basename}.json"

            # Get the frame image
            if overlay_camera and overlay_renderer is not None:
                # Render HD map frame
                camera_poses_frame = {
                    camera_name: all_camera_poses[camera_name][frame_id] for camera_name in camera_names
                }
                rendered_frames = overlay_renderer.render_frame(camera_poses_frame, frame_id)
                rendered_frame = rendered_frames[camera_name]

                # Blend with camera video if available
                if camera_name in video_arrays and frame_id < len(video_arrays[camera_name]):
                    frame = overlay_renderer.blend_with_video(
                        rendered_frame=rendered_frame,
                        video_frame=video_arrays[camera_name][frame_id],
                        alpha=alpha,
                    )
                else:
                    frame = rendered_frame
            else:
                # Extract from original video without overlay
                mapped_name_check = camera_name.replace(":", "_")
                video_path_patterns = [
                    data_path / f"{scene_data.scene_id}.{mapped_name_check}.mp4",
                    data_path / f"ftheta_{mapped_name_check}" / f"{scene_data.scene_id}.mp4",
                ]
                video_path = None
                for path in video_path_patterns:
                    if path.exists():
                        video_path = path
                        break

                if video_path is None:
                    logger.warning(f"No video found for {camera_name}, skipping frame extraction")
                    continue

                # Load just this frame from video
                if camera_name not in video_arrays:
                    camera_model = all_camera_models[camera_name]
                    h, w = camera_model.height, camera_model.width
                    video_arrays[camera_name] = read_video_simple(video_path.as_posix(), h, w)

                if frame_id < len(video_arrays[camera_name]):
                    frame = video_arrays[camera_name][frame_id]
                else:
                    logger.warning(f"Frame {frame_id} not found in video for {camera_name}")
                    continue

            # Save frame image
            imageio.imwrite(str(frame_image_path), frame, quality=95)

            # Extract camera parameters
            camera_params = extract_camera_params(camera_model)

            # Save annotations JSON
            annotation_data = {
                "frame_id": frame_id,
                "camera": camera_name,
                "camera_params": camera_params,
                "annotations": annotations,
            }
            with open(frame_json_path, "w") as f:
                json.dump(annotation_data, f, indent=2)

            # Add entry to metadata
            meta_entries.append(
                {
                    "id": uuid_part,
                    "conversation": f"text/{frame_basename}.json",
                    "media": f"images/{frame_basename}.jpg",
                }
            )

            frames_extracted += 1

        if (frame_id + 1) % 100 == 0 or frame_id == 0:
            logger.info(f"Extracted frame {frame_id + 1}/{num_frames}")

    # Cleanup
    if overlay_renderer is not None:
        overlay_renderer.cleanup()

    # Write meta.json file
    meta_json_path = output_path / "meta.json"
    with open(meta_json_path, "w") as f:
        json.dump(meta_entries, f, indent=2)
    logger.info(f"Created meta.json with {len(meta_entries)} entries")

    logger.info(f"Extracted {frames_extracted} frames with annotations to {output_path}")
    logger.info(f"  - Images: {images_dir}")
    logger.info(f"  - Annotations: {text_dir}")
    logger.info(f"  - Metadata: {meta_json_path}")


def render_multi_camera_tiled(
    all_camera_models: Dict[str, FThetaCamera],
    all_camera_poses: Dict[str, np.ndarray],
    scene_data: SceneData,
    camera_names: list[str],
    output_dir: str,
    clip_id: str,
    max_frames: int,
    chunk_output: bool,
    overlay_camera: bool,
    alpha: float,
    clipgt_path: Path,
    use_persistent_vbos: bool,
    multi_sample: int,
) -> None:
    """
    Render HD map videos using tiled multi-camera renderer.

    This function uses a high-performance tiled renderer that can render multiple
    cameras in a single OpenGL pass. It supports both single video output and
    chunked output for long sequences.

    Args:
        all_camera_models: Dictionary mapping camera names to FThetaCamera models
        all_camera_poses: Dictionary mapping camera names to pose arrays (N, 4, 4)
        scene_data: Scene data containing HD map and dynamic objects
        camera_names: List of camera names to render
        output_dir: Directory to save rendered videos
        clip_id: Scene/clip identifier for output file naming
        max_frames: Maximum number of frames to render (-1 for all)
        chunk_output: If True, output chunked videos instead of single video
        overlay_camera: If True, overlay camera video on rendered HD map
        alpha: Alpha blending value for overlay (0.0 = full camera, 1.0 = full HD map)
        clipgt_path: Path to data directory containing video files
        use_persistent_vbos: Use persistent VBOs for static geometry
        multi_sample: MSAA samples for anti-aliasing

    Output Files:
        - Single video: <clip_id>_<camera_name>_overlayed.mp4
        - Chunked videos: <clip_id>_<camera_name>_overlayed_chunk_<N>.mp4
        - Original videos: <clip_id>_<camera_name>.mp4 (copied to output)

    Performance Notes:
        - Tiled rendering processes all cameras in one OpenGL pass
        - Chunked output reduces memory usage for long sequences
        - Persistent VBOs improve performance for static geometry
    """

    # Create tiled multi-camera renderer
    logger.info("Creating tiled multi-camera renderer...")

    # Filter camera models to only requested cameras
    selected_camera_models = {name: all_camera_models[name] for name in camera_names}

    # Create renderer (with vehicle filtering if overlaying)
    if overlay_camera:
        # Use overlay renderer for vehicle-only rendering
        overlay_renderer = create_overlay_renderer(
            camera_models=selected_camera_models,
            scene_data=scene_data,
            use_persistent_vbos=use_persistent_vbos,
            multi_sample=multi_sample,
            filter_vehicles_only=True,
        )
        tiled_renderer = overlay_renderer.renderer
    else:
        # Use standard renderer with full scene data
        tiled_renderer = TiledMultiCameraRenderer(
            camera_models=selected_camera_models,
            scene_data=scene_data,
            hdmap_color_version="v3",
            bbox_color_version="v3",
            enable_height_filter=False,
            use_persistent_vbos=use_persistent_vbos,
            multi_sample=multi_sample,
        )
        overlay_renderer = None

    # Determine frames to render
    num_frames = len(all_camera_poses[camera_names[0]])
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    # Load camera videos if overlaying
    video_arrays: Dict[str, np.ndarray] = {}
    if overlay_camera:
        video_arrays = load_camera_videos(
            camera_names=camera_names,
            scene_id=clip_id,
            data_path=clipgt_path,
            camera_models=all_camera_models,
        )

    # Prepare output paths and writers
    output_path = Path(output_dir)
    writers = {}
    output_files = {camera_name: [] for camera_name in camera_names}
    chunk_buffers = {camera_name: [] for camera_name in camera_names}
    current_chunks = {camera_name: 0 for camera_name in camera_names}

    # Create output directory for videos
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy original videos to output directory
    for camera_name in camera_names:
        mapped_name = camera_name.replace(":", "_")
        # Try multiple possible video path patterns
        video_path_patterns = [
            clipgt_path / f"{clip_id}.{mapped_name}.mp4",  # Original ClipGT format
            clipgt_path / f"ftheta_{mapped_name}" / f"{clip_id}.mp4",  # RDS-HQ/MADS format
        ]
        video_path = None
        for path in video_path_patterns:
            if path.exists():
                video_path = path
                break

        if video_path is not None:
            original_output = output_path / f"{clip_id}_{mapped_name}.mp4"
            shutil.copy2(video_path, original_output)
            logger.info(f"Copied original video: {original_output.name}")

    # Process frames
    logger.info(f"Starting tiled multi-camera render for {num_frames} frames...")

    for frame_id in range(num_frames):
        # Prepare camera poses for this frame
        camera_poses_frame = {camera_name: all_camera_poses[camera_name][frame_id] for camera_name in camera_names}

        # Render all cameras in a single OpenGL pass
        rendered_frames = tiled_renderer.render_all_cameras(camera_poses_frame, frame_id)

        # Process each rendered frame
        for camera_name, rendered_frame in rendered_frames.items():
            # Apply overlay if needed
            if (
                overlay_camera
                and overlay_renderer is not None
                and camera_name in video_arrays
                and frame_id < len(video_arrays[camera_name])
            ):
                frame = overlay_renderer.blend_with_video(
                    rendered_frame=rendered_frame,
                    video_frame=video_arrays[camera_name][frame_id],
                    alpha=alpha,
                )
            else:
                frame = rendered_frame

            if chunk_output:
                # Handle chunked output
                chunk_buffers[camera_name].append(frame)

                if len(chunk_buffers[camera_name]) == SETTINGS["TARGET_CHUNK_FRAME"]:
                    # Write chunk
                    mapped_name = camera_name.replace(":", "_")
                    output_file = (
                        output_path / f"{clip_id}_{mapped_name}_overlayed_chunk_{current_chunks[camera_name]}.mp4"
                    )

                    writer = imageio.get_writer(
                        str(output_file),
                        fps=SETTINGS["TARGET_RENDER_FPS"],
                        codec="libx264",
                        macro_block_size=None,
                        ffmpeg_params=["-crf", "18", "-preset", "slow"],
                    )

                    for chunk_frame in chunk_buffers[camera_name]:
                        writer.append_data(chunk_frame)
                    writer.close()

                    output_files[camera_name].append(output_file)
                    logger.info(f"Saved chunk {current_chunks[camera_name]} for {camera_name}")

                    # Prepare for next chunk with overlap
                    if SETTINGS["OVERLAP_FRAME"] > 0:
                        chunk_buffers[camera_name] = chunk_buffers[camera_name][-SETTINGS["OVERLAP_FRAME"] :]
                    else:
                        chunk_buffers[camera_name] = []

                    current_chunks[camera_name] += 1

                    # Check max chunks
                    if SETTINGS.get("MAX_CHUNK", -1) > 0 and current_chunks[camera_name] >= SETTINGS["MAX_CHUNK"]:
                        break
            else:
                # Single video output - initialize writer if needed
                if camera_name not in writers:
                    mapped_name = camera_name.replace(":", "_")
                    output_file = output_path / f"{clip_id}_{mapped_name}_overlayed.mp4"
                    writers[camera_name] = imageio.get_writer(
                        str(output_file),
                        fps=SETTINGS["TARGET_RENDER_FPS"],
                        codec="libx264",
                        macro_block_size=None,
                        ffmpeg_params=["-crf", "18", "-preset", "slow"],
                    )
                    output_files[camera_name] = [output_file]

                writers[camera_name].append_data(frame)

        # Log progress
        if (frame_id + 1) % 100 == 0:
            logger.info(f"Processed {frame_id + 1}/{num_frames} frames")

    # Write remaining frames and cleanup
    if chunk_output:
        # Write remaining chunks
        for camera_name in camera_names:
            if chunk_buffers[camera_name]:
                mapped_name = camera_name.replace(":", "_")
                output_file = output_path / f"{clip_id}_{mapped_name}_overlayed_chunk_{current_chunks[camera_name]}.mp4"

                writer = imageio.get_writer(
                    str(output_file),
                    fps=SETTINGS["TARGET_RENDER_FPS"],
                    codec="libx264",
                    macro_block_size=None,
                    ffmpeg_params=["-crf", "18", "-preset", "slow"],
                )

                for chunk_frame in chunk_buffers[camera_name]:
                    writer.append_data(chunk_frame)
                writer.close()

                output_files[camera_name].append(output_file)
                logger.info(f"Saved final chunk for {camera_name}")
    else:
        # Close single video writers
        for _, writer in writers.items():
            writer.close()

        # Cleanup renderer
        if overlay_renderer is not None:
            overlay_renderer.cleanup()
        else:
            tiled_renderer.cleanup()

        # Report results
        for camera_name in camera_names:
            mapped_name = camera_name.replace(":", "_")
            camera_folder = f"ftheta_{mapped_name}"
            logger.info(
                f"Saved {len(output_files[camera_name])} files for {camera_name} to {output_path / 'hdmap' / camera_folder}"
            )


if __name__ == "__main__":
    main()
