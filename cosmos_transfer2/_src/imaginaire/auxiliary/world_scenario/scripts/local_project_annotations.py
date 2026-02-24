# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Project 3D obstacle annotations onto images using FThetaCamera.

This script:
1. Loads sequence IDs from a file (mandatory)
2. Filters images by matching UUIDs from filenames to sequence IDs
3. Groups frames by scene_id (sequence ID) for efficient processing
4. Downloads scene data from S3 to local temp directory if needed (one per sequence)
5. Loads scene data using load_scene() (once per sequence)
6. Reads 3D obstacle annotations from text directory (RDF camera coordinates)
7. Converts camera coordinates to FLU world coordinates
8. Uses FThetaCamera.ray2pixel_np() to project 3D boxes onto images
9. Draws projected boxes on images and saves results

The script processes frames grouped by sequence ID, loading scene data once per sequence
for efficiency. Supports both local and S3 paths for images, annotations, and scene data.

Usage:
    # Basic usage (local paths)
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
        --sequence-ids-file cosmos3/assets/test_sequence.txt \
        --images-dir cosmos3/dataset/output/mads-sample-03/images \
        --text-dir cosmos3/dataset/output/mads-sample-03/text \
        --input-dir cosmos3/dataset/test/mads-sample \
        --output-dir cosmos3/dataset/output/mads-sample-03/images_annotated
    
    # S3 paths with AWS profile
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
        --sequence-ids-file cosmos3/assets/test_sequence.txt \
        --images-dir s3://bucket/path/to/images \
        --text-dir s3://bucket/path/to/text \
        --input-dir s3://bucket/path/to/scene/data \
        --output-dir /local/output/path \
        --aws-profile team-cosmos-benchmark
    
    # S3 paths with different directories
    uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
        --sequence-ids-file cosmos3/assets/test_sequence.txt \
        --images-dir s3://bucket1/images \
        --text-dir s3://bucket2/annotations/text \
        --input-dir s3://bucket3/scene/data \
        --output-dir /local/output/path
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
from loguru import logger
from PIL import Image
from scipy.spatial.transform import Rotation

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import load_scene
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import SceneData
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.config import SETTINGS
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local import convert_scene_data_for_rendering
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io


def is_s3_path(path: str) -> bool:
    """Check if path is an S3 path."""
    return str(path).startswith("s3://")


def list_files(path: str, suffix: str | None = None, recursive: bool = False) -> list[str]:
    """
    List files from local or S3 directory.

    Args:
        path: Local or S3 directory path
        suffix: Optional file suffix filter (e.g., ".jpg", ".json")
        recursive: Whether to list recursively

    Returns:
        List of file paths (relative to the directory if local, full paths if S3)
    """
    if is_s3_path(path):
        # Ensure path ends with /
        if not path.endswith("/"):
            path = path + "/"

        # List files from S3
        files = list(
            easy_io.list_dir_or_file(
                path,
                list_dir=False,
                list_file=True,
                suffix=suffix,
                recursive=recursive,
            )
        )

        # Convert relative paths to full S3 paths
        return [f"{path.rstrip('/')}/{f}" if not f.startswith("s3://") else f for f in files]
    else:
        # Local path
        path_obj = Path(path)
        if not path_obj.exists():
            return []

        pattern = f"**/*{suffix}" if suffix and recursive else f"*{suffix}" if suffix else "*"
        if recursive:
            files = list(path_obj.glob(pattern))
        else:
            files = list(path_obj.glob(pattern))

        return [str(f) for f in files if f.is_file()]


def load_json_file(path: str) -> dict[str, Any]:
    """
    Load JSON file from local or S3 path.

    Args:
        path: Local or S3 file path

    Returns:
        Loaded JSON data
    """
    if is_s3_path(path):
        return easy_io.load(path)
    else:
        with open(path, "r") as f:
            return json.load(f)


def load_image_file(path: str) -> Image.Image:
    """
    Load image file from local or S3 path.

    Args:
        path: Local or S3 file path

    Returns:
        PIL Image object
    """
    if is_s3_path(path):
        image_bytes = easy_io.load(path, mode="rb")
        return Image.open(BytesIO(image_bytes))
    else:
        return Image.open(path)


def load_npy_file(path: str) -> np.ndarray:
    """
    Load numpy array from local or S3 path.

    Args:
        path: Local or S3 file path

    Returns:
        Numpy array
    """
    if is_s3_path(path):
        npy_bytes = easy_io.load(path, mode="rb")
        return np.load(BytesIO(npy_bytes))
    else:
        return np.load(path)


def path_exists(path: str) -> bool:
    """Check if path exists (local or S3)."""
    if is_s3_path(path):
        return easy_io.exists(path)
    else:
        return Path(path).exists()


def get_path_basename(path: str) -> str:
    """Get basename from local or S3 path."""
    return Path(path).name


def get_path_stem(path: str) -> str:
    """Get stem (filename without extension) from local or S3 path."""
    return Path(path).stem


def parse_frame_filename(filename: str) -> dict[str, Any] | None:
    """
    Parse frame filename to extract metadata.

    Expected format: <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.jpg

    Args:
        filename: Frame filename (with or without extension)

    Returns:
        Dictionary with uuid, start_ts, end_ts, camera_name, frame_id, or None if parsing fails
    """
    basename = Path(filename).stem

    # Pattern: <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>
    pattern = r"^([0-9a-f-]+)_(\d+)_(\d+)_(.+)_frame_(\d+)$"
    match = re.match(pattern, basename)

    if match:
        uuid, start_ts, end_ts, camera_name, frame_id = match.groups()
        return {
            "uuid": uuid,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "camera_name": camera_name,
            "frame_id": int(frame_id),
            "scene_id": f"{uuid}_{start_ts}_{end_ts}",
        }

    return None


def camera_to_world_coordinates(
    bbox_3d_camera: list[float],
    camera_pose: np.ndarray,
) -> list[float]:
    """
    Transform 3D bounding box from RDF camera coordinates to FLU world coordinates.

    This is the inverse of world_to_camera_coordinates().

    Coordinate Systems:
    - Input (RDF camera): Right=X, Down=Y, Forward=Z
    - Output (FLU world): Forward=X, Left=Y, Up=Z

    Args:
        bbox_3d_camera: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
                       in RDF camera coordinates
                       - x_center, y_center, z_center: Center position in camera frame (meters)
                       - x_size, y_size, z_size: Bounding box dimensions (meters)
                       - roll, pitch, yaw: XYZ Euler angles in radians (rotation in camera frame)
        camera_pose: (4, 4) camera-to-world transformation matrix

    Returns:
        [x, y, z, x_size, y_size, z_size, roll, pitch, yaw] in FLU world coordinates
        - x, y, z: Center position in world coordinate frame (meters)
        - x_size, y_size, z_size: Bounding box dimensions (meters)
        - roll, pitch, yaw: XYZ Euler angles in radians (rotation in world frame)
    """
    if len(bbox_3d_camera) < 6:
        return bbox_3d_camera

    # Extract center and size
    center_camera = np.array(bbox_3d_camera[:3], dtype=np.float32)
    size = bbox_3d_camera[3:6]

    # Transform center: world_coords = camera_pose @ [camera_coords, 1]
    # camera_pose is camera-to-world transformation matrix
    center_camera_homogeneous = np.hstack([center_camera, 1.0])
    center_world_homogeneous = camera_pose @ center_camera_homogeneous
    center_world = center_world_homogeneous[:3]

    # Transform rotation angles
    # The rotation matrix R_bbox_camera rotates vectors in camera frame
    # To get rotation in world frame: R_bbox_world = R_camera_to_world @ R_bbox_camera @ R_world_to_camera
    # This is the inverse of: R_bbox_camera = R_world_to_camera @ R_bbox_world @ R_camera_to_world
    if len(bbox_3d_camera) >= 9:
        roll_camera, pitch_camera, yaw_camera = bbox_3d_camera[6], bbox_3d_camera[7], bbox_3d_camera[8]

        # Convert Euler angles to rotation matrix (XYZ order) in camera frame
        R_bbox_camera = Rotation.from_euler("xyz", [roll_camera, pitch_camera, yaw_camera], degrees=False).as_matrix()

        # Transform rotation matrix from camera frame to world frame
        # R_bbox_world rotates vectors in world frame the same way R_bbox_camera rotates in camera frame
        R_camera_to_world = camera_pose[:3, :3]
        R_world_to_camera = np.linalg.inv(camera_pose)[:3, :3]
        R_bbox_world = R_camera_to_world @ R_bbox_camera @ R_world_to_camera

        # Convert back to Euler angles (XYZ order) in world frame
        rot = Rotation.from_matrix(R_bbox_world)
        euler_world = rot.as_euler("xyz", degrees=False)
        roll_world, pitch_world, yaw_world = euler_world[0], euler_world[1], euler_world[2]
    else:
        roll_world, pitch_world, yaw_world = 0.0, 0.0, 0.0

    # Note: Dimensions don't change, but we need to map them correctly:
    # RDF camera: x_size=right, y_size=down, z_size=forward
    # FLU world: x_size=forward, y_size=left, z_size=up
    # The mapping depends on how the bbox was originally defined, but typically:
    # - RDF x_size (right) -> FLU y_size (left, but flipped)
    # - RDF y_size (down) -> FLU z_size (up, but flipped)
    # - RDF z_size (forward) -> FLU x_size (forward)
    # However, since dimensions are scalars, we keep them as-is and let the coordinate system handle it
    # In practice, the dimensions should be interpreted in their respective coordinate systems

    # Output format: [x, y, z, x_size, y_size, z_size, roll, pitch, yaw] in FLU world coordinates
    return [
        float(center_world[0]),  # x in world (FLU: Forward)
        float(center_world[1]),  # y in world (FLU: Left)
        float(center_world[2]),  # z in world (FLU: Up)
        float(size[0]),  # x_size (interpreted in FLU: length along Forward/X)
        float(size[1]),  # y_size (interpreted in FLU: width along Left/Y)
        float(size[2]),  # z_size (interpreted in FLU: height along Up/Z)
        float(roll_world),  # roll in world frame (rotation around Forward/X)
        float(pitch_world),  # pitch in world frame (rotation around Left/Y)
        float(yaw_world),  # yaw in world frame (rotation around Up/Z)
    ]


def get_bbox_corners_3d(
    bbox_3d: list[float],
    coordinate_system: str = "camera",
) -> np.ndarray:
    """
    Get 8 corners of 3D bounding box.

    Args:
        bbox_3d: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
        coordinate_system: "camera" (RDF) or "world" (FLU)

    Returns:
        (8, 3) array of corner coordinates
    """
    x_center, y_center, z_center = bbox_3d[0], bbox_3d[1], bbox_3d[2]
    x_size, y_size, z_size = bbox_3d[3], bbox_3d[4], bbox_3d[5]
    roll, pitch, yaw = bbox_3d[6], bbox_3d[7], bbox_3d[8] if len(bbox_3d) >= 9 else (0.0, 0.0, 0.0)

    # Create 8 corners in local space (centered at origin)
    if coordinate_system == "camera":
        # RDF: X=right, Y=down, Z=forward
        corners_local = np.array(
            [
                [-x_size / 2, -y_size / 2, -z_size / 2],  # 0: back-left-bottom
                [x_size / 2, -y_size / 2, -z_size / 2],  # 1: back-right-bottom
                [x_size / 2, y_size / 2, -z_size / 2],  # 2: back-right-top
                [-x_size / 2, y_size / 2, -z_size / 2],  # 3: back-left-top
                [-x_size / 2, -y_size / 2, z_size / 2],  # 4: front-left-bottom
                [x_size / 2, -y_size / 2, z_size / 2],  # 5: front-right-bottom
                [x_size / 2, y_size / 2, z_size / 2],  # 6: front-right-top
                [-x_size / 2, y_size / 2, z_size / 2],  # 7: front-left-top
            ],
            dtype=np.float32,
        )
    else:
        # FLU: X=forward, Y=left, Z=up
        corners_local = np.array(
            [
                [-x_size / 2, -y_size / 2, -z_size / 2],  # 0: back-left-bottom
                [x_size / 2, -y_size / 2, -z_size / 2],  # 1: front-left-bottom
                [x_size / 2, y_size / 2, -z_size / 2],  # 2: front-right-bottom
                [-x_size / 2, y_size / 2, -z_size / 2],  # 3: back-right-bottom
                [-x_size / 2, -y_size / 2, z_size / 2],  # 4: back-left-top
                [x_size / 2, -y_size / 2, z_size / 2],  # 5: front-left-top
                [x_size / 2, y_size / 2, z_size / 2],  # 6: front-right-top
                [-x_size / 2, y_size / 2, z_size / 2],  # 7: back-right-top
            ],
            dtype=np.float32,
        )

    # Apply rotation if provided
    if len(bbox_3d) >= 9 and (roll != 0.0 or pitch != 0.0 or yaw != 0.0):
        rot_matrix = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
        corners_local = corners_local @ rot_matrix.T

    # Translate to center position
    center = np.array([x_center, y_center, z_center], dtype=np.float32)
    corners_3d = corners_local + center

    return corners_3d


def project_bbox_corners_to_image(
    bbox_3d_world: list[float],
    camera_model: FThetaCamera,
    camera_pose: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Project 3D bounding box corners to 2D image coordinates using FThetaCamera.

    This function converts FLU world coordinates to RDF camera coordinates before projecting.

    Args:
        bbox_3d_world: [x, y, z, x_size, y_size, z_size, roll, pitch, yaw]
                      in FLU world coordinates
        camera_model: FThetaCamera model
        camera_pose: (4, 4) camera-to-world transformation matrix

    Returns:
        List of 8 (x, y) tuples representing projected pixel coordinates
    """
    # Get 3D corners in FLU world coordinates
    corners_3d_world = get_bbox_corners_3d(bbox_3d_world, coordinate_system="world")

    # Transform corners from FLU world coordinates to RDF camera coordinates
    # world_to_camera = inv(camera_pose)
    world_to_camera = np.linalg.inv(camera_pose)
    corners_world_homogeneous = np.hstack([corners_3d_world, np.ones((8, 1))])
    corners_camera_homogeneous = (world_to_camera @ corners_world_homogeneous.T).T
    corners_3d_camera = corners_camera_homogeneous[:, :3]

    # Filter to corners in front of camera (z > 0 in RDF, where Z is forward)
    valid_mask = corners_3d_camera[:, 2] > 0
    if not np.any(valid_mask):
        return [(float("nan"), float("nan"))] * 8  # All corners behind camera

    valid_corners = corners_3d_camera[valid_mask]

    # Project using FThetaCamera.ray2pixel_np()
    # Input: rays in RDF camera coordinates (Right=X, Down=Y, Forward=Z)
    # Output: pixel coordinates (x, y)
    pixels = camera_model.ray2pixel_np(valid_corners)

    # Create list of (x, y) tuples for all 8 corners
    bbox_2d = []
    corner_idx = 0
    for i in range(8):
        if valid_mask[i]:
            bbox_2d.append((float(pixels[corner_idx, 0]), float(pixels[corner_idx, 1])))
            corner_idx += 1
        else:
            bbox_2d.append((float("nan"), float("nan")))

    return bbox_2d


def draw_bbox_on_image(
    image: np.ndarray,
    bbox_2d: list[tuple[float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw 3D bounding box on image using projected 2D corners.

    Args:
        image: (H, W, 3) uint8 image array
        bbox_2d: List of 8 (x, y) tuples (some may be NaN)
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        Image with drawn bbox
    """
    img = image.copy()

    # Define edges of the bounding box (12 edges connecting 8 corners)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # Bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # Top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]

    # Draw edges
    for i, j in edges:
        pt1 = bbox_2d[i]
        pt2 = bbox_2d[j]

        # Skip if either point is NaN or out of bounds
        if np.isnan(pt1[0]) or np.isnan(pt1[1]) or np.isnan(pt2[0]) or np.isnan(pt2[1]):
            continue

        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])

        # Check bounds
        h, w = img.shape[:2]
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def download_scene_data_to_temp(
    input_dir: str,
    uuid: str,
    aws_profile: str,
) -> tuple[str, Path | None]:
    """
    Download scene data from S3 to a temporary directory, or return local path.

    Args:
        input_dir: Input directory path (local or S3)
        uuid: UUID of the scene sequence
        aws_profile: AWS profile to use for S3 operations

    Returns:
        Tuple of (input_path_for_load, temp_input_dir)
        - input_path_for_load: Path to use for loading scene data
        - temp_input_dir: Path to temp directory if created, None otherwise
    """
    if is_s3_path(input_dir):
        s3_source_path = f"{input_dir.rstrip('/')}/{uuid}/"
        logger.info(f"S3 source path: {s3_source_path}")

        # Check if UUID-specific path exists using AWS CLI
        try:
            cmd = ["aws", "s3", "ls", s3_source_path, "--profile", aws_profile]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                # Path doesn't exist, fallback to base input_dir
                s3_source_path = input_dir.rstrip("/") + "/"
                logger.info(f"UUID-specific path not found, using base input_dir: {s3_source_path}")
        except Exception as e:
            logger.warning(f"Failed to check S3 path with AWS CLI: {e}, using base input_dir")
            s3_source_path = input_dir.rstrip("/") + "/"

        # Copy S3 sequence to local temp directory
        temp_input_dir = Path(tempfile.mkdtemp(prefix=f"scene_data_{uuid}_"))
        logger.info(f"Downloading scene data from S3 to temp directory: {temp_input_dir}")

        try:
            # Use aws s3 sync to copy the directory
            cmd = [
                "aws",
                "s3",
                "sync",
                s3_source_path,
                str(temp_input_dir),
                "--profile",
                aws_profile,
                "--no-progress",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Successfully downloaded scene data to {temp_input_dir}")
            return str(temp_input_dir), temp_input_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download scene data from S3: {e.stderr}")
            # Cleanup temp directory
            if temp_input_dir.exists():
                shutil.rmtree(temp_input_dir)
            raise
    else:
        # Local path
        input_path_for_load = str(Path(input_dir) / uuid)
        if not path_exists(input_path_for_load):
            # Fallback to base input_dir
            input_path_for_load = input_dir
            logger.info(f"UUID-specific path not found, using base input_dir: {input_path_for_load}")
        return input_path_for_load, None


def process_single_frame(
    image_path: str,
    annotation_path: str,
    scene_data: SceneData,
    camera_model: FThetaCamera,
    camera_pose: np.ndarray,
    output_path: Path,
) -> bool:
    """
    Process a single frame: load annotation, add to SceneData, project, and draw.

    Args:
        image_path: Path to input image (local or S3)
        annotation_path: Path to annotation JSON file (local or S3)
        scene_data: SceneData object (will be updated with annotations)
        camera_model: FThetaCamera model for projection
        camera_pose: (4, 4) camera-to-world transformation matrix
        output_path: Path to save output image (must be local)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load annotation
        annotation = load_json_file(annotation_path)

        # Load image
        img = load_image_file(image_path)
        img_array = np.array(img)

        # Get annotations (bbox_3d in RDF camera coordinates)
        annotations = annotation.get("annotations", [])

        # Process each annotation
        for ann in annotations:
            bbox_3d_camera = ann.get("bbox_3d", [])
            if len(bbox_3d_camera) < 6:
                continue

            # Convert from RDF camera coordinates to FLU world coordinates
            bbox_3d_world = camera_to_world_coordinates(bbox_3d_camera, camera_pose)

            # Project 3D box corners to 2D using FThetaCamera
            # Now we work in FLU world coordinates and convert to RDF camera coords for projection
            bbox_2d = project_bbox_corners_to_image(bbox_3d_world, camera_model, camera_pose)

            # Draw bbox on image
            img_array = draw_bbox_on_image(img_array, bbox_2d, color=(0, 255, 0), thickness=2)

            # Log conversion for debugging
            logger.debug(f"RDF camera coords: {bbox_3d_camera[:3]} -> FLU world coords: {bbox_3d_world[:3]}")

        # Save result (output_path must be local)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_img = Image.fromarray(img_array)
        output_img.save(output_path)

        return True

    except Exception as e:
        image_name = get_path_basename(image_path)
        logger.error(f"Error processing {image_name}: {e}")
        logger.error(traceback.format_exc())
        return False


@click.command()
@click.option(
    "--sequence-ids-file",
    type=str,
    required=True,
    help="Path to file containing sequence IDs (UUIDs) to process, one per line.",
)
@click.option(
    "--images-dir",
    type=str,
    required=True,
    help="Directory containing input images (local or S3 path)",
)
@click.option(
    "--text-dir",
    type=str,
    required=True,
    help="Directory containing annotation JSON files (local or S3 path)",
)
@click.option(
    "--input-dir",
    type=str,
    default="cosmos3/dataset/test/mads-sample",
    help="Directory containing original scene data (local or S3 path)",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for projected images (must be local)",
)
@click.option(
    "--aws-profile",
    type=str,
    default="team-cosmos-benchmark",
    help="AWS profile to use for S3 operations",
)
def main(
    sequence_ids_file: str,
    images_dir: str,
    text_dir: str,
    input_dir: str,
    output_dir: str | None,
    aws_profile: str,
) -> None:
    """
    Project 3D obstacle annotations onto images using FThetaCamera.

    Args:
        sequence_ids_file: Path to file containing sequence IDs (UUIDs) to process, one per line.
                          Required. Can be local or S3 path.
        images_dir: Directory containing input images (local or S3 path). Required.
        text_dir: Directory containing annotation JSON files (local or S3 path). Required.
        input_dir: Directory containing original scene data (local or S3 path).
                  For S3 paths, downloads to temp directory before processing.
        output_dir: Output directory for projected images (must be local).
                   If not provided, defaults to images_dir parent / images_annotated.
        aws_profile: AWS profile to use for S3 operations. Defaults to team-cosmos-benchmark.

    The script:
    1. Loads sequence IDs from file and filters images by matching UUIDs
    2. Groups frames by scene_id for efficient processing
    3. Downloads scene data from S3 to temp directory if needed (one per sequence)
    4. Loads scene data once per sequence
    5. Reads 3D annotations from text directory
    6. Projects 3D boxes onto images using FThetaCamera
    7. Saves annotated images to output directory
    8. Cleans up temp directories after each sequence
    """
    # Configure easy_io S3 backend with AWS profile
    os.environ["AWS_PROFILE"] = aws_profile
    logger.info(f"Using AWS profile: {aws_profile}")

    # Configure easy_io backend to use the profile
    # Note: Setting AWS_PROFILE environment variable should be sufficient for boto3
    # The backend configuration is optional and may fail if MSC config is not available
    try:
        easy_io.set_s3_backend(
            backend_args={
                "profile": aws_profile,
            }
        )
        logger.debug(f"Configured easy_io backend with profile: {aws_profile}")
    except Exception as e:
        # This is expected if MSC config is not available - AWS_PROFILE env var will be used instead
        logger.debug(f"Could not configure easy_io backend with profile (using AWS_PROFILE env var instead): {e}")

    # Set default output directory (must be local)
    if output_dir is None:
        if is_s3_path(images_dir):
            # For S3 images, use a local temp directory
            output_dir = str(Path(tempfile.gettempdir()) / "projected_annotations")
        else:
            images_path = Path(images_dir)
            output_dir = str(images_path.parent / "images_annotated")

    if is_s3_path(output_dir):
        raise ValueError("Output directory must be a local path, not S3")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not path_exists(images_dir):
        logger.error(f"Images directory does not exist: {images_dir}")
        return

    if not path_exists(text_dir):
        logger.error(f"Text directory does not exist: {text_dir}")
        return

    # Find all image files
    image_files = sorted(list_files(images_dir, suffix=".jpg"))

    if len(image_files) == 0:
        logger.warning(f"No image files found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} image file(s)")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Text annotations directory: {text_dir}")
    logger.info(f"Input scene data directory: {input_dir}")
    logger.info(f"Output directory: {output_path}")

    # Load sequence IDs from file (mandatory)
    if is_s3_path(sequence_ids_file):
        content = easy_io.load(sequence_ids_file, mode="r")
        sequence_ids_set = {line.strip() for line in content.splitlines() if line.strip()}
    else:
        with open(sequence_ids_file, "r") as f:
            sequence_ids_set = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(sequence_ids_set)} sequence IDs from file: {sequence_ids_file}")

    # Group image files by scene_id (sequence ID), filtering by sequence IDs from file
    files_by_scene_id: dict[str, list[str]] = {}
    for image_path_str in image_files:
        image_name = get_path_basename(image_path_str)
        frame_meta = parse_frame_filename(image_name)
        if frame_meta:
            scene_id = frame_meta["scene_id"]
            uuid = frame_meta["uuid"]

            # Only process sequences that match IDs from file
            if uuid not in sequence_ids_set:
                continue

            if scene_id not in files_by_scene_id:
                files_by_scene_id[scene_id] = []
            files_by_scene_id[scene_id].append(image_path_str)

    logger.info(f"Found {len(files_by_scene_id)} unique scene ID(s)")
    for scene_id, files in files_by_scene_id.items():
        logger.info(f"  - {scene_id}: {len(files)} frame(s)")

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Process each scene_id group
    for scene_id, scene_image_files in files_by_scene_id.items():
        logger.info(f"{'=' * 80}")
        logger.info(f"Processing scene ID: {scene_id} ({len(scene_image_files)} frames)")
        logger.info(f"{'=' * 80}")

        # Extract UUID from scene_id (scene_id format: uuid_start_ts_end_ts)
        scene_id_parts = scene_id.split("_")
        uuid = scene_id_parts[0] if len(scene_id_parts) > 0 else scene_id

        # Download scene data to temp directory if S3, or get local path
        try:
            input_path_for_load, temp_input_dir = download_scene_data_to_temp(
                input_dir=input_dir,
                uuid=uuid,
                aws_profile=aws_profile,
            )
        except Exception as e:
            logger.error(f"Failed to prepare scene data for {scene_id}: {e}")
            skipped_count += len(scene_image_files)
            continue

        logger.info(f"Loading scene data from: {input_path_for_load}")

        # Load scene data for this sequence (once per scene_id)
        try:
            scene_data = load_scene(
                input_path_for_load,
                camera_names=None,  # Load all cameras
                max_frames=-1,
                input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
                resize_resolution_hw=SETTINGS["RESIZE_RESOLUTION"],
            )
            logger.info(f"Loaded scene: {scene_data.scene_id}")
            logger.info(f"  - Frames: {scene_data.num_frames}")
            logger.info(f"  - Dynamic objects: {len(scene_data.dynamic_objects)}")
            logger.info(f"  - Cameras: {len(scene_data.camera_models)}")
        except Exception as e:
            logger.error(f"Failed to load scene data for {scene_id}: {e}")
            logger.error(traceback.format_exc())
            # Cleanup temp directory if it was created
            if temp_input_dir and Path(temp_input_dir).exists():
                shutil.rmtree(temp_input_dir)
            skipped_count += len(scene_image_files)
            continue

        # Convert scene data for rendering (once per scene_id)
        camera_names = list(scene_data.camera_models.keys())
        all_camera_models, all_camera_poses = convert_scene_data_for_rendering(
            scene_data,
            camera_names,
            SETTINGS["RESIZE_RESOLUTION"],
        )

        # Process all frames for this scene_id
        for image_path_str in scene_image_files:
            # Find corresponding annotation file
            image_stem = get_path_stem(image_path_str)
            if is_s3_path(text_dir):
                annotation_path = f"{text_dir.rstrip('/')}/{image_stem}.json"
            else:
                annotation_path = str(Path(text_dir) / f"{image_stem}.json")

            if not path_exists(annotation_path):
                image_name = get_path_basename(image_path_str)
                logger.warning(f"No annotation found for {image_name}, skipping...")
                skipped_count += 1
                continue

            # Parse frame metadata
            image_name = get_path_basename(image_path_str)
            frame_meta = parse_frame_filename(image_name)
            if frame_meta is None:
                logger.warning(f"Failed to parse frame filename: {image_name}, skipping...")
                skipped_count += 1
                continue

            camera_name = frame_meta["camera_name"]
            frame_id = frame_meta["frame_id"]

            # Get camera model and pose
            if camera_name not in all_camera_models:
                logger.warning(f"Camera {camera_name} not found in scene data, skipping...")
                skipped_count += 1
                continue

            camera_model = all_camera_models[camera_name]

            # Get camera pose from annotation JSON or scene data
            camera_pose = None

            # Try from annotation JSON
            ann_data = load_json_file(annotation_path)
            if "camera_pose" in ann_data:
                camera_pose = np.array(ann_data["camera_pose"], dtype=np.float32)

            # Fallback to scene data
            if camera_pose is None:
                if camera_name in all_camera_poses and frame_id < len(all_camera_poses[camera_name]):
                    camera_pose = all_camera_poses[camera_name][frame_id]
                else:
                    logger.warning(f"No camera pose available for {image_name}, skipping...")
                    skipped_count += 1
                    continue

            # Generate output path (must be local)
            output_image_path = output_path / image_name

            if (processed_count + 1) % 10 == 0 or processed_count == 0:
                logger.info(f"Processing {image_name} ({processed_count + 1}/{len(image_files)})...")

            # Process frame
            success = process_single_frame(
                image_path=image_path_str,
                annotation_path=annotation_path,
                scene_data=scene_data,
                camera_model=camera_model,
                camera_pose=camera_pose,
                output_path=output_image_path,
            )

            if success:
                processed_count += 1
            else:
                failed_count += 1

        # Cleanup temp directory after processing this scene
        if temp_input_dir and Path(temp_input_dir).exists():
            logger.info(f"Cleaning up temp directory: {temp_input_dir}")
            shutil.rmtree(temp_input_dir)

        logger.info(f"Completed scene {scene_id}: processed {processed_count} frames so far")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing complete:")
    logger.info(f"  - Processed: {processed_count}/{len(image_files)}")
    logger.info(f"  - Skipped: {skipped_count}")
    logger.info(f"  - Failed: {failed_count}")
    logger.info(f"  - Output directory: {output_path}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
