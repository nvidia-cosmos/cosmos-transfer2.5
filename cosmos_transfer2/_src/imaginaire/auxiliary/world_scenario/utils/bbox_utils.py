# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
File modified from https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_bbox_colors(version: str = "v3") -> Dict:
    """
    Load bbox colors based on the specified version.

    Args:
        version: str, version key from config_color_bbox.json

    Returns:
        dict: Color configuration for the specified version
    """
    # Try to find the config file in the local color_scheme directory
    config_path = Path(__file__).parent.parent / "color_scheme" / "config_color_bbox.json"
    with open(config_path, "r") as f:
        bbox_config = json.load(f)

    if version not in bbox_config:
        available_versions = list(bbox_config.keys())
        raise ValueError(f"Version '{version}' not found in bbox config. Available versions: {available_versions}")

    return bbox_config[version]


def interpolate_pose(prev_pose: np.ndarray, next_pose: np.ndarray, t: float) -> np.ndarray:
    """
    new pose = (1 - t) * prev_pose + t * next_pose.
    - linear interpolation for translation
    - slerp interpolation for rotation

    Args:
        prev_pose: np.ndarray, shape (4, 4), dtype=np.float32, previous pose
        next_pose: np.ndarray, shape (4, 4), dtype=np.float32, next pose
        t: float, interpolation factor

    Returns:
        np.ndarray, shape (4, 4), dtype=np.float32, interpolated pose

    Note:
        if input is list, also return list.
    """
    input_is_list = isinstance(prev_pose, list)
    prev_pose = np.array(prev_pose)
    next_pose = np.array(next_pose)

    prev_translation = prev_pose[:3, 3]
    next_translation = next_pose[:3, 3]
    translation = (1 - t) * prev_translation + t * next_translation

    prev_rotation = Rotation.from_matrix(prev_pose[:3, :3])
    next_rotation = Rotation.from_matrix(next_pose[:3, :3])

    times = [0, 1]
    rotations = Rotation.from_quat([prev_rotation.as_quat(), next_rotation.as_quat()])
    rotation = Slerp(times, rotations)(t)

    new_pose = np.eye(4)
    new_pose[:3, :3] = rotation.as_matrix()
    new_pose[:3, 3] = translation

    if input_is_list:
        return new_pose.tolist()
    else:
        return new_pose


def build_cuboid_bounding_box(
    dimXMeters: float,
    dimYMeters: float,
    dimZMeters: float,
    cuboid_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Args
        dimXMeters, dimYMeters, dimZMeters: float, the dimensions of the cuboid
        cuboid_transform: 4x4 numpy array, the transformation matrix from the cuboid coordinate to the other coordinate

        z
        ^
        |   y
        | /
        |/
        o----------> x  (heading)

           3 ---------------- 0
          /|                 /|
         / |                / |
        2 ---------------- 1  |
        |  |               |  |
        |  7 ------------- |- 4
        | /                | /
        6 ---------------- 5

    Returns
        8x3 numpy array: the 8 vertices of the cuboid
    """
    if cuboid_transform is None:
        cuboid_transform = np.eye(4)

    # Build the cuboid bounding box
    cuboid = np.array(
        [
            [dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
            [dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
            [-dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
            [-dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
            [dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2],
            [dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
            [-dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
            [-dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2],
        ]
    )
    cuboid = np.hstack([cuboid, np.ones((8, 1))])  # [8, 4]
    cuboid = np.dot(cuboid_transform, cuboid.T).T
    return cuboid[:, :3]
