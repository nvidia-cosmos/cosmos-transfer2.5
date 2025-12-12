# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Traffic light utilities for rendering.

This module provides functions for:
- Loading traffic light color schemes
- Creating geometry objects for traffic light rendering

Following the pattern from cosmos-av-sample-toolkits.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.graphics_utils import Polygon2D


def load_traffic_light_colors(version: str = "v2") -> Dict:
    """
    Load traffic light colors based on the specified version.

    Args:
        version: str, version key from config_color_traffic_light.json
                Available versions: 'v2'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent.parent / "color_scheme" / "config_color_traffic_light.json"
    with open(config_path, "r") as f:
        tl_config = json.load(f)

    if version not in tl_config:
        available_versions = list(tl_config.keys())
        raise ValueError(
            f"Version '{version}' not found in traffic light config. Available versions: {available_versions}"
        )

    return tl_config[version]


def prepare_traffic_light_status_data_clipgt(
    traffic_lights: List[np.ndarray],
    traffic_light_status_file: Optional[Path] = None,
    traffic_light_color_version: str = "v2",
) -> Tuple[List[np.ndarray], Optional[Dict], Dict]:
    """
    Prepare traffic light status data for ClipGT format.
    Since ClipGT doesn't include per-frame status, we return None for status_dict.

    This follows the pattern of cosmos-av-sample-toolkits prepare_traffic_light_status_data
    but adapted for ClipGT data structure.

    Args:
        traffic_lights: List of traffic light polylines
        traffic_light_status_file: Optional path to status file (not used in ClipGT)
        traffic_light_color_version: Color version

    Returns:
        tuple: (position_list, status_dict, tl_status_to_rgb)
               For ClipGT, status_dict is None (will use unknown color)
    """
    tl_status_to_rgb = load_traffic_light_colors(traffic_light_color_version)

    # ClipGT doesn't have per-frame status data, so return None for status_dict
    # This will use 'unknown' color in the rendering function
    return traffic_lights, None, tl_status_to_rgb


def create_traffic_light_status_geometry_objects_from_data(
    traffic_light_position_list: List[np.ndarray],
    traffic_light_per_frame_status_dict: Optional[Dict],
    frame_id: int,
    camera_pose: np.ndarray,
    camera_model: Any,  # Camera model instance (FThetaCamera or similar)
    tl_status_to_rgb: Dict,
) -> List[Polygon2D]:
    """
    Build geometry objects (Polygon2D) for traffic lights for a single frame.

    This follows the exact pattern from cosmos-av-sample-toolkits.
    When status data is not available, uses 'unknown' color (black/zeros).

    Args:
        traffic_light_position_list: List of traffic light polylines
        traffic_light_per_frame_status_dict: Per-frame status dict (None uses unknown color)
        frame_id: Current frame ID
        camera_pose: Camera pose matrix (4x4)
        camera_model: Camera model
        tl_status_to_rgb: Mapping from status to RGB values

    Returns:
        List of Polygon2D objects for traffic lights
    """
    if not traffic_light_position_list:
        return []

    polygon_vertices = []
    polygon_colors = []

    for traffic_light_index, tl_polyline in enumerate(traffic_light_position_list):
        if traffic_light_per_frame_status_dict is None:
            # Use unknown color for traffic light without status
            # This is exactly what cosmos-av-sample-toolkits does
            signal_render_color = np.array(tl_status_to_rgb["unknown"]) / 255.0
        else:
            # Use actual status data (for future when available)
            this_frame_status = traffic_light_per_frame_status_dict[str(traffic_light_index)]
            signal_state = this_frame_status["state"][frame_id]
            signal_render_color = np.array(tl_status_to_rgb[signal_state]) / 255.0

        polygon_vertices.append(tl_polyline)
        polygon_colors.append(signal_render_color)

    # Project all points together for efficiency
    if len(polygon_vertices) > 0:
        all_polygon_vertices = np.concatenate(polygon_vertices, axis=0)
        all_xy_and_depth = camera_model.get_xy_and_depth(all_polygon_vertices, camera_pose)

        # Reshape based on 16 vertices per traffic light (from cuboid3d_to_polyline)
        all_xy_and_depth = all_xy_and_depth.reshape(-1, 16, 3)

        geometry_objects = []
        for tl_index, xy_and_depth in enumerate(all_xy_and_depth):
            if np.all(xy_and_depth[:, 2] < 0):
                continue
            geometry_objects.append(Polygon2D(xy_and_depth, base_color=polygon_colors[tl_index]))

        return geometry_objects

    return []
