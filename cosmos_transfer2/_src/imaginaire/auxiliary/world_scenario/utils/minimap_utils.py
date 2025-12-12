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
from typing import Dict, Final, Optional

import numpy as np


def load_hdmap_colors(version: str = "v1") -> Dict:
    """
    Load hdmap colors based on the specified version.

    Args:
        version: str, version key from config_color_hdmap.json
                Available versions: 'v1', 'v3'

    Returns:
        dict: Color configuration for the specified version
    """
    # Try to find the config file in the local color_scheme directory
    config_path = Path(__file__).parent.parent / "color_scheme" / "config_color_hdmap.json"
    with open(config_path, "r") as f:
        hdmap_config = json.load(f)

    if version not in hdmap_config:
        available_versions = list(hdmap_config.keys())
        raise ValueError(f"Version '{version}' not found in hdmap config. Available versions: {available_versions}")

    return hdmap_config[version]


def load_laneline_colors() -> Dict:
    """
    Load laneline type-specific colors.

    Returns:
        dict: Color configuration for laneline types
    """
    config_path = Path(__file__).parent.parent / "color_scheme" / "config_color_geometry_laneline.json"
    with open(config_path, "r") as f:
        laneline_config = json.load(f)
    return laneline_config


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


MINIMAP_TO_TYPE: Final = {
    "lanelines": "polyline",
    "lanes": "polyline",
    "poles": "polyline",
    "road_boundaries": "polyline",
    "wait_lines": "polyline",
    "crosswalks": "polygon",
    "road_markings": "polygon",
    "traffic_signs": "cuboid3d",
    "traffic_lights": "cuboid3d",
    "intersection_areas": "polygon",
    "road_islands": "polygon",
}

MINIMAP_TO_SEMANTIC_LABEL: Final = {
    "lanelines": 5,
    "lanes": 5,
    "poles": 9,
    "road_boundaries": 5,
    "wait_lines": 10,
    "crosswalks": 5,
    "road_markings": 10,
}


def extract_vertices(minimap_data: dict | list, vertices_list: Optional[list] = None) -> list:
    if vertices_list is None:
        vertices_list = []

    if isinstance(minimap_data, dict):
        for key, value in minimap_data.items():
            if key == "vertices":
                vertices_list.append(value)
            else:
                extract_vertices(value, vertices_list)
    elif isinstance(minimap_data, list):
        for item in minimap_data:
            extract_vertices(item, vertices_list)
    else:
        raise ValueError(f"Invalid minimap data type: {type(minimap_data)}")

    return vertices_list


def get_type_from_name(minimap_name: str) -> str:
    """
    Args:
        minimap_name: str, name of the minimap

    Returns:
        minimap_type: str, type of the minimap
    """
    if minimap_name in MINIMAP_TO_TYPE:
        return MINIMAP_TO_TYPE[minimap_name]
    else:
        raise ValueError(f"Invalid minimap name: {minimap_name}")


def cuboid3d_to_polyline(cuboid3d_eight_vertices: np.ndarray) -> np.ndarray:
    """
    Convert cuboid3d to polyline

    Args:
        cuboid3d_eight_vertices: np.ndarray, shape (8, 3), dtype=np.float32,
            eight vertices of the cuboid3d

    Returns:
        polyline: np.ndarray, shape (N, 3), dtype=np.float32,
            polyline vertices
    """
    if isinstance(cuboid3d_eight_vertices, list):
        cuboid3d_eight_vertices = np.array(cuboid3d_eight_vertices)

    connected_vertices_indices = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    connected_polyline = np.array(cuboid3d_eight_vertices)[connected_vertices_indices]

    return connected_polyline
