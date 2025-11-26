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
from typing import Final

import numpy as np

from av_utils.camera.base import CameraBase
from av_utils.graphics_utils import LineSegment2D, Polygon2D, TriangleList2D
from av_utils.pcd_utils import filter_by_height_relative_to_ego, interpolate_polyline_to_points, triangulate_polygon_3d


def load_hdmap_colors(version: str = "v3") -> dict:
    """
    Load hdmap colors based on the specified version.

    Args:
        version: str, version key from config_color_hdmap.json
                Available versions: 'v3'

    Returns:
        dict: Color configuration for the specified version
    """
    # Try to find the config file in the local color_configs directory
    config_path = Path(__file__).parent / "color_configs" / "config_color_hdmap.json"
    with open(config_path) as f:
        hdmap_config = json.load(f)

    if version not in hdmap_config:
        available_versions = list(hdmap_config.keys())
        raise ValueError(f"Version '{version}' not found in hdmap config. Available versions: {available_versions}")

    return hdmap_config[version]


def load_laneline_colors() -> dict:
    """
    Load laneline type-specific colors.

    Returns:
        dict: Color configuration for laneline types
    """
    config_path = Path(__file__).parent / "color_configs" / "config_color_laneline.json"
    with open(config_path) as f:
        laneline_config = json.load(f)
    return laneline_config


def load_traffic_light_colors(version: str = "v2") -> dict:
    """
    Load traffic light colors based on the specified version.

    Args:
        version: str, version key from config_color_traffic_light.json
                Available versions: 'v2'

    Returns:
        dict: Color configuration for the specified version
    """
    config_path = Path(__file__).parent / "color_configs" / "config_color_traffic_light.json"
    with open(config_path) as f:
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


def extract_vertices(minimap_data: dict | list, vertices_list: list | None = None) -> list:
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


def create_minimap_projection(
    minimap_name: str,
    minimap_data_wo_meta_info: list,
    camera_poses: np.ndarray,
    camera_model: CameraBase,
    hdmap_color_version: str = "v3",
) -> np.ndarray:
    """
    Args:
        minimap_name: str, name of the minimap
        minimap_data_wo_meta_info: list of np.ndarray
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model
        hdmap_color_version: str, version key for hdmap colors ('v3')

    Returns:
        minimaps_projection: np.ndarray,
            shape (N, H, W, 3), dtype=np.uint8, projected minimap data across N frames
    """

    # Load colors based on the specified version
    minimap_to_rgb = load_hdmap_colors(hdmap_color_version)

    minimap_type = get_type_from_name(minimap_name)

    if minimap_type == "polygon":
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
        )
    elif minimap_type == "polyline":
        if minimap_name == "lanelines" or minimap_name == "road_boundaries":
            segment_interval = 0.8
        else:
            segment_interval = 0

        projection_images = camera_model.draw_line_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
            segment_interval=segment_interval,
        )
    elif minimap_type == "cuboid3d":
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(minimap_to_rgb[minimap_name]),
        )
    else:
        raise ValueError(f"Invalid minimap type: {minimap_type}")

    return projection_images

def create_minimap_geometry_objects_from_data(
    minimap_name_to_minimap_data,
    camera_pose,
    camera_model,
    minimap_to_rgb,
    camera_pose_init=None,
):
    """
    Build geometry objects for minimap layers for a single frame.

    Args:
        minimap_name_to_minimap_data: dict[name -> list[np.ndarray]]
        camera_pose: np.ndarray (4,4)
        camera_model: CameraModel
        minimap_to_rgb: dict[str, list[np.ndarray]], mapping from minimap name to RGB values
        camera_pose_init: np.ndarray (4,4), reference camera pose for ego space transformation

    Returns:
        list: geometry objects (LineSegment2D/Polygon2D)
    """

    all_geometry_objects = []
    for minimap_name, minimap_data in minimap_name_to_minimap_data.items():
        minimap_type = get_type_from_name(minimap_name)
        
        # We create LineSegment2D geometry object for each polyline
        if minimap_type == 'polyline':
            line_segment_list = []
            for polyline in minimap_data:
                # Filter out minimap that is under the ego vehicle
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    polyline, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                # Subdivide the polyline so that distortion is observed in camera view
                polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.2)
                line_segment = np.stack([polyline_subdivided[0:-1], polyline_subdivided[1:]], axis=1) # [N, 2, 3]
                line_segment_list.append(line_segment)
            if len(line_segment_list) == 0:
                continue

            all_line_segments = np.concatenate(line_segment_list, axis=0) # [N', 2, 3]
            xy_and_depth = camera_model.get_xy_and_depth(all_line_segments.reshape(-1, 3), camera_pose).reshape(-1, 2, 3) # [N', 3]
            
            # filter the line segments with both vertices have depth >= 0
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]
            if len(valid_xy_and_depth) == 0:
                continue

            color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
            all_geometry_objects.append(
                LineSegment2D(
                    valid_xy_and_depth,
                    base_color=color_float,
                    line_width=5 if minimap_name == 'poles' else 12,
                )
            )

        elif minimap_type == 'polygon' or minimap_type == 'cuboid3d':
            # merge all vertices from polygons and record indices
            polygon_vertices = []
            polygon_vertex_counts = []
            for polygon in minimap_data:
                # Filter out minimap that is under the ego vehicle
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    polygon, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                if minimap_name == 'crosswalks':
                    # Subdivide the polygon so that distortion is observed in camera view
                    polygon_subdivided = interpolate_polyline_to_points(polygon, segment_interval=0.8)
                    # Use triangulation for crosswalks to handle concave polygons in camera view
                    triangles_3d = triangulate_polygon_3d(polygon_subdivided)
                    polygon_subdivided = triangles_3d.reshape(-1, 3)
                    if len(polygon_subdivided) == 0:
                        continue
                else:
                    polygon_subdivided = polygon
                polygon_vertices.append(polygon_subdivided)
                polygon_vertex_counts.append(len(polygon_subdivided))
            if len(polygon_vertices) == 0:
                continue
            # get xy and depth for all vertices at once
            all_vertices = np.concatenate(polygon_vertices, axis=0)
            all_xy_and_depth = camera_model.get_xy_and_depth(all_vertices, camera_pose)

            # recover individual polygons using recorded counts, and keep access to original 3D subdivided vertices
            start_idx = 0
            for vertex_count in polygon_vertex_counts:
                polygon_xy_and_depth = all_xy_and_depth[start_idx:start_idx+vertex_count]
                start_idx += vertex_count
                color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
                
                if minimap_name == 'crosswalks':
                    triangles_proj = polygon_xy_and_depth.reshape(-1, 3, 3)
                    # Filter out triangles that have ANY vertex behind camera (z < 0)
                    # This prevents rendering artifacts from triangles crossing the camera plane
                    invalid_triangles_indices = np.any(triangles_proj[:, :, 2] < 0, axis=1)
                    valid_triangles_indices = ~invalid_triangles_indices
                    if valid_triangles_indices.sum() > 0:
                        all_geometry_objects.append(
                            TriangleList2D(triangles_proj[valid_triangles_indices], base_color=color_float)
                        )
                else:
                    # Filter out polygons that have ANY vertex behind camera (z < 0)
                    # This prevents rendering artifacts from polygons crossing the camera plane
                    if np.any(polygon_xy_and_depth[:, 2] < 0):
                        continue
                    # Use regular polygon rendering for other polygon types
                    all_geometry_objects.append(
                        Polygon2D(polygon_xy_and_depth, base_color=color_float)
                    )
        else:
            raise ValueError(f"Invalid minimap type: {minimap_type}")

    return all_geometry_objects

def create_laneline_type_projection(
    laneline_data_with_types: list,
    camera_poses: np.ndarray,
    camera_model: CameraBase,
) -> np.ndarray:
    """
    Create laneline projection with type-specific colors.

    Args:
        laneline_data_with_types: list of tuples (polyline, type_string)
            where type_string is like "WHITE SOLID_SINGLE", "YELLOW DASHED_SINGLE", etc.
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model

    Returns:
        minimaps_projection: np.ndarray,
            shape (N, H, W, 3), dtype=np.uint8, projected laneline data across N frames
    """
    laneline_type_to_rgb = load_laneline_colors()

    # Group lanelines by type for efficient rendering
    type_to_polylines = {}
    for polyline, type_string in laneline_data_with_types:
        if type_string not in type_to_polylines:
            type_to_polylines[type_string] = []
        type_to_polylines[type_string].append(polyline)

    # Initialize output array
    num_frames = len(camera_poses) if camera_poses.ndim == 3 else 1
    projection_images = np.zeros((num_frames, camera_model.height, camera_model.width, 3), dtype=np.uint8)

    # Render each type with its specific color
    for type_string, polylines in type_to_polylines.items():
        # Get color for this type, default to "OTHER" if not found
        color = laneline_type_to_rgb.get(type_string, laneline_type_to_rgb.get("OTHER", [100, 100, 100]))

        # Render this type's lanelines
        type_projection = camera_model.draw_line_depth(
            camera_poses,
            polylines,
            colors=np.array(color),
            segment_interval=0.8,  # Smooth the polyline
        )

        # Merge with existing projection
        projection_images = np.maximum(projection_images, type_projection)

    return projection_images
