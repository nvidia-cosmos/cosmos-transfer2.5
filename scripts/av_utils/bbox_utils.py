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
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from av_utils.camera.ftheta import FThetaCamera
from av_utils.minimap_utils import cuboid3d_to_polyline
from av_utils.graphics_utils import BoundingBox2D

def load_bbox_colors(version: str = "Uniform Color") -> dict:
    """
    Load bbox colors based on the specified version.

    Args:
        version: str, version key from config_color_bbox.json
                Available versions: 'Uniform Color', 'Gradient Color', 'v3'

    Returns:
        dict: Color configuration for the specified version
    """
    # Try to find the config file in the local color_configs directory
    config_path = Path(__file__).parent / "color_configs" / "config_color_bbox.json"
    with open(config_path) as f:
        bbox_config = json.load(f)

    if version not in bbox_config:
        available_versions = list(bbox_config.keys())
        raise ValueError(f"Version '{version}' not found in bbox config. Available versions: {available_versions}")

    return bbox_config[version]

def build_per_vertex_color_map(color_version='Gradient Color'):
    """
    Build per-vertex color arrays for each object class according to the
    bbox color configuration version.

    Returns:
        dict[str, np.ndarray(8,3)]: mapping from class name to per-vertex RGB
    """
    gradient_class_colors = load_bbox_colors(color_version)
    object_type_to_per_vertex_color = {}
    for object_type, colors in gradient_class_colors.items():
        per_vertex_color = np.zeros((8, 3))
        per_vertex_color[[0,1,4,5]] = np.array(colors[0]) / 255.0
        per_vertex_color[[2,3,6,7]] = np.array(colors[1]) / 255.0
        object_type_to_per_vertex_color[object_type] = per_vertex_color
    return object_type_to_per_vertex_color

# Default colors for backward compatibility
CLASS_COLORS = load_bbox_colors("Uniform Color")
GRADIENT_CLASS_COLORS = load_bbox_colors("Gradient Color")


def simplify_type_in_object_info(object_info: dict) -> dict:
    """Simplify object type to standard categories."""
    if object_info["object_type"] not in CLASS_COLORS:
        # labels from category v1
        if object_info["object_type"] == "Bus":
            object_info["object_type"] = "Truck"
        # labels from category v1
        elif object_info["object_type"] == "Vehicle":
            object_info["object_type"] = "Car"
        # labels from category v2
        elif (
            object_info["object_type"] == "Heavy_truck"
            or object_info["object_type"] == "Train_or_tram_car"
            or object_info["object_type"] == "Trolley_bus"
            or object_info["object_type"] == "Trailer"
        ):
            object_info["object_type"] = "Truck"
        # labels from category v2
        elif object_info["object_type"] == "Automobile" or object_info["object_type"] == "Other_vehicle":
            object_info["object_type"] = "Car"
        # labels from category v2
        elif object_info["object_type"] == "Person":
            object_info["object_type"] = "Pedestrian"
        # labels from category v2
        elif object_info["object_type"] == "Rider":
            object_info["object_type"] = "Cyclist"
        else:
            object_info["object_type"] = "Others"
    return object_info


def create_bbox_projection(
    all_object_info: dict[str, Any],
    camera_poses: np.ndarray,
    valid_frame_ids: list[int],
    camera_model: FThetaCamera,
    color_version: str = "Uniform Color",
) -> np.ndarray:
    """
    Create a projection of bounding boxes on the minimap.
    Args:
        all_object_info: dict, containing all object info
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera to world transformation matrix
        camera_model: CameraModel, camera model
        valid_frame_ids: list[int], valid frame ids
        color_version: str, version key for bbox colors ('Uniform Color', 'Gradient Color', or 'v3')

    Returns:
        np.ndarray, shape (N, H, W, 3), dtype=np.uint8, projected bounding boxes on canvas
    """
    # Load colors based on the specified version
    class_colors = load_bbox_colors(color_version)

    bbox_projections = []

    for i in valid_frame_ids:
        current_object_info = all_object_info[f"{i:06d}.all_object_info.json"]

        object_type_to_polylines = {
            "Car": [],
            "Truck": [],
            "Pedestrian": [],
            "Cyclist": [],
            "Others": [],
        }

        # sort tracking ids. avoid jittering when drawing bbox.
        tracking_ids = list(current_object_info.keys())
        tracking_ids.sort()

        for tracking_id in tracking_ids:
            object_info = current_object_info[tracking_id]
            object_info = simplify_type_in_object_info(object_info)

            object_to_world = np.array(object_info["object_to_world"])
            object_lwh = np.array(object_info["object_lwh"])
            cuboid_eight_vertices = build_cuboid_bounding_box(
                object_lwh[0], object_lwh[1], object_lwh[2], object_to_world
            )
            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            # draw by the object type
            if object_info["object_type"] in ["Car", "Truck", "Pedestrian", "Cyclist"]:
                object_type_to_polylines[object_info["object_type"]].append(polyline)
            else:
                object_type_to_polylines["Others"].append(polyline)

        # Handle both uniform and gradient colors
        def get_color_array(color_value: Any) -> np.ndarray:
            if isinstance(color_value, list) and len(color_value) == 2 and isinstance(color_value[0], list):
                # Gradient color - use the first color for simple rendering
                return np.array(color_value[0])
            else:
                # Uniform color
                return np.array(color_value)

        cars_bbox_projection = camera_model.draw_line_depth(
            camera_poses[i], object_type_to_polylines["Car"], radius=5, colors=get_color_array(class_colors["Car"])
        )
        trucks_bbox_projection = camera_model.draw_line_depth(
            camera_poses[i], object_type_to_polylines["Truck"], radius=5, colors=get_color_array(class_colors["Truck"])
        )
        pedestrians_bbox_projection = camera_model.draw_line_depth(
            camera_poses[i],
            object_type_to_polylines["Pedestrian"],
            radius=5,
            colors=get_color_array(class_colors["Pedestrian"]),
        )
        cyclists_bbox_projection = camera_model.draw_line_depth(
            camera_poses[i],
            object_type_to_polylines["Cyclist"],
            radius=5,
            colors=get_color_array(class_colors["Cyclist"]),
        )
        others_bbox_projection = camera_model.draw_line_depth(
            camera_poses[i],
            object_type_to_polylines["Others"],
            radius=5,
            colors=get_color_array(class_colors["Others"]),
        )

        # combine the dynamic and static bbox projection
        bbox_projection = np.maximum.reduce(
            [
                cars_bbox_projection,
                trucks_bbox_projection,
                pedestrians_bbox_projection,
                cyclists_bbox_projection,
                others_bbox_projection,
            ]
        )
        bbox_projections.append(bbox_projection)

    return np.concatenate(bbox_projections, axis=0)

def create_bbox_geometry_objects_for_frame(
    current_object_info,
    current_camera_pose,
    camera_model,
    fill_face='front_and_back',
    fill_face_style='diagonal',
    object_type_to_per_vertex_color=None,
    color_version='Gradient Color',
    line_width=9,
    edge_color=None,
):
    """
    Build BoundingBox2D geometry objects for a single frame.
    Args:
        current_object_info: dict, containing all object info for the current frame
        current_camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, camera model
        fill_face: str, which faces to fill
        fill_face_style: str, style of face filling
        object_type_to_per_vertex_color: dict, mapping from object type to per-vertex color
        color_version: str, version key for bbox colors from config_color_bbox.json
        line_width: int, line width for rendering
        edge_color: np.ndarray, shape (3,), dtype=np.float32, optional edge color

    Returns:
        list[BoundingBox2D], geometry objects for the current frame
    """
    # Prepare per-vertex colors if not supplied
    if object_type_to_per_vertex_color is None:
        object_type_to_per_vertex_color = build_per_vertex_color_map(color_version)
    if edge_color is not None:
        edge_color = np.array(edge_color) / 255.0

    # Store the 8 corner vertices of each object type
    object_type_to_corner_vertices = {
        'Car': [], 'Truck': [], 'Pedestrian': [], 'Cyclist': [], 'Others': []
    }

    tracking_ids = list(current_object_info.keys())
    tracking_ids.sort()

    for tracking_id in tracking_ids:
        object_info = current_object_info[tracking_id]
        object_info = simplify_type_in_object_info(object_info)

        object_to_world = np.array(object_info['object_to_world'])
        object_lwh = np.array(object_info['object_lwh'])
        cuboid_eight_vertices = build_cuboid_bounding_box(
            object_lwh[0], object_lwh[1], object_lwh[2], object_to_world
        ) # (8, 3)

        # Cull objects entirely behind camera
        if np.all(np.dot(cuboid_eight_vertices - current_camera_pose[:3, 3], current_camera_pose[:3, 2]) < 0):
            continue

        if object_info['object_type'] in ['Car', 'Truck', 'Pedestrian', 'Cyclist']:
            object_type_to_corner_vertices[object_info['object_type']].append(cuboid_eight_vertices)
        else:
            object_type_to_corner_vertices['Others'].append(cuboid_eight_vertices)

    # draw the bbox projection. xy are pixel coordinate, depth is in meters.
    geometry_objects = []
    for object_type, all_corner_vertices in object_type_to_corner_vertices.items():
        if len(all_corner_vertices) == 0:
            continue

        # all_corner_vertices is [n, 8, 3]
        n_objects = len(all_corner_vertices)
        all_corner_vertices_flatten = np.array(all_corner_vertices).reshape(-1, 3)
        all_points_in_cam = camera_model.transform_points(
            all_corner_vertices_flatten, np.linalg.inv(current_camera_pose)
        )
        all_depth = all_points_in_cam[:, 2:3] # (n * 8, 1)
        all_xy = camera_model.ray2pixel(all_points_in_cam) # (n * 8, 2)
        all_xy_and_depth = np.hstack([all_xy, all_depth]).reshape(n_objects, 8, 3) # (n, 8, 3)

        # Filter out bounding boxes that cross the camera plane (any vertex with depth <= 0)
        # This prevents rendering artifacts from bboxes that are partially behind camera
        valid_depth_mask = (all_xy_and_depth[:, :, 2] > 0)  # (n, 8)
        valid_object_mask = np.all(valid_depth_mask, axis=1)  # (n,) - all 8 vertices must have positive depth

        all_xy_and_depth = all_xy_and_depth[valid_object_mask]

        for xy_and_depth in all_xy_and_depth:
            geometry_objects.append(
                BoundingBox2D(
                    xy_and_depth=xy_and_depth,
                    base_color_or_per_vertex_color=object_type_to_per_vertex_color[object_type],
                    fill_face=fill_face,
                    fill_face_style=fill_face_style,
                    line_width=line_width,
                    edge_color=edge_color,
                )
            )

    return geometry_objects

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
    cuboid_transform: np.ndarray | None = None,
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
