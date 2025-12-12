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

from typing import Any, Optional

import numpy as np
from scipy.spatial import Delaunay


def interpolate_polyline_to_points(polyline: np.ndarray, segment_interval: float = 0.025) -> np.ndarray:
    """
    polyline:
        numpy.ndarray, shape (N, 3) or list of points

    Returns:
        points: numpy array, shape (interpolate_num*N, 3)
    """

    def interpolate_points(previous_vertex: np.ndarray, vertex: np.ndarray) -> np.ndarray:
        """
        Args:
            previous_vertex: (x, y, z)
            vertex: (x, y, z)

        Returns:
            points: numpy array, shape (interpolate_num, 3)
        """
        interpolate_num = int(np.linalg.norm(np.array(vertex) - np.array(previous_vertex)) / segment_interval)
        interpolate_num = max(interpolate_num, 2)

        # interpolate between previous_vertex and vertex
        x = np.linspace(previous_vertex[0], vertex[0], num=interpolate_num)
        y = np.linspace(previous_vertex[1], vertex[1], num=interpolate_num)
        z = np.linspace(previous_vertex[2], vertex[2], num=interpolate_num)

        # remove the last point, we will include it in the next interpolation
        return np.stack([x, y, z], axis=1)[:-1]

    points = []
    previous_vertex = polyline[0]
    for vertex in polyline[1:]:
        points.extend(interpolate_points(previous_vertex, vertex))
        previous_vertex = vertex

    # add the last point
    points.append(polyline[-1])

    return np.array(points)


def triangulate_polygon_3d(vertices3d: np.ndarray) -> np.ndarray:
    """
    Triangulate a 3D polygon using Delaunay triangulation on the x-y plane.

    Args:
        vertices3d: np.ndarray, shape (N, 3), polygon vertices in 3D

    Returns:
        np.ndarray, shape (M, 3, 3), triangulated faces in 3D
    """
    vertices3d = np.asarray(vertices3d, dtype=float)

    # Remove duplicate last vertex if polygon is closed
    if vertices3d.shape[0] >= 2 and np.allclose(vertices3d[0], vertices3d[-1]):
        vertices3d = vertices3d[:-1]

    if vertices3d.shape[0] < 3:
        return np.empty((0, 3, 3), dtype=float)

    # Perform Delaunay triangulation on x-y plane
    tri = Delaunay(vertices3d[:, :2])

    # Get the triangulated faces
    triangles = vertices3d[tri.simplices]

    return triangles


def filter_by_height_relative_to_ego(
    geometry: np.ndarray,
    camera_model: Any,  # Camera model instance (FThetaCamera or similar)
    camera_pose: np.ndarray,
    camera_pose_init: Optional[np.ndarray] = None,
    underpass_threshold: float = -3.0,
    overpass_threshold: float = 5.0,
    envelope_angle_deg: float = 3.0,
    min_overlap_percentage: float = 0.8,
) -> bool:
    """
    Filter out map elements that are on underpasses/overpasses based on height relative to ego.
    Also filters based on overlap with a road surface envelope for horizontal elements.

    Args:
        geometry: np.ndarray
            Shape (N, 3). 3D points of the map element in world coordinates.
        camera_model: CameraModel
            Camera model (used for coordinate transformations).
        camera_pose: np.ndarray
            Shape (4, 4). Camera-to-world transformation matrix.
        camera_pose_init: Optional[np.ndarray]
            Shape (4, 4). Initial camera-to-world transformation matrix at the start of the clip.
            If None, uses the current camera_pose as reference.
        underpass_threshold: float
            If maximum height is below this threshold, element is considered underpass.
        overpass_threshold: float
            If minimum height is above this threshold, element is considered overpass.
        envelope_angle_deg: float
            Road surface envelope angle in degrees (±angle from horizontal).
        min_overlap_percentage: float
            Minimum percentage of points that must be within road surface envelope.

    Returns:
        bool: True if the element should be filtered out (is overpass/underpass/outside envelope), False otherwise
    """
    if len(geometry) == 0:
        return True  # Filter out empty geometry

    # Use current camera pose as reference if init pose not provided
    if camera_pose_init is None:
        camera_pose_init = camera_pose

    # Transform points from world coordinates to camera coordinates
    world_to_camera = np.linalg.inv(camera_pose)
    points_in_cam = camera_model.transform_points(geometry, world_to_camera)

    # Transform to ego space (initial camera frame)
    points_in_ego_space = camera_model.transform_points(points_in_cam, camera_pose_init)

    # In ego space: x (front), y (left), z (up)
    heights_in_ego_space = points_in_ego_space[:, 2]

    # Check for underpass: if maximum height is below threshold
    max_height = np.max(heights_in_ego_space)
    if max_height < underpass_threshold:
        return True  # Filter out - it's an underpass

    # Check for overpass: if minimum height is above threshold
    min_height = np.min(heights_in_ego_space)
    if min_height > overpass_threshold:
        return True  # Filter out - it's an overpass

    # Additional check: road surface envelope for horizontal elements
    # This helps filter elements that are at roughly the right height but too far from the road surface
    if envelope_angle_deg > 0 and min_overlap_percentage > 0:
        # Get distances (x, y) and heights (z) in ego space
        distances_xy = points_in_ego_space[:, :2]
        heights = points_in_ego_space[:, 2]
        distance_norm = np.linalg.norm(distances_xy, axis=1)

        # Calculate the road surface envelope bounds for each point
        # The envelope expands with distance based on the angle
        envelope_angle_rad = np.radians(envelope_angle_deg)

        # Expected road height at each distance (assuming flat road at ego height)
        expected_height = 0.0

        # Envelope bounds: height can vary by ±(distance * tan(angle))
        height_tolerance = distance_norm * np.tan(envelope_angle_rad) + 1.0  # Add 1m base tolerance

        # Check which points are within the envelope
        within_envelope = np.abs(heights - expected_height) <= height_tolerance

        # Calculate percentage of points within envelope
        overlap_percentage = np.sum(within_envelope) / len(within_envelope)

        # If too few points are within the road surface envelope, filter out
        if overlap_percentage < min_overlap_percentage:
            return True  # Filter out - not enough points on road surface

    return False  # Don't filter - element is at appropriate height
