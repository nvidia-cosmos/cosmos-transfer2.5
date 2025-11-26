# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Lane line pattern rendering utilities for V3 rendering."""

import json
from pathlib import Path

import numpy as np

from av_utils.pcd_utils import interpolate_polyline_to_points
from av_utils.graphics_utils import LineSegment2D
from av_utils.pcd_utils import filter_by_height_relative_to_ego

def load_laneline_geometry_config() -> dict:
    """Load lane line geometry configuration."""
    config_path = Path(__file__).parent / "color_configs" / "config_color_geometry_laneline.json"

    if not config_path.exists():
        # Fallback to basic config
        return {"OTHER": {"color": [128, 128, 128], "pattern": "solid", "thickness_multiplier": 1.0}}

    with open(config_path) as f:
        return json.load(f)


def create_long_dashed_segments(line_segments: np.ndarray, segment_interval: float = 0.05) -> np.ndarray:
    """
    Create long dashed line pattern following US highway standards:
    10 feet dash, 30 feet gap (1:3 ratio, 25% visible).

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dashing
        segment_interval: float, distance between adjacent segments in meters

    Returns:
        dashed_segments: np.ndarray, shape [M, 2, 3], dashed line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # US highway standard: 10ft (3m) dash, 30ft (9m) gap
    dash_length = 3.0  # meters
    gap_length = 9.0  # meters
    pattern_length = dash_length + gap_length

    # Calculate cumulative distances
    segment_lengths = np.linalg.norm(line_segments[:, 1] - line_segments[:, 0], axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_distances[-1]

    dashed_segments = []
    pattern_start = 0

    while pattern_start < total_length:
        dash_end = min(pattern_start + dash_length, total_length)

        # Find segments that overlap with this dash
        start_idx = np.searchsorted(cumulative_distances[1:], pattern_start, side="right")
        end_idx = np.searchsorted(cumulative_distances[1:], dash_end, side="left")

        if start_idx <= end_idx and end_idx < len(line_segments):
            # Add segments that are fully within the dash
            for i in range(start_idx, end_idx + 1):
                if cumulative_distances[i] >= pattern_start and cumulative_distances[i + 1] <= dash_end:
                    dashed_segments.append(line_segments[i])
                elif cumulative_distances[i] < pattern_start and cumulative_distances[i + 1] > pattern_start:
                    # Partial segment at start
                    t = (pattern_start - cumulative_distances[i]) / segment_lengths[i]
                    start_point = line_segments[i, 0] + t * (line_segments[i, 1] - line_segments[i, 0])
                    if cumulative_distances[i + 1] <= dash_end:
                        dashed_segments.append(np.array([start_point, line_segments[i, 1]]))
                    else:
                        t_end = (dash_end - cumulative_distances[i]) / segment_lengths[i]
                        end_point = line_segments[i, 0] + t_end * (line_segments[i, 1] - line_segments[i, 0])
                        dashed_segments.append(np.array([start_point, end_point]))
                elif cumulative_distances[i] < dash_end and cumulative_distances[i + 1] > dash_end:
                    # Partial segment at end
                    t = (dash_end - cumulative_distances[i]) / segment_lengths[i]
                    end_point = line_segments[i, 0] + t * (line_segments[i, 1] - line_segments[i, 0])
                    dashed_segments.append(np.array([line_segments[i, 0], end_point]))

        pattern_start += pattern_length

    return np.array(dashed_segments) if dashed_segments else np.empty((0, 2, 3))


def create_short_dashed_segments(line_segments: np.ndarray, segment_interval: float = 0.05) -> np.ndarray:
    """
    Create short dashed line pattern for city streets.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3]
        segment_interval: float

    Returns:
        dashed_segments: np.ndarray, shape [M, 2, 3]
    """
    if len(line_segments) == 0:
        return line_segments

    # Shorter pattern for city streets
    dash_length = 1.5  # meters
    gap_length = 1.5  # meters
    pattern_length = dash_length + gap_length

    # Calculate cumulative distances
    segment_lengths = np.linalg.norm(line_segments[:, 1] - line_segments[:, 0], axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_distances[-1]

    dashed_segments = []
    pattern_start = 0

    while pattern_start < total_length:
        dash_end = min(pattern_start + dash_length, total_length)

        # Find segments that overlap with this dash
        start_idx = np.searchsorted(cumulative_distances[1:], pattern_start, side="right")
        end_idx = np.searchsorted(cumulative_distances[1:], dash_end, side="left")

        if start_idx <= end_idx and end_idx < len(line_segments):
            for i in range(start_idx, min(end_idx + 1, len(line_segments))):
                if cumulative_distances[i] >= pattern_start and cumulative_distances[i + 1] <= dash_end:
                    dashed_segments.append(line_segments[i])

        pattern_start += pattern_length

    return np.array(dashed_segments) if dashed_segments else np.empty((0, 2, 3))


def create_dotted_segments(line_segments: np.ndarray, segment_interval: float = 0.05) -> np.ndarray:
    """
    Create dotted line pattern.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3]
        segment_interval: float

    Returns:
        dotted_segments: np.ndarray, shape [M, 2, 3]
    """
    if len(line_segments) == 0:
        return line_segments

    # Very short dashes for dots
    dot_length = 0.3  # meters
    gap_length = 0.6  # meters
    pattern_length = dot_length + gap_length

    # Calculate cumulative distances
    segment_lengths = np.linalg.norm(line_segments[:, 1] - line_segments[:, 0], axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_distances[-1]

    dotted_segments = []
    pattern_start = 0

    while pattern_start < total_length:
        dot_end = min(pattern_start + dot_length, total_length)

        # Find segments that overlap with this dot
        start_idx = np.searchsorted(cumulative_distances[1:], pattern_start, side="right")
        end_idx = np.searchsorted(cumulative_distances[1:], dot_end, side="left")

        if start_idx <= end_idx and end_idx < len(line_segments):
            for i in range(start_idx, min(end_idx + 1, len(line_segments))):
                if cumulative_distances[i] >= pattern_start and cumulative_distances[i + 1] <= dot_end:
                    dotted_segments.append(line_segments[i])

        pattern_start += pattern_length

    return np.array(dotted_segments) if dotted_segments else np.empty((0, 2, 3))


def create_dual_line_segments(line_segments: np.ndarray, separation: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Create dual (double) line pattern by offsetting perpendicular to the line direction.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3]
        separation: float, distance between the two lines in meters

    Returns:
        left_segments: np.ndarray, shape [N, 2, 3]
        right_segments: np.ndarray, shape [N, 2, 3]
    """
    if len(line_segments) == 0:
        return line_segments, line_segments

    left_segments = []
    right_segments = []

    for segment in line_segments:
        # Calculate perpendicular direction
        direction = segment[1] - segment[0]
        direction_2d = direction[:2]  # Use only x, y for perpendicular
        if np.linalg.norm(direction_2d) < 1e-6:
            continue

        direction_2d = direction_2d / np.linalg.norm(direction_2d)
        perpendicular = np.array([-direction_2d[1], direction_2d[0], 0])

        # Create offset segments
        offset = perpendicular * separation
        left_segment = segment + offset
        right_segment = segment - offset

        left_segments.append(left_segment)
        right_segments.append(right_segment)

    return np.array(left_segments), np.array(right_segments)


def create_dot_dashed_segments(line_segments: np.ndarray, segment_interval: float = 0.05) -> np.ndarray:
    """
    Create dot-dashed pattern following US highway standards:
    3 feet dash period, 9 feet gap, with dots within dash period.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dot-dashing
        segment_interval: float, distance between adjacent segments in meters

    Returns:
        dot_dashed_segments: np.ndarray, shape [M, 2, 3], dot-dashed line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # US highway standard: 3ft dash period, 9ft gap
    dash_length_meters = 3 * 0.3048  # 3 feet = ~0.91 meters
    gap_length_meters = 9 * 0.3048  # 9 feet = ~2.74 meters

    # Convert to segment counts
    dash_segments_count = max(1, int(dash_length_meters / segment_interval))
    gap_segments_count = max(1, int(gap_length_meters / segment_interval))

    if len(line_segments) <= dash_segments_count:
        return line_segments[::3]  # Very sparse for short lines

    dot_segments = []
    total_segments = len(line_segments)

    i = 0
    while i < total_segments:
        # Within the dash period, create dots with 1:2 spacing (every 3rd segment)
        dash_end = min(i + dash_segments_count, total_segments)
        for j in range(i, dash_end, 3):  # Every 3rd segment within dash period
            if j < total_segments:
                dot_segments.append(line_segments[j])

        # Skip the gap period
        i = dash_end + gap_segments_count

    return np.array(dot_segments) if dot_segments else line_segments[:1]


def create_dotted_segments_1_9_ratio(line_segments: np.ndarray) -> np.ndarray:
    """
    Create dotted line with 1:9 dot to gap ratio (10% visible).

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply dotting

    Returns:
        dotted_segments: np.ndarray, shape [M, 2, 3], dotted line segments
    """
    if len(line_segments) == 0:
        return line_segments

    # Simple 1:9 ratio - take every 10th segment
    return line_segments[::10]

def create_laneline_geometry_objects_from_data(
    processed_lanelines,
    camera_pose,
    camera_model,
    camera_pose_init=None,
):
    """
    Build LineSegment2D geometry objects for a single frame from preprocessed laneline data.

    Args:
        processed_lanelines: list[dict], each with keys 'pattern_segments_list', 'rgb_float', 'line_width'
        camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, ftheta or pinhole camera model
        camera_pose_init: np.ndarray, shape (4, 4), dtype=np.float32, camera pose at the start of the clip

    Returns:
        geometry_objects: list[LineSegment2D], geometry objects for the current frame
    """
    geometry_objects = []
    for laneline_info in processed_lanelines:
        # Combine all segments into a single polyline
        combined_segments = np.concatenate(laneline_info['pattern_segments_list'], axis=0)
        # Reshape to [N, 3]
        combined_segments = combined_segments.reshape(-1, 3)
        if camera_pose_init is not None and filter_by_height_relative_to_ego(
            combined_segments, camera_model, camera_pose, camera_pose_init):
            continue
        
        for segments in laneline_info['pattern_segments_list']:
            if len(segments) == 0:
                continue
            # Project to image space and filter out the line segments that are out of the image
            xy_and_depth = camera_model.get_xy_and_depth(segments.reshape(-1, 3), camera_pose).reshape(-1, 2, 3)
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]
            if len(valid_xy_and_depth) == 0:
                continue
            geometry_objects.append(
                LineSegment2D(
                    valid_xy_and_depth,
                    base_color=laneline_info['rgb_float'],
                    line_width=laneline_info['line_width']
                )
            )

    return geometry_objects

def offset_line_segments(line_segments: np.ndarray, offset_distance: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Create left and right offset versions of line segments for dual-line patterns.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to offset
        offset_distance: float, distance to offset (in meters)

    Returns:
        tuple: (left_segments, right_segments), both np.ndarray shape [N, 2, 3]
    """
    if len(line_segments) == 0:
        return line_segments, line_segments

    left_segments = []
    right_segments = []

    for segment in line_segments:
        p1, p2 = segment[0], segment[1]

        # Calculate direction vector
        direction = p2 - p1
        if np.linalg.norm(direction) == 0:
            left_segments.append(segment)
            right_segments.append(segment)
            continue

        direction = direction / np.linalg.norm(direction)

        # Calculate perpendicular vector (left is 90° counter-clockwise)
        perpendicular = np.array([-direction[1], direction[0], 0])

        # Create offset segments
        left_offset = perpendicular * offset_distance
        right_offset = -perpendicular * offset_distance

        left_segment = np.array([p1 + left_offset, p2 + left_offset])
        right_segment = np.array([p1 + right_offset, p2 + right_offset])

        left_segments.append(left_segment)
        right_segments.append(right_segment)

    return np.array(left_segments), np.array(right_segments)


def apply_laneline_pattern(line_segments: np.ndarray, specs: dict) -> list[np.ndarray]:
    """
    Apply the specified pattern to line segments.

    Args:
        line_segments: np.ndarray, shape [N, 2, 3], line segments to apply pattern to
        specs: dict, specifications from configuration containing pattern type

    Returns:
        list: list of np.ndarray segments arrays for rendering
    """
    pattern = specs.get("pattern", "solid")

    if pattern == "solid":
        return [line_segments]

    elif pattern == "long_dashed":
        dashed_segments = create_long_dashed_segments(line_segments)
        return [dashed_segments]

    elif pattern == "short_dashed":
        dashed_segments = create_short_dashed_segments(line_segments)
        return [dashed_segments]

    elif pattern == "dot_dashed":
        dot_dashed_segments = create_dot_dashed_segments(line_segments)
        return [dot_dashed_segments]

    elif pattern == "dotted_1_9":
        dotted_segments = create_dotted_segments_1_9_ratio(line_segments)
        return [dotted_segments]

    elif pattern == "dual":
        dual_pattern = specs.get("dual_pattern")
        if not dual_pattern:
            return [line_segments]

        left_segments, right_segments = offset_line_segments(line_segments)
        left_pattern, right_pattern = dual_pattern

        result = []

        # Apply left pattern using recursive call
        left_specs = {"pattern": left_pattern}
        left_results = apply_laneline_pattern(left_segments, left_specs)
        result.extend(left_results)

        # Apply right pattern using recursive call
        right_specs = {"pattern": right_pattern}
        right_results = apply_laneline_pattern(right_segments, right_specs)
        result.extend(right_results)

        return result

    else:
        return [line_segments]  # Fallback


def prepare_laneline_geometry_data(lanelines_with_types: list[tuple[np.ndarray, str]]) -> list[dict]:
    """
    Preprocess laneline data into segments with patterns for V3 rendering.

    Args:
        lanelines_with_types: List of tuples (polyline, type_string)

    Returns:
        processed_lanelines: List of dicts with pattern segments and rendering info
    """
    config = load_laneline_geometry_config()
    processed_lanelines = []

    for polyline, lane_type in lanelines_with_types:
        # Get specs for this lane type
        specs = config.get(
            lane_type, config.get("OTHER", {"color": [128, 128, 128], "pattern": "solid", "thickness_multiplier": 1.0})
        )

        # Subdivide polyline for smooth curves
        polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.05)

        # Create line segments
        if len(polyline_subdivided) < 2:
            continue

        line_segments = np.stack([polyline_subdivided[:-1], polyline_subdivided[1:]], axis=1)

        # Apply pattern
        pattern_segments_list = apply_laneline_pattern(line_segments, specs)

        # Get color
        rgb_float = np.array(specs.get("color", [128, 128, 128])) / 255.0

        # Get line width
        base_width = 12
        line_width = base_width * specs.get("thickness_multiplier", 1.0)

        processed_lanelines.append(
            {
                "pattern_segments_list": pattern_segments_list,
                "rgb_float": rgb_float,
                "line_width": line_width,
                "lane_type": lane_type,
                "original_polyline": polyline,  # Keep original for height filtering
            }
        )

    return processed_lanelines
