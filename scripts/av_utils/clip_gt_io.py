# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Functions for loading data from clipGT artefacts to be used in map rendering pipeline."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from av_utils.bbox_utils import interpolate_pose
from av_utils.camera.ftheta import FThetaCamera


def load_obstacle_data_interpolated(
    obstacle_file: Path | None,
    target_timestamps: np.ndarray,
    start_buffer_ms: int = 200,  # 200ms buffer at start
    end_buffer_ms: int = 200,  # 200ms buffer at end
    diff_buffer_ms: int = 50,  # 50ms for single observations
) -> dict[str, dict]:
    """
    Load obstacle data and interpolate tracks to 30Hz.
    Handles tracks appearing at different times properly.
    """
    if obstacle_file is None:
        return {}

    df = pd.read_parquet(obstacle_file)

    # First, group all observations by track ID
    tracks = defaultdict(list)

    for _, row in df.iterrows():
        obstacle = row["obstacle"]
        key = row["key"]

        if "trackline_id" in obstacle and "timestamp_micros" in key:
            track_id = obstacle["trackline_id"]
            timestamp = key["timestamp_micros"]

            tracks[track_id].append({"timestamp": timestamp, "obstacle": obstacle})

    # Sort observations for each track by timestamp
    for track_id in tracks:
        tracks[track_id].sort(key=lambda x: x["timestamp"])

    # Initialize all frames with empty dictionaries
    frame_obstacles = {}
    for i in range(len(target_timestamps)):
        frame_obstacles[f"{i:06d}.all_object_info.json"] = {}

    # Process each track
    for track_id, observations in tracks.items():
        if len(observations) < 2:
            # Handle single observation tracks
            if len(observations) == 1:
                obs = observations[0]
                timestamp = obs["timestamp"]

                # Find closest target timestamp
                time_diffs = np.abs(target_timestamps - timestamp)
                closest_idx = np.argmin(time_diffs)

                # Only add if within reasonable time window
                if time_diffs[closest_idx] < diff_buffer_ms * 1000:
                    frame_key = f"{closest_idx:06d}.all_object_info.json"
                    obstacle = obs["obstacle"]

                    if "center" in obstacle and "size" in obstacle:
                        object_to_world = np.eye(4)
                        object_to_world[:3, 3] = [
                            obstacle["center"]["x"],
                            obstacle["center"]["y"],
                            obstacle["center"]["z"],
                        ]

                        if "orientation" in obstacle:
                            ori = obstacle["orientation"]
                            rot = Rotation.from_quat([ori["x"], ori["y"], ori["z"], ori["w"]])
                            object_to_world[:3, :3] = rot.as_matrix()

                        category_map = {
                            "automobile": "Car",
                            "other_vehicle": "Car",
                            "vehicle": "Car",
                            "car": "Car",
                            "pedestrian": "Pedestrian",
                            "person": "Pedestrian",
                            "bicycle": "Cyclist",
                            "cyclist": "Cyclist",
                            "motorcycle": "Cyclist",
                            "rider": "Cyclist",
                            "bus": "Truck",
                            "truck": "Truck",
                            "heavy_truck": "Truck",
                            "train_or_tram_car": "Truck",
                            "trolley_bus": "Truck",
                            "trailer": "Truck",
                        }
                        object_type = category_map.get(obstacle.get("category", "default").lower(), "Others")

                        frame_obstacles[frame_key][str(track_id)] = {
                            "object_to_world": object_to_world.tolist(),
                            "object_lwh": [obstacle["size"]["x"], obstacle["size"]["y"], obstacle["size"]["z"]],
                            "object_type": object_type,
                            "object_is_moving": True,
                        }
            continue

        # For tracks with 2+ observations, interpolate
        track_timestamps = np.array([obs["timestamp"] for obs in observations])

        # Find which target timestamps this track covers
        track_start = track_timestamps[0]
        track_end = track_timestamps[-1]

        # Get target timestamps within this track's time range
        # Add buffers at both start and end to handle timestamp misalignment
        # and tracks that end slightly before the sequence ends
        start_buffer_us = start_buffer_ms * 1000  # Convert to microseconds
        end_buffer_us = end_buffer_ms * 1000
        valid_mask = (target_timestamps >= track_start - start_buffer_us) & (
            target_timestamps <= track_end + end_buffer_us
        )
        valid_target_timestamps = target_timestamps[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        # Extract data and build transformation matrices for each observation
        poses = []
        sizes = np.array(
            [
                [obs["obstacle"]["size"]["x"], obs["obstacle"]["size"]["y"], obs["obstacle"]["size"]["z"]]
                for obs in observations
            ]
        )

        for obs in observations:
            pose = np.eye(4)
            pose[:3, 3] = [
                obs["obstacle"]["center"]["x"],
                obs["obstacle"]["center"]["y"],
                obs["obstacle"]["center"]["z"],
            ]

            ori = obs["obstacle"]["orientation"]
            rot = Rotation.from_quat([ori["x"], ori["y"], ori["z"], ori["w"]])
            pose[:3, :3] = rot.as_matrix()

            poses.append(pose)

        poses = np.array(poses)

        # Interpolate poses and sizes for each target timestamp
        interp_poses = []
        interp_sizes = []

        for target_ts in valid_target_timestamps:
            # Find the two surrounding observations for this timestamp
            if target_ts <= track_timestamps[0]:
                # Before first observation - use first pose
                interp_poses.append(poses[0])
                interp_sizes.append(sizes[0])
            elif target_ts >= track_timestamps[-1]:
                # After last observation - use last pose
                interp_poses.append(poses[-1])
                interp_sizes.append(sizes[-1])
            else:
                # Find surrounding observations
                idx_after = np.searchsorted(track_timestamps, target_ts)
                idx_before = idx_after - 1

                # Calculate interpolation factor
                t = (target_ts - track_timestamps[idx_before]) / (
                    track_timestamps[idx_after] - track_timestamps[idx_before]
                )

                # Use interpolate_pose for pose interpolation
                interp_pose = interpolate_pose(poses[idx_before], poses[idx_after], t)
                interp_poses.append(interp_pose)

                # Linear interpolation for sizes
                interp_size = (1 - t) * sizes[idx_before] + t * sizes[idx_after]
                interp_sizes.append(interp_size)

        interp_poses = np.array(interp_poses)
        interp_sizes = np.array(interp_sizes)

        # Get object type
        first_obs = observations[0]["obstacle"]
        category_map = {
            "automobile": "Car",
            "other_vehicle": "Car",
            "vehicle": "Car",
            "car": "Car",
            "pedestrian": "Pedestrian",
            "person": "Pedestrian",
            "bicycle": "Cyclist",
            "cyclist": "Cyclist",
            "motorcycle": "Cyclist",
            "rider": "Cyclist",
            "bus": "Truck",
            "truck": "Truck",
            "heavy_truck": "Truck",
            "train_or_tram_car": "Truck",
            "trolley_bus": "Truck",
            "trailer": "Truck",
        }
        object_type = category_map.get(first_obs.get("category", "default").lower(), "Others")

        # Add interpolated observations to frames
        for i, frame_idx in enumerate(valid_indices):
            frame_key = f"{frame_idx:06d}.all_object_info.json"

            frame_obstacles[frame_key][str(track_id)] = {
                "object_to_world": interp_poses[i].tolist(),
                "object_lwh": interp_sizes[i].tolist(),
                "object_type": object_type,
                "object_is_moving": True,
            }

    return frame_obstacles


def load_camera_from_calibration_estimate(
    calibration_file: Path,
    camera_names: list[str],
    resize_hw: tuple[int, int],
) -> dict[str, tuple[FThetaCamera, dict]]:
    """Load camera models and extrinsics from calibration_estimate.parquet."""

    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    cal_df = pd.read_parquet(calibration_file)
    cal_data = cal_df.iloc[0]["calibration_estimate"]

    rig_data = json.loads(str(cal_data["rig_json"]))

    camera_dicts = {}
    for camera_name in camera_names:
        camera_dict = None
        found_cameras = []
        for sensor in rig_data["rig"]["sensors"]:
            found_cameras.append(sensor["name"])
            if sensor["name"] == camera_name:
                camera_dict = sensor
                break

        if camera_dict is None:
            raise ValueError(f"Camera {camera_name} not found in calibration data. Available cameras: {found_cameras}")

        # Use the existing FThetaCamera factory method to parse calibration data
        camera = FThetaCamera.from_dict(camera_dict)

        resize_w, resize_h = resize_hw
        rescale_h = resize_h / camera.height
        rescale_w = resize_w / camera.width

        if abs(rescale_h - rescale_w) > 0.02:
            logger.warning(f"Warning: Non-uniform scaling detected ({rescale_h:.3f} vs {rescale_w:.3f})")
        camera.rescale(rescale_h)

        extrinsics = camera_dict.get("nominalSensor2Rig_FLU", {})
        camera_dicts[camera_name] = (camera, extrinsics)

    return camera_dicts


def load_ego_poses_interpolated(egomotion_file: Path, input_pose_fps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ego poses from egomotion_estimate.parquet and interpolate from 10Hz to 30Hz."""

    if not egomotion_file.exists():
        raise FileNotFoundError(f"Ego motion file not found: {egomotion_file}")

    ego_df = pd.read_parquet(egomotion_file)

    positions_10hz = []
    quaternions_10hz = []
    timestamps_10hz = []

    for _, row in ego_df.iterrows():
        ego_data = row["egomotion_estimate"]
        key = row["key"]

        if "location" in ego_data and "orientation" in ego_data:
            loc = ego_data["location"]
            ori = ego_data["orientation"]

            positions_10hz.append([loc["x"], loc["y"], loc["z"]])
            quaternions_10hz.append([ori["x"], ori["y"], ori["z"], ori["w"]])

            if isinstance(key, dict) and "timestamp_micros" in key:
                timestamps_10hz.append(key["timestamp_micros"])

    positions_10hz = np.array(positions_10hz)
    quaternions_10hz = np.array(quaternions_10hz)
    timestamps_10hz = np.array(timestamps_10hz)

    # Create 30Hz timestamps
    duration = (timestamps_10hz[-1] - timestamps_10hz[0]) / 1e6  # seconds
    num_frames_30hz = int(duration * input_pose_fps) + 1
    timestamps_30hz = np.linspace(timestamps_10hz[0], timestamps_10hz[-1], num_frames_30hz)

    # Interpolate positions
    positions_30hz = []
    for i in range(3):  # x, y, z
        f = interp1d(timestamps_10hz, positions_10hz[:, i], kind="linear", fill_value="extrapolate")  # pyright: ignore[reportArgumentType]
        positions_30hz.append(f(timestamps_30hz))
    positions_30hz = np.array(positions_30hz).T

    # Interpolate quaternions using SLERP
    rotations_10hz = Rotation.from_quat(quaternions_10hz)
    slerp = Slerp(timestamps_10hz, rotations_10hz)
    rotations_30hz = slerp(timestamps_30hz)
    quaternions_30hz = rotations_30hz.as_quat()

    return positions_30hz, quaternions_30hz, timestamps_30hz


def interpolate_polyline(points: np.ndarray, factor: int = 10) -> np.ndarray:
    """Interpolate between points in a polyline to increase density."""
    if len(points) < 2:
        return points

    start_points = points[:-1]
    end_points = points[1:]
    t_values = np.linspace(0, 1, factor + 1, endpoint=False)[1:]
    t_values = t_values[:, None, None]
    start_points = start_points[None, :, :]
    end_points = end_points[None, :, :]
    interpolated = (1 - t_values) * start_points + t_values * end_points
    interpolated = interpolated.transpose(1, 0, 2).reshape(-1, 3)
    return np.vstack([points[0:1], interpolated, points[-1:]])


def load_map_data(
    lane_file: Path | None = None,
    lane_line_file: Path | None = None,
    road_boundary_file: Path | None = None,
    crosswalk_file: Path | None = None,
    pole_file: Path | None = None,
    road_marking_file: Path | None = None,
    wait_line_file: Path | None = None,
    traffic_light_file: Path | None = None,
    traffic_sign_file: Path | None = None,
) -> dict[str, list]:
    """Load ALL map data from parquet files.

    Returns:
        Dict with map elements. For lanelines, each element is a tuple (polyline, type_string)
        where type_string is like "WHITE SOLID_SINGLE", "YELLOW DASHED_SINGLE", etc.
    """
    map_data = {
        "lanes": [],  # For Cosmos compatibility
        "lanelines": [],  # List of tuples: (polyline, type_string)
        "road_boundaries": [],
        "crosswalks": [],
        "traffic_lights": [],
        "traffic_signs": [],
        "poles": [],
        "road_markings": [],
        "wait_lines": [],
        "intersection_areas": [],
        "road_islands": [],
    }

    # Load lane boundaries
    if lane_file is not None:
        lanes_df = pd.read_parquet(lane_file)
        for _, row in lanes_df.iterrows():
            lane = row["lane"]

            # Process left rail
            if "left_rail" in lane and lane["left_rail"] is not None:
                points = []
                for pt in lane["left_rail"]:
                    points.append([pt["x"], pt["y"], pt["z"]])
                if len(points) > 1:
                    dense_points = interpolate_polyline(np.array(points), factor=10)
                    map_data["lanes"].append(dense_points)

            # Process right rail
            if "right_rail" in lane and lane["right_rail"] is not None:
                points = []
                for pt in lane["right_rail"]:
                    points.append([pt["x"], pt["y"], pt["z"]])
                if len(points) > 1:
                    dense_points = interpolate_polyline(np.array(points), factor=10)
                    map_data["lanes"].append(dense_points)

    # Load lane lines
    if lane_line_file is not None:
        lane_lines_df = pd.read_parquet(lane_line_file)
        for _, row in lane_lines_df.iterrows():
            lane_line = row["lane_line"]

            # Check if we have the newer format with line_rail instead of path
            if "line_rail" in lane_line and lane_line["line_rail"] is not None:
                points = []
                for pt in lane_line["line_rail"]:
                    points.append([pt["x"], pt["y"], pt["z"]])

                if len(points) > 1:
                    dense_points = interpolate_polyline(np.array(points), factor=10)

                    # Extract type information if available
                    lane_type = "WHITE SOLID_SINGLE"  # Default
                    if "colors" in lane_line and "styles" in lane_line:
                        colors = lane_line["colors"]
                        styles = lane_line["styles"]

                        # Find most common type combination for this line
                        if len(colors) > 0 and len(styles) > 0:
                            type_combinations = [f"{c} {s}" for c, s in zip(colors, styles, strict=False)]
                            most_common_type = Counter(type_combinations).most_common(1)[0][0]
                            lane_type = most_common_type

                    # Store as tuple (polyline, type)
                    map_data["lanelines"].append((dense_points, lane_type))

            # Fallback to old format with path
            elif "path" in lane_line and lane_line["path"] is not None:
                points = []
                for pt in lane_line["path"]:
                    points.append([pt["x"], pt["y"], pt["z"]])
                if len(points) > 1:
                    dense_points = interpolate_polyline(np.array(points), factor=10)
                    # For old format, use default type
                    map_data["lanelines"].append((dense_points, "WHITE SOLID_SINGLE"))

    # Load road boundaries
    if road_boundary_file is not None:
        road_boundaries_df = pd.read_parquet(road_boundary_file)
        for _, row in road_boundaries_df.iterrows():
            boundary = row["road_boundary"]
            if "location" in boundary:
                loc = boundary["location"]
                points = [[pt["x"], pt["y"], pt["z"]] for pt in loc]
                if len(points) > 1:
                    dense_points = interpolate_polyline(np.array(points), factor=10)
                    map_data["road_boundaries"].append(dense_points)

    # Load crosswalks (polygons)
    if crosswalk_file is not None:
        crosswalks_df = pd.read_parquet(crosswalk_file)
        for _, row in crosswalks_df.iterrows():
            crosswalk = row["crosswalk"]
            if "location" in crosswalk:
                loc = crosswalk["location"]
                points = [[pt["x"], pt["y"], pt["z"]] for pt in loc]
                if len(points) > 2:
                    map_data["crosswalks"].append(np.array(points))

    # Load poles (2-point features)
    if pole_file is not None:
        poles_df = pd.read_parquet(pole_file)
        for _, row in poles_df.iterrows():
            pole = row["pole"]
            if "location" in pole:
                loc = pole["location"]
                if len(loc) >= 2:
                    # Use the provided 2 points (base and top)
                    points = [[pt["x"], pt["y"], pt["z"]] for pt in loc]
                    dense_points = interpolate_polyline(np.array(points), factor=10)
                    map_data["poles"].append(dense_points)
                elif len(loc) == 1:
                    # If only one point, create vertical line
                    base = [loc[0]["x"], loc[0]["y"], loc[0]["z"]]
                    top = [loc[0]["x"], loc[0]["y"], loc[0]["z"] + 3.0]
                    dense_points = interpolate_polyline(np.array([base, top]), factor=10)
                    map_data["poles"].append(dense_points)

    # Load road markings (polygons)
    if road_marking_file is not None:
        road_markings_df = pd.read_parquet(road_marking_file)
        for _, row in road_markings_df.iterrows():
            marking = row["road_marking"]
            if "location" in marking:
                loc = marking["location"]
                points = [[pt["x"], pt["y"], pt["z"]] for pt in loc]
                if len(points) > 1:
                    # Road markings are polygons, close them if needed
                    if len(points) > 2:
                        map_data["road_markings"].append(np.array(points))

    # Load wait lines
    if wait_line_file is not None:
        wait_lines_df = pd.read_parquet(wait_line_file)
        for _, row in wait_lines_df.iterrows():
            wait_line = row["wait_line"]
            if "location" in wait_line:
                loc = wait_line["location"]
                points = [[pt["x"], pt["y"], pt["z"]] for pt in loc]
                if len(points) >= 2:
                    dense_points = interpolate_polyline(np.array(points), factor=5)
                    map_data["wait_lines"].append(dense_points)

    # Load traffic lights (point features with center)
    if traffic_light_file is not None:
        traffic_lights_df = pd.read_parquet(traffic_light_file)
        for _, row in traffic_lights_df.iterrows():
            light = row["traffic_light"]
            center = [light["center"][k] for k in ["x", "y", "z"]]
            dimension = [light["dimensions"][k] for k in ["x", "y", "z"]]
            orientation = [light["orientation"][k] for k in ["x", "y", "z", "w"]]

            if not all(c is not None for c in dimension):
                dimension = [0.3 * 2, 0.3 * 2, 0.5 * 2]

            # Validate that all coordinates are not None
            if all(c is not None for c in center) and all(p is not None for p in orientation):
                half_w = dimension[0] / 2
                half_d = dimension[1] / 2
                half_h = dimension[2] / 2

                # Create 8 vertices for the cuboid in local coordinates
                local_vertices = np.array(
                    [
                        # Bottom face (counter-clockwise from top view)
                        [-half_w, -half_d, -half_h],  # 0
                        [+half_w, -half_d, -half_h],  # 1
                        [+half_w, +half_d, -half_h],  # 2
                        [-half_w, +half_d, -half_h],  # 3
                        # Top face (counter-clockwise from top view)
                        [-half_w, -half_d, +half_h],  # 4
                        [+half_w, -half_d, +half_h],  # 5
                        [+half_w, +half_d, +half_h],  # 6
                        [-half_w, +half_d, +half_h],  # 7
                    ]
                )

                # Apply rotation using the orientation quaternion
                rot = Rotation.from_quat(orientation)
                rotated_vertices = rot.apply(local_vertices)

                # Translate to world position
                vertices = rotated_vertices + np.array(center)

                # Convert to polyline using the same connectivity as cuboid3d_to_polyline
                connected_indices = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
                polyline = [vertices[i] for i in connected_indices]

                map_data["traffic_lights"].append(np.array(polyline))

    # Load traffic signs (point features with center)
    if traffic_sign_file is not None:
        traffic_signs_df = pd.read_parquet(traffic_sign_file)
        for _, row in traffic_signs_df.iterrows():
            sign = row["traffic_sign"]
            center = [sign["center"][k] for k in ["x", "y", "z"]]
            dimension = [sign["dimensions"][k] for k in ["x", "y", "z"]]
            orientation = [sign["orientation"][k] for k in ["x", "y", "z", "w"]]
            # not used currently
            category = sign["category"]

            if not all(c is not None for c in dimension):
                dimension = [0.4 * 2, 0.15 * 2, 0.4 * 2]

            # Validate that all coordinates are not None
            if all(c is not None for c in center) and all(p is not None for p in orientation):
                half_w = dimension[0] / 2
                half_d = dimension[1] / 2
                half_h = dimension[2] / 2

                # Create 8 vertices for the cuboid in local coordinates
                local_vertices = np.array(
                    [
                        # Bottom face (counter-clockwise from top view)
                        [-half_w, -half_d, -half_h],  # 0
                        [+half_w, -half_d, -half_h],  # 1
                        [+half_w, +half_d, -half_h],  # 2
                        [-half_w, +half_d, -half_h],  # 3
                        # Top face (counter-clockwise from top view)
                        [-half_w, -half_d, +half_h],  # 4
                        [+half_w, -half_d, +half_h],  # 5
                        [+half_w, +half_d, +half_h],  # 6
                        [-half_w, +half_d, +half_h],  # 7
                    ]
                )

                # Apply rotation using the orientation quaternion
                rot = Rotation.from_quat(orientation)
                rotated_vertices = rot.apply(local_vertices)

                # Translate to world position
                vertices = rotated_vertices + np.array(center)

                # Convert to polyline using the same connectivity as cuboid3d_to_polyline
                connected_indices = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
                polyline = [vertices[i] for i in connected_indices]

                map_data["traffic_signs"].append(np.array(polyline))

    # Count loaded elements
    logger.info("Loaded map elements:")
    for name, elements in map_data.items():
        if elements:
            if name == "lanelines" and elements and isinstance(elements[0], tuple):
                # Count unique lane line types
                type_counts = {}
                for _, lane_type in elements:
                    type_counts[lane_type] = type_counts.get(lane_type, 0) + 1
                logger.info(f"  {name:20s}: {len(elements):4d} elements")
                for lane_type, count in sorted(type_counts.items()):
                    logger.info(f"    - {lane_type:30s}: {count:3d}")
            else:
                logger.info(f"  {name:20s}: {len(elements):4d} elements")

    return map_data
