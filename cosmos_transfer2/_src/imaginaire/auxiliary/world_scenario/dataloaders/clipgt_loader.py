# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
ClipGT data loader implementation.

This module provides a loader for ClipGT format data, converting it to the
unified SceneData representation.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import SceneDataLoader, auto_register
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import (
    Crosswalk,
    DynamicObject,
    EgoPose,
    IntersectionArea,
    LaneBoundary,
    LaneLine,
    LaneLineColor,
    LaneLineStyle,
    ObjectType,
    Pole,
    RoadBoundary,
    RoadIsland,
    RoadMarking,
    SceneData,
    TrafficLight,
    TrafficSign,
    TrafficSignType,
    WaitLine,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.data_utils import (
    convert_points_flu_to_rdf,
    convert_quaternions_flu_to_rdf,
    normalize_quaternions,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.laneline_utils import build_lane_line_type


@auto_register(priority=10)  # High priority for ClipGT format
class ClipGTLoader(SceneDataLoader):
    """Loader for ClipGT format data."""

    @property
    def name(self) -> str:
        """Get the loader name."""
        return "clipgt"

    @property
    def description(self) -> str:
        """Get loader description."""
        return "Loader for ClipGT parquet-based scene data format"

    def _detect_clip_id(self, path: Path) -> Optional[str]:
        """
        Detect the clip_id from a directory.
        First tries to use the directory name, then scans for parquet files.
        """
        # Try directory name as clip_id first
        clip_id = path.name
        required_files = [
            f"{clip_id}.calibration_estimate.parquet",
            f"{clip_id}.egomotion_estimate.parquet",
        ]
        if all((path / f).exists() for f in required_files):
            return clip_id

        # If not found, scan for any calibration_estimate.parquet files
        calib_files = list(path.glob("*.calibration_estimate.parquet"))
        for calib_file in calib_files:
            # Extract clip_id from filename: {clip_id}.calibration_estimate.parquet
            clip_id = calib_file.name.replace(".calibration_estimate.parquet", "")
            required_files = [
                f"{clip_id}.calibration_estimate.parquet",
                f"{clip_id}.egomotion_estimate.parquet",
            ]
            if all((path / f).exists() for f in required_files):
                return clip_id

        return None

    def can_load(self, source: Union[Path, str, Dict[str, Any]]) -> bool:
        """Check if source is a ClipGT directory."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return self._detect_clip_id(path) is not None
        return False

    def load(
        self,
        source: Union[Path, str, Dict[str, Any]],
        camera_names: Optional[List[str]] = None,
        max_frames: int = -1,
        input_pose_fps: int = 30,
        resize_resolution_hw: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> SceneData:
        """
        Load ClipGT scene data.

        Args:
            source: ClipGT directory path
            camera_names: Optional list of camera names to load
            max_frames: Maximum frames to load
            input_pose_fps: Target frame rate for interpolation
            resize_resolution_hw: Optional camera resize resolution
            **kwargs: Additional arguments

        Returns:
            Loaded scene data
        """
        if not isinstance(source, (str, Path)):
            raise TypeError(f"ClipGTLoader only supports string or Path sources, got {type(source)!r}")

        clipgt_path = Path(source)
        clip_id = self._detect_clip_id(clipgt_path)

        if clip_id is None:
            raise ValueError(
                f"Could not detect clip_id from directory: {clipgt_path}. "
                "Expected files: {{clip_id}}.calibration_estimate.parquet and {{clip_id}}.egomotion_estimate.parquet"
            )

        logger.debug(f"Loading ClipGT data from: {clipgt_path} (clip_id: {clip_id})")

        # Initialize scene data
        scene_data = SceneData(scene_id=clip_id, frame_rate=input_pose_fps, duration_seconds=0.0)

        # Define file paths
        files = {
            "calibration": clipgt_path / f"{clip_id}.calibration_estimate.parquet",
            "egomotion": clipgt_path / f"{clip_id}.egomotion_estimate.parquet",
            "obstacle": clipgt_path / f"{clip_id}.obstacle.parquet",
            "lane": clipgt_path / f"{clip_id}.lane.parquet",
            "lane_line": clipgt_path / f"{clip_id}.lane_line.parquet",
            "road_boundary": clipgt_path / f"{clip_id}.road_boundary.parquet",
            "crosswalk": clipgt_path / f"{clip_id}.crosswalk.parquet",
            "pole": clipgt_path / f"{clip_id}.pole.parquet",
            "road_marking": clipgt_path / f"{clip_id}.road_marking.parquet",
            "wait_line": clipgt_path / f"{clip_id}.wait_line.parquet",
            "traffic_light": clipgt_path / f"{clip_id}.traffic_light.parquet",
            "traffic_sign": clipgt_path / f"{clip_id}.traffic_sign.parquet",
            "intersection_area": clipgt_path / f"{clip_id}.intersection_area.parquet",
            "road_island": clipgt_path / f"{clip_id}.road_island.parquet",
            "buffer_zone": clipgt_path / f"{clip_id}.buffer_zone.parquet",
            "camera_timestamps": clipgt_path / f"{clip_id}.camera_front_wide_120fov.json",
        }

        # Load ego poses (use camera timestamps if available for frame-accurate sync)
        if files["egomotion"].exists():
            self._load_ego_poses(
                scene_data,
                files["egomotion"],
                input_pose_fps,
                max_frames,
                # Use camera timestamps as reference (cameras are assumed to be synchronized)
                camera_timestamps_file=files["camera_timestamps"] if files["camera_timestamps"].exists() else None,
            )

        # Load camera calibrations
        if files["calibration"].exists():
            self._load_camera_calibrations(scene_data, files["calibration"], camera_names, resize_resolution_hw)

        # Load dynamic objects
        if files["obstacle"].exists():
            self._load_dynamic_objects(scene_data, files["obstacle"])

        # Load map elements
        self._load_map_elements(scene_data, files)

        return scene_data

    def _load_ego_poses(
        self,
        scene_data: SceneData,
        ego_file: Path,
        target_fps: int,
        max_frames: int,
        camera_timestamps_file: Optional[Path] = None,
    ) -> None:
        """Load and interpolate ego poses.

        If a camera timestamp JSON file is provided, the poses will be interpolated
        to match camera frame timestamps for frame-accurate synchronization.
        Otherwise, poses are interpolated to a uniform target_fps.

        Args:
            scene_data: Scene data to populate with ego poses
            ego_file: Path to egomotion parquet file
            target_fps: Target frame rate for interpolation (used if no camera timestamps)
            max_frames: Maximum number of frames to load (-1 for all)
            camera_timestamps_file: Optional path to camera timestamp JSON file
        """
        ego_df = pd.read_parquet(ego_file)

        positions = []
        quaternions = []
        timestamps = []

        for _, row in ego_df.iterrows():
            ego_data = row["egomotion_estimate"]
            key = row["key"]

            if "location" in ego_data and "orientation" in ego_data:
                loc = ego_data["location"]
                ori = ego_data["orientation"]

                positions.append([loc["x"], loc["y"], loc["z"]])
                quaternions.append([ori["x"], ori["y"], ori["z"], ori["w"]])

                if isinstance(key, dict) and "timestamp_micros" in key:
                    timestamps.append(key["timestamp_micros"])

        if not timestamps:
            logger.warning("No ego poses found")
            return

        positions = np.array(positions)
        quaternions = np.array(quaternions)
        timestamps = np.array(timestamps)

        # Try to load camera timestamps for frame-accurate synchronization
        camera_timestamps = self._load_camera_timestamps(camera_timestamps_file)

        if camera_timestamps is not None:
            # Use camera timestamps for interpolation (frame-accurate sync)
            target_timestamps_micros = camera_timestamps.astype(np.float64)

            # Apply max_frames limit
            if max_frames > 0 and len(target_timestamps_micros) > max_frames:
                target_timestamps_micros = target_timestamps_micros[:max_frames]

            # Log if extrapolation will be needed
            ego_start, ego_end = timestamps[0], timestamps[-1]
            n_before = int(np.sum(target_timestamps_micros < ego_start))
            n_after = int(np.sum(target_timestamps_micros > ego_end))
            if n_before > 0 or n_after > 0:
                logger.warning(
                    f"Extrapolating {n_before} frames before and {n_after} frames after ego pose range "
                    f"(using constant velocity assumption)"
                )

            num_frames = len(target_timestamps_micros)
            duration = (target_timestamps_micros[-1] - target_timestamps_micros[0]) / 1e6
            sync_mode = "camera timestamps"
        else:
            # Fall back to uniform FPS interpolation
            duration = (timestamps[-1] - timestamps[0]) / 1e6  # seconds
            num_frames = int(duration * target_fps) + 1

            if max_frames > 0:
                num_frames = min(num_frames, max_frames)

            # Create target timestamps at the desired frame rate
            target_timestamps_seconds = np.linspace(0, (num_frames - 1) / target_fps, num_frames)
            target_timestamps_micros = timestamps[0] + (target_timestamps_seconds * 1e6)
            sync_mode = f"{target_fps} Hz"

        # Interpolate positions
        interp_positions = []
        for i in range(3):  # x, y, z
            f = interp1d(
                timestamps,
                positions[:, i],
                kind="linear",
                fill_value=cast(Any, "extrapolate"),
            )
            interp_positions.append(f(target_timestamps_micros))
        interp_positions = np.array(interp_positions).T

        # Interpolate quaternions using SLERP with extrapolation support
        interp_quaternions = self._interpolate_quaternions_with_extrapolation(
            timestamps, quaternions, target_timestamps_micros
        )

        # Create EgoPose objects with proper microsecond timestamps (OpenCV RDF)
        for i in range(num_frames):
            # Convert position
            pos_flu = interp_positions[i].astype(np.float32)
            pos_rdf = convert_points_flu_to_rdf(pos_flu.reshape(1, 3))[0]

            # Convert orientation via basis change: R_rdf = S * R_flu * S^T
            quat_rdf = convert_quaternions_flu_to_rdf(
                interp_quaternions[i].reshape(1, 4),
                double_sided=True,
            )[0]

            # Maintain original microsecond resolution
            timestamp_us = int(target_timestamps_micros[i])

            scene_data.ego_poses.append(
                EgoPose(
                    timestamp=timestamp_us,
                    position=pos_rdf,
                    orientation=quat_rdf,
                )
            )

        scene_data.duration_seconds = duration
        scene_data.metadata["coordinate_frame"] = "opencv_rdf"
        logger.debug(f"Loaded {num_frames} ego poses at {sync_mode} (OpenCV RDF)")

    def _load_camera_timestamps(self, camera_timestamps_file: Optional[Path]) -> Optional[np.ndarray]:
        """Load camera frame timestamps from JSON file.

        Args:
            camera_timestamps_file: Path to camera timestamp JSON file

        Returns:
            Array of timestamps in microseconds, or None if file not provided or invalid
        """
        if camera_timestamps_file is None:
            return None

        try:
            with open(camera_timestamps_file) as f:
                cam_data = json.load(f)

            timestamps = np.array([frame["timestamp"] for frame in cam_data], dtype=np.int64)
            logger.info(
                f"Using camera timestamps from {camera_timestamps_file.name} "
                f"({len(timestamps)} frames) for pose interpolation"
            )
            return timestamps
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to load camera timestamps from {camera_timestamps_file}: {e}")
            return None

    def _interpolate_quaternions_with_extrapolation(
        self,
        keyframe_timestamps: np.ndarray,
        keyframe_quaternions: np.ndarray,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Interpolate quaternions with constant angular velocity extrapolation.

        Uses SLERP for interpolation within the keyframe range, and constant
        angular velocity extrapolation for timestamps outside the range.

        Args:
            keyframe_timestamps: Timestamps of keyframe poses (microseconds)
            keyframe_quaternions: Quaternions at keyframe poses, shape (N, 4)
            target_timestamps: Target timestamps to interpolate to (microseconds)

        Returns:
            Interpolated quaternions, shape (M, 4)
        """
        t_start, t_end = keyframe_timestamps[0], keyframe_timestamps[-1]

        # Identify which timestamps need extrapolation vs interpolation
        before_mask = target_timestamps < t_start
        after_mask = target_timestamps > t_end
        interp_mask = ~before_mask & ~after_mask

        result = np.zeros((len(target_timestamps), 4), dtype=np.float64)

        # Handle interpolation (within range) using SLERP
        if np.any(interp_mask):
            rotations = Rotation.from_quat(keyframe_quaternions)
            slerp = Slerp(keyframe_timestamps.astype(np.float64), rotations)
            interp_rotations = slerp(target_timestamps[interp_mask])
            result[interp_mask] = interp_rotations.as_quat()

        # Handle extrapolation before the first keyframe
        if np.any(before_mask):
            # Compute angular velocity from first two keyframes
            r0 = Rotation.from_quat(keyframe_quaternions[0])
            r1 = Rotation.from_quat(keyframe_quaternions[1])
            dt = keyframe_timestamps[1] - keyframe_timestamps[0]  # microseconds
            # Relative rotation from r0 to r1
            delta_r = r0.inv() * r1
            # Angular velocity as rotation vector (radians per microsecond)
            omega = delta_r.as_rotvec() / dt

            # Extrapolate backwards
            for i in np.where(before_mask)[0]:
                dt_extrap = target_timestamps[i] - t_start  # negative value
                delta_extrap = Rotation.from_rotvec(omega * dt_extrap)
                result[i] = (r0 * delta_extrap).as_quat()

        # Handle extrapolation after the last keyframe
        if np.any(after_mask):
            # Compute angular velocity from last two keyframes
            r_m1 = Rotation.from_quat(keyframe_quaternions[-2])
            r_m0 = Rotation.from_quat(keyframe_quaternions[-1])
            dt = keyframe_timestamps[-1] - keyframe_timestamps[-2]  # microseconds
            # Relative rotation from r_m1 to r_m0
            delta_r = r_m1.inv() * r_m0
            # Angular velocity as rotation vector (radians per microsecond)
            omega = delta_r.as_rotvec() / dt

            # Extrapolate forwards
            for i in np.where(after_mask)[0]:
                dt_extrap = target_timestamps[i] - t_end  # positive value
                delta_extrap = Rotation.from_rotvec(omega * dt_extrap)
                result[i] = (r_m0 * delta_extrap).as_quat()

        return result

    def _load_camera_calibrations(
        self,
        scene_data: SceneData,
        cal_file: Path,
        camera_names: Optional[List[str]],
        resize_hw: Optional[Tuple[int, int]],
    ) -> None:
        """Load camera calibration data."""
        cal_df = pd.read_parquet(cal_file)
        cal_data = cal_df.iloc[0]["calibration_estimate"]
        rig_data = json.loads(str(cal_data["rig_json"]))

        # Build a map of sensor name variants: colon and underscore
        sensors = rig_data["rig"]["sensors"]
        name_to_sensor: dict[str, dict] = {}
        for sensor in sensors:
            name = sensor.get("name", "")
            if name:
                name_to_sensor[name] = sensor
                name_to_sensor[name.replace(":", "_")] = sensor

        # If no specific cameras requested, load all available cameras (underscore form)
        if camera_names is None:
            camera_names = [s["name"].replace(":", "_") for s in sensors if s.get("name", "").startswith("camera:")]

        for camera_name in camera_names:
            # Accept either underscore or colon input names
            camera_dict = name_to_sensor.get(camera_name)
            if camera_dict is None:
                camera_dict = name_to_sensor.get(camera_name.replace("_", ":"))

            if camera_dict is None:
                logger.warning(f"Camera {camera_name} not found in calibration")
                continue

            props = camera_dict["properties"]

            # Get polynomial coefficients
            poly_key = "polynomial" if "polynomial" in props else "bw-poly"
            if poly_key not in props:
                logger.warning(f"No polynomial coefficients for {camera_name}")
                continue

            poly_str = props[poly_key]
            poly_coeffs = np.array([float(x) for x in poly_str.split()], dtype=np.float32)
            if len(poly_coeffs) == 5:
                poly_coeffs = np.append(poly_coeffs, 0.0)

            # Get intrinsics
            cx = float(props["cx"])
            cy = float(props["cy"])
            width = int(props["width"])
            height = int(props["height"])

            # Determine polynomial direction from polynomial-type field
            # pixeldistance-to-angle = backward (r → θ) = needs inversion
            # angle-to-pixeldistance = forward (θ → r) = use directly
            poly_type = props.get("polynomial-type", "")
            if poly_type == "angle-to-pixeldistance":
                is_bw_poly = False  # Forward polynomial
            elif poly_type == "pixeldistance-to-angle":
                is_bw_poly = True  # Backward polynomial
            elif poly_key == "bw-poly":
                is_bw_poly = True  # Explicitly named backward poly
            else:
                # Heuristic fallback: backward poly has small c1 (radians/pixel)
                # Forward poly has large c1 (pixels/radian, ~focal length)
                is_bw_poly = len(poly_coeffs) > 1 and abs(poly_coeffs[1]) < 1.0

            # Get linear affine term [[C,D],[D,E]] - defaults to identity
            linear_c = float(props.get("linear-c", 1.0))
            linear_d = float(props.get("linear-d", 0.0))
            linear_e = float(props.get("linear-e", 0.0))
            linear_cde = np.array([linear_c, linear_d, linear_e], dtype=np.float32)

            camera_model = FThetaCamera(
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                poly=poly_coeffs.copy(),
                is_bw_poly=is_bw_poly,
                linear_cde=linear_cde.copy(),
            )

            # Apply resize if specified
            if resize_hw:
                resize_h, resize_w = resize_hw
                scale_h = resize_h / height
                scale_w = resize_w / width

                camera_model.rescale(ratio_h=scale_h, ratio_w=scale_w)

            cx = float(camera_model._center[0])
            cy = float(camera_model._center[1])
            width = int(camera_model.width)
            height = int(camera_model.height)
            poly_coeffs = camera_model._intrinsics[4:10].astype(np.float32)
            linear_cde = camera_model.linear_cde.astype(np.float32)

            # Get extrinsics
            extrinsics = camera_dict.get("nominalSensor2Rig_FLU", {})
            camera_to_vehicle = np.eye(4, dtype=np.float32)

            if extrinsics:
                if "t" in extrinsics:
                    camera_to_vehicle[:3, 3] = extrinsics["t"]
                if "roll-pitch-yaw" in extrinsics:
                    rpy = extrinsics["roll-pitch-yaw"]
                    rot = Rotation.from_euler("xyz", np.radians(rpy))
                    camera_to_vehicle[:3, :3] = rot.as_matrix()

            # Convert camera_to_vehicle from FLU to OpenCV RDF
            S = np.array(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            R_flu = camera_to_vehicle[:3, :3]
            t_flu = camera_to_vehicle[:3, 3]
            R_rdf = S @ R_flu @ S.T
            t_rdf = S @ t_flu
            camera_to_vehicle[:3, :3] = R_rdf
            camera_to_vehicle[:3, 3] = t_rdf

            # Use underscore canonical name for calibration key and record
            canon_name = camera_name.replace(":", "_")
            scene_data.camera_models[canon_name] = camera_model
            scene_data.camera_extrinsics[canon_name] = camera_to_vehicle

        logger.debug(f"Loaded calibrations for {len(scene_data.camera_models)} cameras")

    def _load_dynamic_objects(self, scene_data: SceneData, obstacle_file: Path) -> None:
        """Load dynamic object tracks."""
        df = pd.read_parquet(obstacle_file)

        # Group observations by track ID
        tracks = defaultdict(list)

        for _, row in df.iterrows():
            obstacle = row["obstacle"]
            key = row["key"]

            if "trackline_id" in obstacle and "timestamp_micros" in key:
                track_id = str(obstacle["trackline_id"])
                if track_id.startswith("labelstore:"):
                    # Skip duplicated labelstore entries (numeric counterpart exists)
                    continue
                timestamp = key["timestamp_micros"]

                tracks[track_id].append(
                    {
                        "timestamp": timestamp,
                        "obstacle": obstacle,
                    }
                )

        # Process each track
        for track_id, observations in tracks.items():
            if len(observations) < 2:
                continue  # Skip single-observation tracks

            # Sort by timestamp
            observations.sort(key=lambda x: x["timestamp"])

            # Extract trajectory data
            timestamps = []
            centers = []
            dimensions = []
            orientations = []

            for obs in observations:
                obstacle = obs["obstacle"]

                timestamps.append(obs["timestamp"])

                center = obstacle["center"]
                centers.append([center["x"], center["y"], center["z"]])

                size = obstacle["size"]
                dimensions.append([size["x"], size["y"], size["z"]])

                ori = obstacle["orientation"]
                orientations.append([ori["x"], ori["y"], ori["z"], ori["w"]])

            # Convert to numpy arrays
            timestamps = np.array(timestamps, dtype=np.int64)
            centers = np.array(centers, dtype=np.float32)
            dimensions = np.array(dimensions, dtype=np.float32)
            orientations = np.array(orientations, dtype=np.float32)

            normalized_quats, valid_mask = normalize_quaternions(orientations)
            if not np.all(valid_mask):
                logger.warning(
                    "Skipping %d zero-norm quaternions for track %s",
                    np.count_nonzero(~valid_mask),
                    track_id,
                )
            timestamps = timestamps[valid_mask]
            centers = centers[valid_mask]
            dimensions = dimensions[valid_mask]

            if (
                len(timestamps) < 2
                or np.isnan(centers).any()
                or np.isnan(dimensions).any()
                or np.isnan(orientations).any()
            ):
                continue

            centers = convert_points_flu_to_rdf(centers)
            orientations = convert_quaternions_flu_to_rdf(normalized_quats)

            # Map object type
            first_obs = observations[0]["obstacle"]
            category = first_obs.get("category", "unknown").lower()

            type_map = {
                "automobile": ObjectType.CAR,
                "other_vehicle": ObjectType.CAR,
                "vehicle": ObjectType.CAR,
                "car": ObjectType.CAR,
                "pedestrian": ObjectType.PEDESTRIAN,
                "person": ObjectType.PEDESTRIAN,
                "bicycle": ObjectType.CYCLIST,
                "cyclist": ObjectType.CYCLIST,
                "motorcycle": ObjectType.CYCLIST,
                "rider": ObjectType.CYCLIST,
                "bus": ObjectType.TRUCK,
                "truck": ObjectType.TRUCK,
                "heavy_truck": ObjectType.TRUCK,
                "train_or_tram_car": ObjectType.TRUCK,
                "trolley_bus": ObjectType.TRUCK,
                "trailer": ObjectType.TRUCK,
            }
            object_type = type_map.get(category, ObjectType.OTHER)

            # Create dynamic object
            scene_data.dynamic_objects[track_id] = DynamicObject(
                track_id=track_id,
                object_type=object_type,
                timestamps=timestamps,
                centers=centers,
                dimensions=dimensions,
                orientations=orientations,
                is_moving=True,
                max_extrapolation_us=500_000.0,
            )

        logger.debug(f"Loaded {len(scene_data.dynamic_objects)} dynamic object tracks")

    def _load_map_elements(self, scene_data: SceneData, files: Dict[str, Path]) -> None:
        """Load all map elements."""

        # Load lane boundaries
        if files["lane"].exists():
            self._load_lane_boundaries(scene_data, files["lane"])

        # Load lane lines
        if files["lane_line"].exists():
            self._load_lane_lines(scene_data, files["lane_line"])

        # Load road boundaries
        if files["road_boundary"].exists():
            self._load_road_boundaries(scene_data, files["road_boundary"])

        # Load crosswalks
        if files["crosswalk"].exists():
            self._load_crosswalks(scene_data, files["crosswalk"])

        # Load poles
        if files["pole"].exists():
            self._load_poles(scene_data, files["pole"])

        # Load road markings
        if files["road_marking"].exists():
            self._load_road_markings(scene_data, files["road_marking"])

        # Load wait lines
        if files["wait_line"].exists():
            self._load_wait_lines(scene_data, files["wait_line"])

        # Load traffic lights
        if files["traffic_light"].exists():
            self._load_traffic_lights(scene_data, files["traffic_light"])

        # Load traffic signs
        if files["traffic_sign"].exists():
            self._load_traffic_signs(scene_data, files["traffic_sign"])

        # Load intersection areas
        if files["intersection_area"].exists():
            self._load_intersection_areas(scene_data, files["intersection_area"])

        # Load road islands
        if files["road_island"].exists():
            self._load_road_islands(scene_data, files["road_island"])

    def _load_lane_boundaries(self, scene_data: SceneData, lane_file: Path) -> None:
        """Load lane boundaries."""
        df = pd.read_parquet(lane_file)

        for idx, row in df.iterrows():
            lane = row["lane"]
            lane_id = str(idx)

            # Process left rail
            if "left_rail" in lane and lane["left_rail"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane["left_rail"]],
                    dtype=np.float32,
                )
                if len(points) > 1 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.lane_boundaries.append(
                        LaneBoundary(
                            element_id=f"{lane_id}_left",
                            points=points,
                            is_left_boundary=True,
                            lane_id=lane_id,
                        )
                    )

            # Process right rail
            if "right_rail" in lane and lane["right_rail"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane["right_rail"]],
                    dtype=np.float32,
                )
                if len(points) > 1 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.lane_boundaries.append(
                        LaneBoundary(
                            element_id=f"{lane_id}_right",
                            points=points,
                            is_left_boundary=False,
                            lane_id=lane_id,
                        )
                    )

    def _load_lane_lines(self, scene_data: SceneData, lane_line_file: Path) -> None:
        """Load lane lines."""
        df = pd.read_parquet(lane_line_file)

        for idx, row in df.iterrows():
            lane_line = row["lane_line"]

            # Get points (check both new and old format)
            points = None
            if "line_rail" in lane_line and lane_line["line_rail"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane_line["line_rail"]],
                    dtype=np.float32,
                )
            elif "path" in lane_line and lane_line["path"] is not None:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in lane_line["path"]],
                    dtype=np.float32,
                )

            if points is None or len(points) < 2 or np.isnan(points).any():
                continue

            color = LaneLineColor.WHITE
            style = LaneLineStyle.SOLID_SINGLE

            if "colors" in lane_line and "styles" in lane_line:
                colors = lane_line["colors"]
                styles = lane_line["styles"]

                if len(colors) > 0 and len(styles) > 0:
                    # Find most common combination
                    combinations = list(zip(colors, styles, strict=False))
                    most_common = Counter(combinations).most_common(1)[0][0]
                    color_str, style_str = most_common

                    if color_str:
                        try:
                            color = LaneLineColor[color_str.upper()]
                        except KeyError:
                            color = LaneLineColor.UNKNOWN
                            logger.debug(f"Unknown lane line color: {color_str}, using UNKNOWN")
                    if style_str:
                        try:
                            style = LaneLineStyle[style_str.upper()]
                        except KeyError:
                            style = LaneLineStyle.UNKNOWN
                            logger.debug(f"Unknown lane line style: {style_str}, using UNKNOWN")

            points = convert_points_flu_to_rdf(points)
            scene_data.lane_lines.append(
                LaneLine(
                    element_id=f"lane_line_{idx}",
                    points=points,
                    lane_type=build_lane_line_type(color=color, style=style),
                )
            )

    def _load_road_boundaries(self, scene_data: SceneData, road_boundary_file: Path) -> None:
        """Load road boundaries."""
        df = pd.read_parquet(road_boundary_file)

        for idx, row in df.iterrows():
            boundary = row["road_boundary"]

            if "location" in boundary:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in boundary["location"]],
                    dtype=np.float32,
                )
                if len(points) > 1 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.road_boundaries.append(
                        RoadBoundary(
                            element_id=f"road_boundary_{idx}",
                            points=points,
                        )
                    )

    def _load_crosswalks(self, scene_data: SceneData, crosswalk_file: Path) -> None:
        """Load crosswalks."""
        df = pd.read_parquet(crosswalk_file)

        for idx, row in df.iterrows():
            crosswalk = row["crosswalk"]

            if "location" in crosswalk:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in crosswalk["location"]],
                    dtype=np.float32,
                )
                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.crosswalks.append(
                        Crosswalk(
                            element_id=f"crosswalk_{idx}",
                            vertices=vertices,
                        )
                    )

    def _load_poles(self, scene_data: SceneData, pole_file: Path) -> None:
        """Load poles."""
        df = pd.read_parquet(pole_file)

        for idx, row in df.iterrows():
            pole = row["pole"]

            if "location" in pole:
                loc = pole["location"]
                if len(loc) >= 2:
                    # Use provided points
                    points = np.array(
                        [[pt["x"], pt["y"], pt["z"]] for pt in loc],
                        dtype=np.float32,
                    )
                    if np.isnan(points).any():
                        continue
                elif len(loc) == 1:
                    # Create vertical pole from single point
                    base = np.array([loc[0]["x"], loc[0]["y"], loc[0]["z"]], dtype=np.float32)
                    base_rdf = convert_points_flu_to_rdf(base.reshape(1, 3))[0]
                    scene_data.poles.append(Pole.from_base_point(f"pole_{idx}", base_rdf, height=3.0))
                    continue
                else:
                    continue

                points = convert_points_flu_to_rdf(points)
                scene_data.poles.append(Pole(element_id=f"pole_{idx}", points=points))

    def _load_road_markings(self, scene_data: SceneData, road_marking_file: Path) -> None:
        """Load road markings."""
        df = pd.read_parquet(road_marking_file)

        for idx, row in df.iterrows():
            marking = cast(Dict[str, Any], row["road_marking"])

            if "location" in marking:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in marking["location"]],
                    dtype=np.float32,
                )
                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.road_markings.append(
                        RoadMarking(
                            element_id=f"road_marking_{idx}",
                            vertices=vertices,
                            marking_type=marking.get("type", "unknown"),
                        )
                    )

    def _load_wait_lines(self, scene_data: SceneData, wait_line_file: Path) -> None:
        """Load wait lines."""
        df = pd.read_parquet(wait_line_file)

        for idx, row in df.iterrows():
            wait_line = cast(Dict[str, Any], row["wait_line"])

            if "location" in wait_line:
                points = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in wait_line["location"]],
                    dtype=np.float32,
                )
                if len(points) >= 2 and not np.isnan(points).any():
                    points = convert_points_flu_to_rdf(points)
                    scene_data.wait_lines.append(
                        WaitLine(
                            element_id=f"wait_line_{idx}",
                            points=points,
                        )
                    )

    def _load_traffic_lights(self, scene_data: SceneData, traffic_light_file: Path) -> None:
        """Load traffic lights."""
        df = pd.read_parquet(traffic_light_file)

        for idx, row in df.iterrows():
            light = cast(Dict[str, Any], row["traffic_light"])

            center = np.array(
                [light["center"]["x"], light["center"]["y"], light["center"]["z"]],
                dtype=np.float32,
            )

            dimensions = np.array([0.6, 0.6, 1.0], dtype=np.float32)  # Default
            if "dimensions" in light:
                dims = light["dimensions"]
                if all(dims[k] is not None for k in ["x", "y", "z"]):
                    dimensions = np.array([dims["x"], dims["y"], dims["z"]], dtype=np.float32)

            orientation = np.array(
                [
                    light["orientation"]["x"],
                    light["orientation"]["y"],
                    light["orientation"]["z"],
                    light["orientation"]["w"],
                ],
                dtype=np.float32,
            )

            if np.isnan(center).any() or np.isnan(dimensions).any() or np.isnan(orientation).any():
                continue

            center = convert_points_flu_to_rdf(center.reshape(1, 3))[0]
            orientation = convert_quaternions_flu_to_rdf(orientation.reshape(1, 4))[0]

            scene_data.traffic_lights.append(
                TrafficLight(
                    element_id=f"traffic_light_{idx}",
                    center=center,
                    dimensions=dimensions,
                    orientation=orientation,
                )
            )

    def _load_traffic_signs(self, scene_data: SceneData, traffic_sign_file: Path) -> None:
        """Load traffic signs."""
        df = pd.read_parquet(traffic_sign_file)

        for idx, row in df.iterrows():
            sign = cast(Dict[str, Any], row["traffic_sign"])

            center = np.array(
                [sign["center"]["x"], sign["center"]["y"], sign["center"]["z"]],
                dtype=np.float32,
            )

            dimensions = np.array([0.8, 0.3, 0.8], dtype=np.float32)  # Default
            if "dimensions" in sign:
                dims = sign["dimensions"]
                if all(dims[k] is not None for k in ["x", "y", "z"]):
                    dimensions = np.array([dims["x"], dims["y"], dims["z"]], dtype=np.float32)

            orientation = np.array(
                [
                    sign["orientation"]["x"],
                    sign["orientation"]["y"],
                    sign["orientation"]["z"],
                    sign["orientation"]["w"],
                ],
                dtype=np.float32,
            )

            # Map sign type
            category = sign.get("category", "")
            if category is not None:
                category = category.upper()
            sign_type_map = {
                "STOP": TrafficSignType.STOP,
                "YIELD": TrafficSignType.YIELD,
                "SPEED_LIMIT": TrafficSignType.SPEED_LIMIT,
            }
            sign_type = sign_type_map.get(category, TrafficSignType.UNKNOWN)

            if np.isnan(center).any() or np.isnan(dimensions).any() or np.isnan(orientation).any():
                continue

            center = convert_points_flu_to_rdf(center.reshape(1, 3))[0]
            orientation = convert_quaternions_flu_to_rdf(orientation.reshape(1, 4))[0]

            scene_data.traffic_signs.append(
                TrafficSign(
                    element_id=f"traffic_sign_{idx}",
                    center=center,
                    dimensions=dimensions,
                    orientation=orientation,
                    sign_type=sign_type,
                )
            )

    def _load_intersection_areas(self, scene_data: SceneData, intersection_file: Path) -> None:
        """Load intersection areas."""
        df = pd.read_parquet(intersection_file)

        for idx, row in df.iterrows():
            area = cast(Dict[str, Any], row["intersection_area"])

            if "location" in area:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in area["location"]],
                    dtype=np.float32,
                )

                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.intersection_areas.append(
                        IntersectionArea(
                            element_id=f"intersection_{idx}",
                            vertices=vertices,
                        )
                    )

    def _load_road_islands(self, scene_data: SceneData, road_island_file: Path) -> None:
        """Load road islands."""
        df = pd.read_parquet(road_island_file)

        for idx, row in df.iterrows():
            island = row["road_island"]

            if "location" in island:
                vertices = np.array(
                    [[pt["x"], pt["y"], pt["z"]] for pt in island["location"]],
                    dtype=np.float32,
                )
                if len(vertices) > 2 and not np.isnan(vertices).any():
                    vertices = convert_points_flu_to_rdf(vertices)
                    scene_data.road_islands.append(
                        RoadIsland(
                            element_id=f"road_island_{idx}",
                            vertices=vertices,
                        )
                    )
