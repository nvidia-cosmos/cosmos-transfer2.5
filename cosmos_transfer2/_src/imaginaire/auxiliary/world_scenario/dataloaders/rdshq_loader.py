# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""RDS-HQ data loader for MADS/DeepMap format data."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation
from webdataset import WebDataset, non_empty  # pyright: ignore[reportAttributeAccessIssue]

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import SceneDataLoader, auto_register
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import (
    Crosswalk,
    DynamicObject,
    EgoPose,
    LaneBoundary,
    LaneLine,
    LaneLineColor,
    LaneLineStyle,
    ObjectType,
    Pole,
    RoadBoundary,
    RoadMarking,
    SceneData,
    TrafficLight,
    TrafficLightState,
    TrafficSign,
    TrafficSignType,
    WaitLine,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.data_utils import (
    fix_static_objects,
    normalize_traffic_light_state_sequence,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.laneline_utils import build_lane_line_type


def get_sample(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract data from a tar file using WebDataset."""
    dataset = WebDataset(str(file_path), nodesplitter=non_empty, shardshuffle=False).decode()  # pyright: ignore[reportUndefinedVariable]
    return next(iter(dataset))


def _extract_json_blob(sample: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Return the first decodable JSON blob for the provided keys."""

    for key in keys:
        blob = sample.get(key)
        if blob is None:
            continue

        try:
            if isinstance(blob, (bytes, bytearray, str)):
                return json.loads(blob)
            if isinstance(blob, dict):
                return blob
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to decode JSON blob for key '{key}': {exc}")

        logger.debug(f"Unsupported JSON blob type for key '{key}': {type(blob)}")

    raise ValueError(f"No keys {keys} found in sample: {sample.keys()}")


@auto_register(priority=15)
class RDSHQLoader(SceneDataLoader):
    """Loader for RDS-HQ (MADS/DeepMap) format data."""

    @property
    def name(self) -> str:
        """Get loader name."""
        return "rdshq"

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported format descriptions."""
        return ["RDS-HQ MADS/DeepMap format"]

    def can_load(self, source: Union[Path, str, Dict[str, Any]]) -> bool:
        """
        Check if this loader can handle the given source.

        RDS-HQ data is identified by:
        - Directory containing attribute folders like 'pose', 'vehicle_pose', 'ftheta_intrinsic'
        - Clip ID format: {session_id}_{start_time}_{end_time}
        """
        if isinstance(source, dict):
            return False

        path = Path(source)
        if not path.is_dir():
            return False

        # Check for key RDS-HQ attribute directories
        required_attrs = ["pose", "vehicle_pose", "ftheta_intrinsic"]
        optional_attrs = ["all_object_info", "3d_lanelines", "3d_lanes", "3d_road_boundaries"]

        found_required = sum(1 for attr in required_attrs if (path / attr).exists())
        found_optional = sum(1 for attr in optional_attrs if (path / attr).exists())

        # Need at least 2 required attributes and 1 optional to be confident
        return found_required >= 2 and found_optional >= 1

    def load(
        self,
        source: Union[Path, str, Dict[str, Any]],
        clip_id: Optional[str] = None,
        camera_names: Optional[List[str]] = None,
        max_frames: int = -1,
        target_fps: int = 30,
        resize_resolution_hw: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> SceneData:
        """
        Load RDS-HQ data.

        Args:
            source: Path to RDS-HQ data directory
            clip_id: Optional clip ID to load. If None, will try to detect from available files
            camera_names: List of camera names to load
            max_frames: Maximum number of frames to load (-1 for all)
            target_fps: Target frame rate (RDS-HQ has 30 FPS camera data)
            resize_resolution_hw: Optional resize resolution
            **kwargs: Additional arguments

        Returns:
            Loaded scene data
        """
        if not isinstance(source, (str, Path)):
            raise TypeError(f"RDSHQLoader only supports string or Path sources, got {type(source)!r}")

        data_root = Path(source)
        logger.debug(f"Loading RDS-HQ data from: {data_root}")

        # Detect clip ID if not provided
        if clip_id is None:
            clip_id = self._detect_clip_id(data_root)
            logger.info(f"Detected clip ID: {clip_id}")

        # Default camera names for RDS-HQ
        if camera_names is None:
            camera_names = [
                "camera_front_wide_120fov",
                "camera_cross_left_120fov",
                "camera_cross_right_120fov",
                "camera_rear_left_70fov",
                "camera_rear_right_70fov",
                "camera_rear_tele_30fov",
            ]

        # Create scene data
        scene_data = SceneData(
            scene_id=clip_id,
            frame_rate=target_fps,
            duration_seconds=0.0,
        )

        # Load vehicle poses and camera poses
        self._load_poses(scene_data, data_root, clip_id, max_frames)

        # Load camera calibrations
        self._load_camera_calibrations(scene_data, data_root, clip_id, camera_names, resize_resolution_hw)

        # Load dynamic objects
        self._load_dynamic_objects(scene_data, data_root, clip_id)

        # Load HD map elements
        self._load_map_elements(scene_data, data_root, clip_id)

        # Update duration
        if scene_data.ego_poses:
            scene_data.duration_seconds = len(scene_data.ego_poses) / scene_data.frame_rate

        return scene_data

    def _detect_clip_id(self, data_root: Path) -> str:
        """Detect clip ID from available files."""
        # Try to find a clip ID from pose directory
        pose_dir = data_root / "pose"
        if pose_dir.exists():
            tar_files = list(pose_dir.glob("*.tar"))
            if tar_files:
                # Extract clip ID from filename
                return tar_files[0].stem

        # Try vehicle_pose directory
        vehicle_pose_dir = data_root / "vehicle_pose"
        if vehicle_pose_dir.exists():
            tar_files = list(vehicle_pose_dir.glob("*.tar"))
            if tar_files:
                return tar_files[0].stem

        raise ValueError(f"Could not detect clip ID from {data_root}")

    def _load_poses(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
        max_frames: int,
    ) -> None:
        """Load vehicle poses (ego poses)."""
        vehicle_pose_file = data_root / "vehicle_pose" / f"{clip_id}.tar"
        if not vehicle_pose_file.exists():
            logger.warning(f"Vehicle pose file not found: {vehicle_pose_file}")
            return

        data = get_sample(vehicle_pose_file)

        # Get all vehicle pose keys and sort by frame index
        pose_keys = sorted([k for k in data.keys() if k.endswith(".vehicle_pose.npy")])

        if max_frames > 0:
            pose_keys = pose_keys[:max_frames]

        for key in pose_keys:
            frame_idx = int(key.split(".")[0])
            pose_matrix = data[key]  # 4x4 transformation matrix

            # Extract translation and rotation
            translation = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat().astype(np.float32)

            ego_pose = EgoPose(
                timestamp=round(frame_idx * 1_000_000 / scene_data.frame_rate),
                position=translation,
                orientation=rotation_quat,
            )
            scene_data.ego_poses.append(ego_pose)

        logger.debug(f"Loaded {len(scene_data.ego_poses)} ego poses")
        scene_data.metadata["coordinate_frame"] = "flu"

    def _load_camera_calibrations(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
        camera_names: List[str],
        resize_hw: Optional[Tuple[int, int]],
    ) -> None:
        """Load camera calibrations."""
        # Load FTheta intrinsics
        intrinsic_file = data_root / "ftheta_intrinsic" / f"{clip_id}.tar"
        if not intrinsic_file.exists():
            logger.warning(f"Intrinsic file not found: {intrinsic_file}")
            return

        intrinsic_data = get_sample(intrinsic_file)

        # Load camera poses for extrinsics
        pose_file = data_root / "pose" / f"{clip_id}.tar"
        pose_data = get_sample(pose_file) if pose_file.exists() else {}

        # Load vehicle poses to compute camera_to_vehicle consistently
        vehicle_pose_file = data_root / "vehicle_pose" / f"{clip_id}.tar"
        vehicle_pose_data = get_sample(vehicle_pose_file) if vehicle_pose_file.exists() else {}

        for camera_name in camera_names:
            # Get intrinsic parameters
            intrinsic_key = f"ftheta_intrinsic.{camera_name}.npy"
            if intrinsic_key not in intrinsic_data:
                logger.warning(f"Intrinsic not found for {camera_name}")
                continue

            params = intrinsic_data[intrinsic_key]

            # Parse intrinsic parameters
            # Format: [cx, cy, w, h, *poly (6 params), is_bw_poly, *linear_cde (3 params, optional)]
            cx, cy, width, height = params[0], params[1], int(params[2]), int(params[3])
            poly_coeffs = params[4:10]
            is_bw_poly = bool(params[10])

            # Linear CDE parameters (default to [1, 0, 0] if not present)
            if len(params) > 11:
                linear_cde = params[11:14]
            else:
                linear_cde = np.array([1.0, 0.0, 0.0])

            # Derive camera_to_vehicle using matched-frame camera and vehicle poses
            camera_to_vehicle = np.eye(4, dtype=np.float32)
            # Find the earliest available camera pose key for this camera
            cam_pose_keys = (
                [k for k in pose_data.keys() if k.endswith(f".pose.{camera_name}.npy")]
                if isinstance(pose_data, dict)
                else []
            )
            if cam_pose_keys:
                cam_pose_keys.sort()
                first_cam_key = cam_pose_keys[0]
                frame_token = first_cam_key.split(".")[0]
                cam_to_world = pose_data[first_cam_key]
                veh_key = f"{frame_token}.vehicle_pose.npy"
                if isinstance(vehicle_pose_data, dict) and veh_key in vehicle_pose_data:
                    ego_to_world = vehicle_pose_data[veh_key]
                    vehicle_to_world_inv = np.linalg.inv(ego_to_world)
                    camera_to_vehicle = (vehicle_to_world_inv @ cam_to_world).astype(np.float32)
                else:
                    logger.warning(
                        f"Vehicle pose for frame {frame_token} not found when deriving extrinsics for {camera_name}; using identity"
                    )
            else:
                logger.warning(f"No camera pose entries found for {camera_name}; using identity extrinsics")

            camera_model = FThetaCamera(
                cx=float(cx),
                cy=float(cy),
                width=width,
                height=height,
                poly=np.array(poly_coeffs, dtype=np.float32),
                is_bw_poly=is_bw_poly,
                linear_cde=np.array(linear_cde, dtype=np.float32),
            )

            if resize_hw:
                resize_h, resize_w = resize_hw
                scale_h = resize_h / camera_model.height
                scale_w = resize_w / camera_model.width
                if abs(scale_h - scale_w) > 0.02:
                    logger.warning(f"Non-uniform scaling for {camera_name}: {scale_h:.3f} vs {scale_w:.3f}")
                camera_model.rescale(ratio_h=scale_h, ratio_w=scale_w)

            scene_data.camera_models[camera_name] = camera_model
            scene_data.camera_extrinsics[camera_name] = camera_to_vehicle.astype(np.float32)

        logger.debug(f"Loaded calibrations for {len(scene_data.camera_models)} cameras")

    def _load_dynamic_objects(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load dynamic objects from all_object_info."""
        object_info_file = data_root / "all_object_info" / f"{clip_id}.tar"
        if not object_info_file.exists():
            logger.debug(f"Dynamic objects file not found: {object_info_file}")
            return

        data = get_sample(object_info_file)

        # Get all object info keys and sort by frame index
        info_keys = sorted([k for k in data.keys() if k.endswith(".all_object_info.json")])

        # Track all unique object IDs and their trajectories
        objects_dict: Dict[str, Dict[str, Any]] = {}

        for key in info_keys:
            frame_idx = int(key.split(".")[0])
            frame_objects = data[key]

            for track_id, obj_info in frame_objects.items():
                if track_id not in objects_dict:
                    # Initialize tracking data for new object
                    objects_dict[track_id] = {
                        "type": obj_info.get("object_type", "Others"),
                        "is_moving": obj_info.get("object_is_moving", False),
                        "lwh_values": [],
                        "poses": [],
                        "timestamps": [],
                    }

                # Add pose to trajectory
                pose_matrix = np.array(obj_info["object_to_world"])
                objects_dict[track_id]["poses"].append(pose_matrix)
                objects_dict[track_id]["timestamps"].append(round(frame_idx * 1_000_000 / scene_data.frame_rate))

                lwh_value = obj_info.get("object_lwh")
                if lwh_value is None:
                    if objects_dict[track_id]["lwh_values"]:
                        lwh_value = objects_dict[track_id]["lwh_values"][-1]
                    else:
                        lwh_value = [2.0, 2.0, 2.0]
                objects_dict[track_id]["lwh_values"].append(lwh_value)

        # Convert to DynamicObject instances
        for track_id, obj_data in objects_dict.items():
            poses = np.array(obj_data["poses"])  # Shape: (N, 4, 4)
            timestamps = np.array(obj_data["timestamps"], dtype=np.int64)

            # Extract centers and orientations from pose matrices
            centers = poses[:, :3, 3]  # Translation part
            orientations = []
            for pose in poses:
                rotation_matrix = pose[:3, :3]
                quat = Rotation.from_matrix(rotation_matrix).as_quat()
                orientations.append(quat)
            orientations = np.array(orientations, dtype=np.float32)

            # Dimensions (per observation)
            dimensions = np.array(obj_data["lwh_values"], dtype=np.float32)
            if dimensions.ndim == 1:
                dimensions = np.repeat(dimensions.reshape(1, -1), poses.shape[0], axis=0)
            elif dimensions.shape[0] != poses.shape[0]:
                last = dimensions[-1]
                pad_count = poses.shape[0] - dimensions.shape[0]
                if pad_count > 0:
                    pad = np.repeat(last.reshape(1, -1), pad_count, axis=0)
                    dimensions = np.concatenate([dimensions, pad], axis=0)

            obj_type = self._map_object_type(str(obj_data["type"]))

            dynamic_obj = DynamicObject(
                track_id=track_id,
                object_type=obj_type,
                timestamps=timestamps,
                centers=centers.astype(np.float32),
                dimensions=dimensions,
                orientations=orientations,
                is_moving=bool(obj_data["is_moving"]),
                metadata={"original_type": obj_data["type"]},
                max_extrapolation_us=0.0,
            )

            scene_data.dynamic_objects[track_id] = dynamic_obj

        # Fix static objects by averaging pose and dimensions
        fix_static_objects(scene_data.dynamic_objects)
        logger.debug(f"Loaded {len(scene_data.dynamic_objects)} dynamic object tracks")

    def _map_object_type(self, type_str: str) -> ObjectType:
        """Map RDS-HQ object type to our ObjectType enum."""
        type_mapping = {
            "Automobile": ObjectType.CAR,
            "Other_vehicle": ObjectType.CAR,
            "Vehicle": ObjectType.CAR,
            "Car": ObjectType.CAR,
            "Pedestrian": ObjectType.PEDESTRIAN,
            "Person": ObjectType.PEDESTRIAN,
            "Bicycle": ObjectType.CYCLIST,
            "Cyclist": ObjectType.CYCLIST,
            "Motorcycle": ObjectType.CYCLIST,
            "Rider": ObjectType.CYCLIST,
            "Bus": ObjectType.TRUCK,
            "Truck": ObjectType.TRUCK,
            "Heavy_truck": ObjectType.TRUCK,
            "Train_or_tram_car": ObjectType.TRUCK,
            "Trolley_bus": ObjectType.TRUCK,
            "Trailer": ObjectType.TRUCK,
        }
        return type_mapping.get(type_str, ObjectType.OTHER)

    def _load_map_elements(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load HD map elements."""
        # Load lane lines
        self._load_lane_lines(scene_data, data_root, clip_id)

        # Load lanes (boundaries)
        self._load_lanes(scene_data, data_root, clip_id)

        # Load road boundaries
        self._load_road_boundaries(scene_data, data_root, clip_id)

        # Load crosswalks
        self._load_crosswalks(scene_data, data_root, clip_id)

        # Load traffic lights
        self._load_traffic_lights(scene_data, data_root, clip_id)

        # Load traffic signs
        self._load_traffic_signs(scene_data, data_root, clip_id)

        # Load poles
        self._load_poles(scene_data, data_root, clip_id)

        # Load wait lines
        self._load_wait_lines(scene_data, data_root, clip_id)

        # Load road markings
        self._load_road_markings(scene_data, data_root, clip_id)

    def _load_lane_lines(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load lane lines (center lines)."""
        lanelines_file = data_root / "3d_lanelines" / f"{clip_id}.tar"
        if not lanelines_file.exists():
            logger.info(f"Lane lines file not found: {lanelines_file}")
            return

        data = get_sample(lanelines_file)
        lanelines_data = data.get("lanelines.json", {})
        labels = lanelines_data.get("labels", [])

        added = 0
        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("polyline3d", {}).get("vertices", [])
            if not vertices:
                continue

            # Handle different vertex formats
            if isinstance(vertices[0], dict):
                points = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
            elif isinstance(vertices[0], (list, tuple)):
                points = np.array(vertices)
            else:
                raise ValueError(f"Unexpected vertex format: {type(vertices[0])}. Expected dict or list/tuple.")

            # Extract attributes from label data
            attributes = label_data.get("shape3d", {}).get("attributes", [])

            # Default values
            color = LaneLineColor.WHITE
            style = LaneLineStyle.SOLID_SINGLE

            # Parse attributes list to find colors and styles
            colors_list = []
            styles_list = []
            for attr in attributes:
                attr_name = attr.get("name")
                if attr_name == "colors":
                    colors_list = attr.get("enumsList", {}).get("enumsList", [])
                elif attr_name == "styles":
                    styles_list = attr.get("enumsList", {}).get("enumsList", [])

            # Process colors and styles lists
            if colors_list and styles_list:
                # Find most common combination
                combinations = list(zip(colors_list, styles_list, strict=False))
                if combinations:
                    most_common = Counter(combinations).most_common(1)[0][0]
                    color_str, style_str = most_common

                    # Parse color and style
                    if color_str:
                        try:
                            color = LaneLineColor[color_str.upper()]
                        except KeyError:
                            logger.debug(f"Unknown lane line color: {color_str}, using UNKNOWN")
                            color = LaneLineColor.UNKNOWN
                    if style_str:
                        try:
                            style = LaneLineStyle[style_str.upper()]
                        except KeyError:
                            logger.debug(f"Unknown lane line style: {style_str}, using UNKNOWN")
                            style = LaneLineStyle.UNKNOWN

            # Build lane line type using the configuration
            lane_line = LaneLine(
                element_id=f"laneline_{len(scene_data.lane_lines)}",
                points=points,
                lane_type=build_lane_line_type(
                    color=color,
                    style=style,
                ),
            )
            scene_data.lane_lines.append(lane_line)
            added += 1

        logger.debug(f"Lane lines parsed: labels={len(labels)}, added={added}, total={len(scene_data.lane_lines)}")

    def _load_lanes(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load lanes (left and right boundaries)."""
        lanes_file = data_root / "3d_lanes" / f"{clip_id}.tar"
        if not lanes_file.exists():
            return

        data = get_sample(lanes_file)
        lanes_data = data.get("lanes.json", {})
        labels = lanes_data.get("labels", [])

        for label in labels:
            label_data = label.get("labelData", {})
            if not label_data:
                continue

            polylines = label_data.get("shape3d", {}).get("polylines3d", {}).get("polylines", [])

            # Each lane has left and right boundaries
            for polyline in polylines:
                vertices = polyline.get("vertices", [])
                if not vertices:
                    continue

                # Handle different vertex formats
                if isinstance(vertices[0], dict):
                    points = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
                elif isinstance(vertices[0], (list, tuple)):
                    points = np.array(vertices)
                else:
                    raise ValueError(
                        f"Unexpected lane vertex format: {type(vertices[0])}. Expected dict or list/tuple."
                    )

                boundary = LaneBoundary(
                    element_id=f"lane_boundary_{len(scene_data.lane_boundaries)}",
                    points=points,
                )
                scene_data.lane_boundaries.append(boundary)

        logger.debug(f"Loaded {len(scene_data.lane_boundaries)} lane boundaries")

    def _load_road_boundaries(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load road boundaries."""
        boundaries_file = data_root / "3d_road_boundaries" / f"{clip_id}.tar"
        if not boundaries_file.exists():
            return

        data = get_sample(boundaries_file)
        boundaries_data = data.get("road_boundaries.json", {})
        labels = boundaries_data.get("labels", [])

        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("polyline3d", {}).get("vertices", [])
            if not vertices:
                continue

            # Handle different vertex formats
            if isinstance(vertices[0], dict):
                points = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
            elif isinstance(vertices[0], (list, tuple)):
                points = np.array(vertices)
            else:
                raise ValueError(f"Unexpected vertex format: {type(vertices[0])}. Expected dict or list/tuple.")

            boundary = RoadBoundary(
                element_id=f"road_boundary_{len(scene_data.road_boundaries)}",
                points=points,
            )
            scene_data.road_boundaries.append(boundary)

        logger.debug(f"Loaded {len(scene_data.road_boundaries)} road boundaries")

    def _load_crosswalks(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load crosswalks."""
        crosswalks_file = data_root / "3d_crosswalks" / f"{clip_id}.tar"
        if not crosswalks_file.exists():
            return

        data = get_sample(crosswalks_file)
        crosswalks_data = data.get("crosswalks.json", {})
        labels = crosswalks_data.get("labels", [])

        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("surface", {}).get("vertices", [])
            if not vertices:
                continue
            # Normalize vertex format to numpy array (N, 3)
            if isinstance(vertices[0], dict):
                vertices = np.array([[v["x"], v["y"], v["z"]] for v in vertices], dtype=np.float32)
            elif isinstance(vertices[0], (list, tuple, np.ndarray)):
                vertices = np.array(vertices, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected surface vertex format: {type(vertices[0])}. Expected dict or list/tuple.")
            crosswalk = Crosswalk(
                element_id=f"crosswalk_{len(scene_data.crosswalks)}",
                vertices=vertices,
            )
            scene_data.crosswalks.append(crosswalk)

        logger.debug(f"Loaded {len(scene_data.crosswalks)} crosswalks")

    def _load_traffic_lights(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load traffic lights."""
        lights_file = data_root / "3d_traffic_lights" / f"{clip_id}.tar"
        if not lights_file.exists():
            return

        data = get_sample(lights_file)
        lights_data = _extract_json_blob(data, "3d_traffic_lights.json", "traffic_lights.json")
        if not lights_data:
            return
        labels = lights_data.get("labels", [])

        status_mapping: Dict[str, List[str]] = {}
        status_file = data_root / "3d_traffic_lights_status" / f"{clip_id}.tar"
        if status_file.exists():
            status_sample = get_sample(status_file)
            status_data = _extract_json_blob(status_sample, "aggregated_states.json")
            traffic_states = status_data.get("traffic_light_states", {}) if status_data else {}
            for info in traffic_states.values():
                feature_id = str(info.get("feature_id")) if info.get("feature_id") is not None else None
                state_sequence = info.get("state", [])
                if feature_id is None:
                    continue
                status_mapping[feature_id] = state_sequence

        added = 0
        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("cuboid3d", {}).get("vertices", [])
            if not vertices or len(vertices) != 8:
                continue

            # Convert 8 corners to center, dimensions, and orientation
            if isinstance(vertices[0], dict):
                corners = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
            elif isinstance(vertices[0], (list, tuple)):
                corners = np.array(vertices)
            else:
                raise ValueError(f"Unexpected cuboid vertex format: {type(vertices[0])}. Expected dict or list/tuple.")
            center = corners.mean(axis=0)

            # Estimate dimensions from corners
            # Based on vertex ordering from build_cuboid_bounding_box:
            # 0-3 is along X axis, 0-1 is along Y axis, 0-4 is along Z axis
            dimensions = np.array(
                [
                    np.linalg.norm(corners[3] - corners[0]),  # length (X dimension)
                    np.linalg.norm(corners[1] - corners[0]),  # width (Y dimension)
                    np.linalg.norm(corners[4] - corners[0]),  # height (Z dimension)
                ]
            )

            # Compute orientation from cuboid vertices
            # Extract local coordinate axes from the cuboid
            # Note: Need to ensure right-handed coordinate system

            # Get primary axes from vertices
            x_vec = corners[3] - corners[0]  # Along X dimension
            y_vec = corners[1] - corners[0]  # Along Y dimension
            z_vec = corners[4] - corners[0]  # Along Z dimension

            # Normalize to get unit vectors
            x_axis = x_vec / np.linalg.norm(x_vec)
            y_axis = y_vec / np.linalg.norm(y_vec)
            z_axis = z_vec / np.linalg.norm(z_vec)

            # Ensure orthogonality and right-handedness
            # Re-orthogonalize using Gram-Schmidt
            y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
            y_axis = y_axis / np.linalg.norm(y_axis)

            # Ensure z-axis forms right-handed system
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)

            # Build rotation matrix and convert to quaternion
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

            # Check determinant to ensure right-handedness
            det = np.linalg.det(rotation_matrix)
            if det < 0:
                # If left-handed, flip one axis
                z_axis = -z_axis
                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

            orientation = Rotation.from_matrix(rotation_matrix).as_quat().astype(np.float32)

            # Determine feature ID for status lookup
            feature_id: Optional[str] = None
            attributes = label_data.get("shape3d", {}).get("attributes", [])
            for attr in attributes:
                if attr.get("name") == "feature_id" and attr.get("text") is not None:
                    feature_id = str(attr.get("text"))
                    break

            state_seq = status_mapping.get(feature_id, [])  # pyright: ignore[reportCallIssue,reportArgumentType]
            num_frames = max(1, scene_data.num_frames)
            normalized_states = normalize_traffic_light_state_sequence(
                state_seq,
                num_frames,
                feature_id=feature_id,
            )

            states_dict = {
                frame_idx: state
                for frame_idx, state in enumerate(normalized_states)
                if state is not TrafficLightState.UNKNOWN
            }

            light = TrafficLight(
                element_id=f"traffic_light_{len(scene_data.traffic_lights)}",
                center=center.astype(np.float32),
                dimensions=dimensions.astype(np.float32),
                orientation=orientation,
                states=states_dict,
            )
            light.metadata["feature_id"] = feature_id
            light.metadata["state_sequence"] = normalized_states
            scene_data.traffic_lights.append(light)
            added += 1

        logger.debug(
            f"Traffic lights parsed: labels={len(labels)}, added={added}, total={len(scene_data.traffic_lights)}"
        )

    def _load_traffic_signs(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load traffic signs."""
        signs_file = data_root / "3d_traffic_signs" / f"{clip_id}.tar"
        if not signs_file.exists():
            return

        data = get_sample(signs_file)
        signs_data = _extract_json_blob(data, "3d_traffic_signs.json", "traffic_signs.json")
        if not signs_data:
            return
        labels = signs_data.get("labels", [])
        added = 0
        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("cuboid3d", {}).get("vertices", [])
            if not vertices or len(vertices) != 8:
                continue

            # Convert 8 corners to center, dimensions, and orientation
            if isinstance(vertices[0], dict):
                corners = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
            elif isinstance(vertices[0], (list, tuple)):
                corners = np.array(vertices)
            else:
                raise ValueError(f"Unexpected cuboid vertex format: {type(vertices[0])}. Expected dict or list/tuple.")
            center = corners.mean(axis=0)

            # Estimate dimensions from corners
            # Based on vertex ordering from build_cuboid_bounding_box:
            # 0-3 is along X axis, 0-1 is along Y axis, 0-4 is along Z axis
            dimensions = np.array(
                [
                    np.linalg.norm(corners[3] - corners[0]),  # length (X dimension)
                    np.linalg.norm(corners[1] - corners[0]),  # width (Y dimension)
                    np.linalg.norm(corners[4] - corners[0]),  # height (Z dimension)
                ]
            )

            # Compute orientation from cuboid vertices
            # Extract local coordinate axes from the cuboid
            # Note: Need to ensure right-handed coordinate system

            # Get primary axes from vertices
            x_vec = corners[3] - corners[0]  # Along X dimension
            y_vec = corners[1] - corners[0]  # Along Y dimension
            z_vec = corners[4] - corners[0]  # Along Z dimension

            # Normalize to get unit vectors
            x_axis = x_vec / np.linalg.norm(x_vec)
            y_axis = y_vec / np.linalg.norm(y_vec)
            z_axis = z_vec / np.linalg.norm(z_vec)

            # Ensure orthogonality and right-handedness
            # Re-orthogonalize using Gram-Schmidt
            y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
            y_axis = y_axis / np.linalg.norm(y_axis)

            # Ensure z-axis forms right-handed system
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)

            # Build rotation matrix and convert to quaternion
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

            # Check determinant to ensure right-handedness
            det = np.linalg.det(rotation_matrix)
            if det < 0:
                # If left-handed, flip one axis
                z_axis = -z_axis
                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

            orientation = Rotation.from_matrix(rotation_matrix).as_quat().astype(np.float32)

            sign = TrafficSign(
                element_id=f"traffic_sign_{len(scene_data.traffic_signs)}",
                center=center.astype(np.float32),
                dimensions=dimensions.astype(np.float32),
                orientation=orientation,
                sign_type=TrafficSignType.UNKNOWN,
            )
            scene_data.traffic_signs.append(sign)
            added += 1

        logger.debug(
            f"Traffic signs parsed: labels={len(labels)}, added={added}, total={len(scene_data.traffic_signs)}"
        )

    def _load_poles(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load poles."""
        poles_file = data_root / "3d_poles" / f"{clip_id}.tar"
        if not poles_file.exists():
            return

        data = get_sample(poles_file)
        poles_data = data.get("poles.json", {})
        labels = poles_data.get("labels", [])

        added = 0
        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("polyline3d", {}).get("vertices", [])
            if not vertices:
                continue

            # Handle different vertex formats
            if isinstance(vertices[0], dict):
                points = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
            elif isinstance(vertices[0], (list, tuple)):
                points = np.array(vertices)
            else:
                raise ValueError(f"Unexpected vertex format: {type(vertices[0])}. Expected dict or list/tuple.")

            pole = Pole(
                element_id=f"pole_{len(scene_data.poles)}",
                points=points,
            )
            scene_data.poles.append(pole)
            added += 1

        logger.debug(f"Poles parsed: labels={len(labels)}, added={added}, total={len(scene_data.poles)}")

    def _load_wait_lines(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load wait lines."""
        wait_lines_file = data_root / "3d_wait_lines" / f"{clip_id}.tar"
        if not wait_lines_file.exists():
            return

        data = get_sample(wait_lines_file)
        wait_lines_data = data.get("wait_lines.json", {})
        labels = wait_lines_data.get("labels", [])

        added = 0
        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("polyline3d", {}).get("vertices", [])
            if not vertices:
                continue

            # Handle different vertex formats
            if isinstance(vertices[0], dict):
                points = np.array([[v["x"], v["y"], v["z"]] for v in vertices])
            elif isinstance(vertices[0], (list, tuple)):
                points = np.array(vertices)
            else:
                raise ValueError(f"Unexpected vertex format: {type(vertices[0])}. Expected dict or list/tuple.")

            wait_line = WaitLine(
                element_id=f"wait_line_{len(scene_data.wait_lines)}",
                points=points,
            )
            scene_data.wait_lines.append(wait_line)
            added += 1

        logger.debug(f"Wait lines parsed: labels={len(labels)}, added={added}, total={len(scene_data.wait_lines)}")

    def _load_road_markings(
        self,
        scene_data: SceneData,
        data_root: Path,
        clip_id: str,
    ) -> None:
        """Load road markings."""
        markings_file = data_root / "3d_road_markings" / f"{clip_id}.tar"
        if not markings_file.exists():
            return

        data = get_sample(markings_file)
        markings_data = data.get("road_markings.json", {})
        labels = markings_data.get("labels", [])

        added = 0
        for label in labels:
            label_data = label.get("labelData", {})
            if label_data.get("emptyLabel", False):
                continue

            vertices = label_data.get("shape3d", {}).get("surface", {}).get("vertices", [])
            if not vertices:
                continue
            # Normalize vertex format to numpy array (N, 3)
            if isinstance(vertices[0], dict):
                vertices = np.array([[v["x"], v["y"], v["z"]] for v in vertices], dtype=np.float32)
            elif isinstance(vertices[0], (list, tuple, np.ndarray)):
                vertices = np.array(vertices, dtype=np.float32)
            else:
                raise ValueError(f"Unexpected surface vertex format: {type(vertices[0])}. Expected dict or list/tuple.")
            marking = RoadMarking(
                element_id=f"road_marking_{len(scene_data.road_markings)}",
                vertices=vertices,
            )
            scene_data.road_markings.append(marking)
            added += 1

        logger.debug(
            f"Road markings parsed: labels={len(labels)}, added={added}, total={len(scene_data.road_markings)}"
        )
