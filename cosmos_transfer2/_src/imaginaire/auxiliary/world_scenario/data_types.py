# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Typed dataclasses for internal representation of map elements and dynamic objects for autonomous driving.

This module provides a unified data model that can be populated from different
data sources (ClipGT, WebDataset, etc.) and used consistently by the rendering pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.base import CameraBase

# ============================================================================
# Enums for type safety
# ============================================================================


class ObjectType(str, Enum):
    """Dynamic object types."""

    CAR = "Car"
    PEDESTRIAN = "Pedestrian"
    CYCLIST = "Cyclist"
    TRUCK = "Truck"
    BUS = "Bus"
    MOTORCYCLE = "Motorcycle"
    OTHER = "Other"


class LaneLineStyle(str, Enum):
    """Lane line styles."""

    SOLID_SINGLE = "SOLID_SINGLE"
    DASHED_SINGLE = "DASHED_SINGLE"
    SOLID_DOUBLE = "SOLID_DOUBLE"
    DASHED_DOUBLE = "DASHED_DOUBLE"
    SOLID_DASHED = "SOLID_DASHED"
    DASHED_SOLID = "DASHED_SOLID"
    LONG_DASHED_SINGLE = "LONG_DASHED_SINGLE"
    LONG_DASHED_GROUP = "LONG_DASHED_GROUP"
    SHORT_DASHED_SINGLE = "SHORT_DASHED_SINGLE"
    DOT_DASHED_SINGLE = "DOT_DASHED_SINGLE"
    DOT_DASHED_GROUP = "DOT_DASHED_GROUP"
    SOLID_GROUP = "SOLID_GROUP"
    DOT_SOLID_SINGLE = "DOT_SOLID_SINGLE"
    DOT_SOLID_GROUP = "DOT_SOLID_GROUP"
    UNKNOWN = "UNKNOWN"


class LaneLineColor(str, Enum):
    """Lane line colors."""

    WHITE = "WHITE"
    YELLOW = "YELLOW"
    BLUE = "BLUE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LaneLineType:
    """Lane line type composed of color and pattern attributes."""

    color: LaneLineColor
    style: LaneLineStyle

    @property
    def canonical_name(self) -> str:
        """Return the canonical name for this lane line type."""
        return self.color.value + " " + self.style.value


class TrafficLightState(str, Enum):
    """Traffic light states."""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"
    OFF = "OFF"


class TrafficSignType(str, Enum):
    """Traffic sign types."""

    STOP = "STOP"
    YIELD = "YIELD"
    SPEED_LIMIT = "SPEED_LIMIT"
    NO_ENTRY = "NO_ENTRY"
    ONE_WAY = "ONE_WAY"
    PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Base classes
# ============================================================================


@dataclass
class MapElement:
    """Base class for all map elements."""

    element_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolylineElement(MapElement):
    """Base class for polyline-based map elements."""

    points: NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )  # Shape: (N, 3) for N points with x, y, z

    def __post_init__(self) -> None:
        """Validate polyline data."""
        if self.points.size > 0 and (self.points.ndim != 2 or self.points.shape[1] != 3):
            raise ValueError(f"Polyline points must have shape (N, 3), got {self.points.shape}")


@dataclass
class PolygonElement(MapElement):
    """Base class for polygon-based map elements."""

    vertices: NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )  # Shape: (N, 3) for N vertices
    is_closed: bool = True

    def __post_init__(self) -> None:
        """Validate polygon data."""
        if self.vertices.size > 0 and (self.vertices.ndim != 2 or self.vertices.shape[1] != 3):
            raise ValueError(f"Polygon vertices must have shape (N, 3), got {self.vertices.shape}")


@dataclass
class OrientedBoxElement(MapElement):
    """Base class for 3D oriented bounding box elements."""

    center: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )  # Shape: (3,) for x, y, z
    dimensions: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )  # Shape: (3,) for length, width, height
    orientation: NDArray[np.float32] = field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)
    )  # Shape: (4,) quaternion [x, y, z, w]

    def __post_init__(self) -> None:
        """Validate oriented box data."""
        if self.center.shape != (3,):
            raise ValueError(f"Center must have shape (3,), got {self.center.shape}")
        if self.dimensions.shape != (3,):
            raise ValueError(f"Dimensions must have shape (3,), got {self.dimensions.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Orientation must have shape (4,), got {self.orientation.shape}")

    @property
    def transformation_matrix(self) -> NDArray[np.float32]:
        """Get 4x4 transformation matrix from center and orientation."""
        from scipy.spatial.transform import Rotation

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = self.center
        matrix[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        return matrix


# ============================================================================
# Map Elements
# ============================================================================


@dataclass
class LaneLine(PolylineElement):
    """Lane line with color and style information."""

    lane_type: LaneLineType = field(
        default_factory=lambda: LaneLineType(color=LaneLineColor.UNKNOWN, style=LaneLineStyle.UNKNOWN)
    )
    line_width: float = 0.15  # meters

    @property
    def color(self) -> LaneLineColor:
        """Get the lane line color from the type."""
        return self.lane_type.color

    @property
    def style(self) -> LaneLineStyle:
        """Get the lane line style from the type."""
        return self.lane_type.style


@dataclass
class LaneBoundary(PolylineElement):
    """Lane boundary (left or right rail of a lane)."""

    is_left_boundary: bool = True
    lane_id: Optional[str] = None


@dataclass
class RoadBoundary(PolylineElement):
    """Road boundary/edge."""

    boundary_type: str = "curb"  # curb, barrier, etc.


@dataclass
class Crosswalk(PolygonElement):
    """Pedestrian crosswalk area."""

    pass


@dataclass
class RoadMarking(PolygonElement):
    """Road surface markings (arrows, text, etc.)."""

    marking_type: str = "unknown"


@dataclass
class WaitLine(PolylineElement):
    """Stop/wait line at intersections."""

    associated_sign_id: Optional[str] = None


@dataclass
class Pole(PolylineElement):
    """Vertical pole (traffic light pole, sign pole, etc.)."""

    height: float = 3.0  # Default height in meters

    @classmethod
    def from_base_point(cls, element_id: str, base: NDArray[np.float32], height: float = 3.0) -> "Pole":
        """Create pole from base point and height."""
        top = base.copy()
        top[2] += height
        points = np.vstack([base, top])
        return cls(element_id=element_id, points=points, height=height)


@dataclass
class TrafficLight(OrientedBoxElement):
    """Traffic light with state information."""

    states: Dict[int, TrafficLightState] = field(default_factory=dict)  # frame_id -> state

    def get_state(self, frame_id: int) -> TrafficLightState:
        """Get traffic light state for a given frame."""
        return self.states.get(frame_id, TrafficLightState.UNKNOWN)


@dataclass
class TrafficSign(OrientedBoxElement):
    """Traffic sign."""

    sign_type: TrafficSignType = TrafficSignType.UNKNOWN
    text: Optional[str] = None  # For speed limit signs, etc.


@dataclass
class IntersectionArea(PolygonElement):
    """Intersection area polygon."""

    pass


@dataclass
class RoadIsland(PolygonElement):
    """Traffic island/median area."""

    pass


@dataclass
class BufferZone(PolygonElement):
    """Buffer/safety zone."""

    zone_type: str = "general"


# ============================================================================
# Dynamic Objects
# ============================================================================


@dataclass
class DynamicObject:
    """Dynamic object with trajectory over time."""

    track_id: str
    object_type: ObjectType

    # Trajectory data - all arrays should have same length (num_frames)
    timestamps: NDArray[np.int64]  # Microseconds
    centers: NDArray[np.float32]  # Shape: (N, 3) for N frames
    dimensions: NDArray[np.float32]  # Shape: (N, 3) or (3,) if constant
    orientations: NDArray[np.float32]  # Shape: (N, 4) quaternions

    # Optional attributes
    velocities: Optional[NDArray[np.float32]] = None  # Shape: (N, 3)
    is_moving: bool = True
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Maximum time (microseconds) to extrapolate using constant velocity.
    # Beyond this, return None.
    max_extrapolation_us: float = 0.0

    def __post_init__(self) -> None:
        """Validate trajectory data."""
        n_frames = len(self.timestamps)

        if self.centers.shape[0] != n_frames or self.centers.shape[1] != 3:
            raise ValueError(f"Centers shape mismatch: expected ({n_frames}, 3), got {self.centers.shape}")

        if self.orientations.shape[0] != n_frames or self.orientations.shape[1] != 4:
            raise ValueError(f"Orientations shape mismatch: expected ({n_frames}, 4), got {self.orientations.shape}")

        # Handle constant or per-frame dimensions
        if self.dimensions.ndim == 1:
            if self.dimensions.shape[0] != 3:
                raise ValueError(f"Dimensions must be (3,) or (N, 3), got {self.dimensions.shape}")
            # Broadcast to all frames
            self.dimensions = np.tile(self.dimensions, (n_frames, 1))
        elif self.dimensions.shape != (n_frames, 3):
            raise ValueError(f"Dimensions shape mismatch: expected ({n_frames}, 3), got {self.dimensions.shape}")

    def get_pose_at_timestamp(
        self, timestamp: int
    ) -> Optional[Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]:
        """Get interpolated or extrapolated pose at a specific timestamp.

        Uses constant velocity extrapolation for timestamps outside the track range,
        up to max_extrapolation_us. Returns None if beyond the extrapolation limit.
        """
        t_start, t_end = int(self.timestamps[0]), int(self.timestamps[-1])

        # Handle timestamps before the first keyframe
        if timestamp < t_start:
            dt = t_start - timestamp  # positive value
            if dt <= self.max_extrapolation_us and len(self.timestamps) >= 2:
                return self._extrapolate_before(timestamp)
            return None

        # Handle timestamps after the last keyframe
        if timestamp > t_end:
            dt = timestamp - t_end  # positive value
            if dt <= self.max_extrapolation_us and len(self.timestamps) >= 2:
                return self._extrapolate_after(timestamp)
            return None

        # Interpolation within range
        idx = np.searchsorted(self.timestamps, timestamp)
        idx = max(idx, 1)  # when idx is 0, force it to 1 for interpolation logic below

        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

        center = (1 - alpha) * self.centers[idx - 1] + alpha * self.centers[idx]
        dims = (1 - alpha) * self.dimensions[idx - 1] + alpha * self.dimensions[idx]

        # SLERP for orientation
        times = [0, 1]
        rotations = Rotation.from_quat([self.orientations[idx - 1], self.orientations[idx]])
        slerp = Slerp(times, rotations)
        orientation = slerp([alpha])[0].as_quat()

        return center, dims, orientation.astype(np.float32)

    def _extrapolate_before(
        self, timestamp: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Extrapolate pose before the first keyframe using constant velocity."""
        # Compute velocity from first two keyframes
        dt_keyframes = float(self.timestamps[1] - self.timestamps[0])  # microseconds
        if dt_keyframes <= 0:
            return self.centers[0].copy(), self.dimensions[0].copy(), self.orientations[0].copy()

        # Linear velocity for position
        velocity = (self.centers[1] - self.centers[0]) / dt_keyframes

        # Time delta for extrapolation (negative, going backwards)
        dt_extrap = float(timestamp - self.timestamps[0])  # negative value
        center = self.centers[0] + velocity * dt_extrap

        # Angular velocity for orientation
        r0 = Rotation.from_quat(self.orientations[0])
        r1 = Rotation.from_quat(self.orientations[1])
        delta_r = r0.inv() * r1
        omega = delta_r.as_rotvec() / dt_keyframes  # radians per microsecond

        delta_extrap = Rotation.from_rotvec(omega * dt_extrap)
        orientation = (r0 * delta_extrap).as_quat()

        # Dimensions stay constant
        dims = self.dimensions[0].copy()

        return center.astype(np.float32), dims, orientation.astype(np.float32)

    def _extrapolate_after(
        self, timestamp: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Extrapolate pose after the last keyframe using constant velocity."""
        # Compute velocity from last two keyframes
        dt_keyframes = float(self.timestamps[-1] - self.timestamps[-2])  # microseconds
        if dt_keyframes <= 0:
            return self.centers[-1].copy(), self.dimensions[-1].copy(), self.orientations[-1].copy()

        # Linear velocity for position
        velocity = (self.centers[-1] - self.centers[-2]) / dt_keyframes

        # Time delta for extrapolation (positive, going forwards)
        dt_extrap = float(timestamp - self.timestamps[-1])  # positive value
        center = self.centers[-1] + velocity * dt_extrap

        # Angular velocity for orientation
        r_m1 = Rotation.from_quat(self.orientations[-2])
        r_m0 = Rotation.from_quat(self.orientations[-1])
        delta_r = r_m1.inv() * r_m0
        omega = delta_r.as_rotvec() / dt_keyframes  # radians per microsecond

        delta_extrap = Rotation.from_rotvec(omega * dt_extrap)
        orientation = (r_m0 * delta_extrap).as_quat()

        # Dimensions stay constant
        dims = self.dimensions[-1].copy()

        return center.astype(np.float32), dims, orientation.astype(np.float32)

    def get_transformation_at_frame(
        self, frame_id: int, frame_timestamps: NDArray[np.int64]
    ) -> Optional[NDArray[np.float32]]:
        """Get 4x4 transformation matrix at a specific frame."""
        if frame_id >= len(frame_timestamps):
            return None

        timestamp = frame_timestamps[frame_id]
        pose = self.get_pose_at_timestamp(timestamp)

        if pose is None:
            return None

        center, _, orientation = pose
        from scipy.spatial.transform import Rotation

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = center
        matrix[:3, :3] = Rotation.from_quat(orientation).as_matrix()
        return matrix


@dataclass
class EgoPose:
    """Ego vehicle pose."""

    timestamp: int  # Microseconds
    position: NDArray[np.float32]  # Shape: (3,)
    orientation: NDArray[np.float32]  # Shape: (4,) quaternion

    @property
    def transformation_matrix(self) -> NDArray[np.float32]:
        """Get 4x4 transformation matrix."""
        from scipy.spatial.transform import Rotation

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = self.position
        matrix[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        return matrix


# ============================================================================
# Scene Data Container
# ============================================================================


@dataclass
class SceneData:
    """Complete scene data container."""

    # Metadata
    scene_id: str
    duration_seconds: float
    frame_rate: int = 30

    # Ego motion
    ego_poses: List[EgoPose] = field(default_factory=list)

    # Camera models and extrinsics
    camera_models: Dict[str, CameraBase] = field(default_factory=dict)
    camera_extrinsics: Dict[str, NDArray[np.float32]] = field(default_factory=dict)

    # Static map elements
    lane_lines: List[LaneLine] = field(default_factory=list)
    lane_boundaries: List[LaneBoundary] = field(default_factory=list)
    road_boundaries: List[RoadBoundary] = field(default_factory=list)
    crosswalks: List[Crosswalk] = field(default_factory=list)
    road_markings: List[RoadMarking] = field(default_factory=list)
    wait_lines: List[WaitLine] = field(default_factory=list)
    poles: List[Pole] = field(default_factory=list)
    traffic_lights: List[TrafficLight] = field(default_factory=list)
    traffic_signs: List[TrafficSign] = field(default_factory=list)
    intersection_areas: List[IntersectionArea] = field(default_factory=list)
    road_islands: List[RoadIsland] = field(default_factory=list)
    buffer_zones: List[BufferZone] = field(default_factory=list)

    # Dynamic objects
    dynamic_objects: Dict[str, DynamicObject] = field(default_factory=dict)  # track_id -> object

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        """Get number of frames in the scene."""
        return len(self.ego_poses)

    @property
    def timestamps(self) -> NDArray[np.int64]:
        """Get array of timestamps."""
        return np.array([pose.timestamp for pose in self.ego_poses], dtype=np.int64)

    def get_objects_at_frame(self, frame_id: int) -> Dict[str, Dict[str, Any]]:
        """Get all dynamic objects at a specific frame in legacy format for compatibility."""
        result = {}
        timestamps = self.timestamps

        for track_id, obj in self.dynamic_objects.items():
            pose_data = obj.get_pose_at_timestamp(timestamps[frame_id])
            if pose_data is not None:
                _center, dims, _orientation = pose_data
                transform = obj.get_transformation_at_frame(frame_id, timestamps)

                if transform is not None:
                    result[track_id] = {
                        "object_to_world": transform.tolist(),
                        "object_lwh": dims.tolist(),
                        "object_type": obj.object_type.value,
                        "object_is_moving": obj.is_moving,
                    }

        return result
