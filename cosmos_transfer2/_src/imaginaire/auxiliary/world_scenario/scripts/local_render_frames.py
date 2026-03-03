# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Overlay rendering module for blending bounding box renderings with camera video frames.

This module provides functionality for:
- Creating scene data with only bounding boxes (no HD map elements)
- Filtering scene data to only include vehicles for overlay rendering
- Creating and managing overlay renderers that render only bboxes (not HD maps)
- Blending rendered bbox frames with camera video frames
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import moderngl  # pyright: ignore[reportMissingImports]
import numpy as np
import torch
from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import ObjectType, SceneData
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.tiled_multi_camera_renderer import (
    TiledMultiCameraRenderer,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local import read_video_simple
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.bbox_utils import build_cuboid_bounding_box
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.graphics_utils import (
    EDGE_INDICES,
    BoundingBox2D,
    Geometry2D,
)


def filter_scene_data_to_vehicles_only(scene_data: SceneData) -> SceneData:
    """
    Filter scene_data to only include vehicle objects (CAR, TRUCK, BUS, MOTORCYCLE).

    Excludes pedestrians, cyclists, and other non-vehicle objects from the overlay.

    Args:
        scene_data: Original scene data with all dynamic objects

    Returns:
        New SceneData instance with only vehicle objects in dynamic_objects
    """
    # Vehicle types to include in overlay
    vehicle_types = {ObjectType.CAR, ObjectType.TRUCK, ObjectType.BUS, ObjectType.MOTORCYCLE}

    # Filter dynamic_objects to only include vehicles
    filtered_dynamic_objects = {
        track_id: obj for track_id, obj in scene_data.dynamic_objects.items() if obj.object_type in vehicle_types
    }

    # Create a new SceneData instance with filtered dynamic_objects
    filtered_scene_data = replace(scene_data, dynamic_objects=filtered_dynamic_objects)

    logger.info(
        f"Filtered scene data: {len(scene_data.dynamic_objects)} -> {len(filtered_dynamic_objects)} objects (vehicles only)"
    )

    return filtered_scene_data


def create_bbox_only_scene_data(scene_data: SceneData, filter_vehicles_only: bool = True) -> SceneData:
    """
    Create a scene_data with only bounding boxes (no HD map elements).

    This function creates a minimal SceneData instance that contains only dynamic objects
    (bounding boxes) and clears all HD map elements. This allows rendering only bboxes
    without the HD map background.

    Args:
        scene_data: Original scene data with all elements
        filter_vehicles_only: If True, filter to only vehicle objects (default: True)

    Returns:
        New SceneData instance with only dynamic objects, all HD map elements cleared
    """
    # Filter to vehicles only if requested
    if filter_vehicles_only:
        filtered_dynamic_objects = {
            track_id: obj
            for track_id, obj in scene_data.dynamic_objects.items()
            if obj.object_type in {ObjectType.CAR, ObjectType.TRUCK, ObjectType.BUS, ObjectType.MOTORCYCLE}
        }
    else:
        filtered_dynamic_objects = scene_data.dynamic_objects

    # Create a new SceneData with only dynamic objects, all HD map elements empty
    bbox_only_scene_data = replace(
        scene_data,
        dynamic_objects=filtered_dynamic_objects,
        # Clear all HD map elements
        lane_lines=[],
        lane_boundaries=[],
        road_boundaries=[],
        crosswalks=[],
        road_markings=[],
        wait_lines=[],
        poles=[],
        traffic_lights=[],
        traffic_signs=[],
        intersection_areas=[],
        road_islands=[],
        buffer_zones=[],
    )

    logger.info(
        f"Created bbox-only scene data: {len(filtered_dynamic_objects)} objects, "
        f"HD map elements cleared (vehicles only: {filter_vehicles_only})"
    )

    return bbox_only_scene_data


class BboxLinesOnly2D(Geometry2D):
    """
    Custom geometry class that renders only bounding box lines (wireframe) with no face fills.

    Unlike BoundingBox2D, this class does NOT render any faces (no gray fills),
    only the colorful edge lines.
    """

    def __init__(
        self,
        xy_and_depth: np.ndarray,
        per_vertex_color: np.ndarray,
        line_width: float = 5.0,
    ) -> None:
        """
        Initialize bbox lines-only geometry.

        Args:
            xy_and_depth: np.ndarray, [8, 3] - pixel coordinates and depth of 8 vertices
            per_vertex_color: np.ndarray, [8, 3] - color for each vertex (RGB in [0, 1])
            line_width: float - line width for rendering
        """
        self.xy_and_depth = xy_and_depth
        self.per_vertex_color = per_vertex_color
        self.line_width = line_width

    def render(
        self,
        ctx: Any,
        polyline_program: Any,
        polygon_program: Any,
        **kwargs: Any,
    ) -> None:
        """
        Render only the edge lines, no faces.

        Args:
            ctx: moderngl.Context
            polyline_program: moderngl.Program for rendering lines
            polygon_program: moderngl.Program (unused, kept for compatibility)
            **kwargs: Additional arguments (must include image_width)
        """
        image_width = kwargs.get("image_width", 1920)
        polyline_program["u_line_width"].value = 2 * self.line_width / image_width  # pyright: ignore[reportAttributeAccessIssue]

        # Build line vertices and colors (only edges, no faces)
        line_vertices = []
        line_colors = []

        for i0, i1 in EDGE_INDICES:
            line_vertices.append(self.xy_and_depth[i0])
            line_vertices.append(self.xy_and_depth[i1])
            line_colors.append(self.per_vertex_color[i0])
            line_colors.append(self.per_vertex_color[i1])

        line_vertices = np.array(line_vertices, dtype="f4")
        line_colors = np.array(line_colors, dtype="f4")

        # Render lines only (no face rendering)
        vbo = ctx.buffer(line_vertices.tobytes())
        vbo_base_color = ctx.buffer(line_colors.tobytes())
        vao = ctx.vertex_array(
            polyline_program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
        )
        vao.render()


class BboxLineOnlyRenderer(TiledMultiCameraRenderer):
    """
    Custom renderer that renders bounding boxes as lines only (wireframe, no filled faces).

    Extends TiledMultiCameraRenderer to override bbox rendering to use fill_face="none"
    with random colorful lines (no overlay/fill colors).
    """

    def __init__(self, *args, **kwargs):
        """Initialize renderer and create color cache for consistent random colors per track_id."""
        super().__init__(*args, **kwargs)
        # Cache for random colors per track_id (consistent across frames)
        self.track_id_colors: Dict[str, np.ndarray] = {}

    def _get_random_color_for_track_id(self, track_id: str) -> np.ndarray:
        """
        Get a bright neon color for a track_id.

        Uses hash of track_id to deterministically select from a palette of brightest neon colors.
        Colors are in RGB format with values in [0, 1] range.
        """
        if track_id not in self.track_id_colors:
            # Brightest neon colors palette (RGB values in [0, 1])
            neon_colors = [
                [1.0, 0.0, 1.0],  # Magenta (brightest)
                [0.0, 1.0, 1.0],  # Cyan (brightest)
                [1.0, 1.0, 0.0],  # Yellow (brightest)
                [0.0, 1.0, 0.0],  # Green (brightest)
                [1.0, 0.0, 0.0],  # Red (brightest)
                [0.0, 0.0, 1.0],  # Blue (brightest)
                [1.0, 0.5, 0.0],  # Orange (bright)
                [1.0, 0.0, 0.5],  # Pink (bright)
                [0.5, 0.0, 1.0],  # Purple (bright)
                [0.0, 1.0, 0.5],  # Spring Green (bright)
                [1.0, 0.5, 1.0],  # Hot Pink (bright)
                [0.5, 1.0, 1.0],  # Light Cyan (bright)
                [1.0, 1.0, 0.5],  # Light Yellow (bright)
                [0.5, 1.0, 0.0],  # Lime (bright)
                [1.0, 0.0, 0.75],  # Rose (bright)
                [0.75, 0.0, 1.0],  # Violet (bright)
            ]

            # Use hash of track_id to deterministically select a neon color
            color_index = hash(track_id) % len(neon_colors)
            color = np.array(neon_colors[color_index], dtype=np.float32)
            self.track_id_colors[track_id] = color
        return self.track_id_colors[track_id]

    def _project_dynamic_geometry(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
        object_info: Dict,
    ) -> list[BoundingBox2D]:
        """
        Project dynamic objects to camera view as line-only bounding boxes.

        Overrides the parent method to render bboxes with fill_face="none" (wireframe only)
        using random colorful lines (no overlay/fill colors).

        Coordinate System Flow:
        - Input: Objects in FLU world coordinates (from SceneData.dynamic_objects)
        - Transform: FLU world → RDF camera coordinates (via world_to_camera = inv(camera_pose))
        - Project: RDF camera coordinates → 2D pixel coordinates (via FThetaCamera.ray2pixel_torch)

        This matches the pattern in local_project_annotations.py:
        1. Work in FLU world coordinates (SceneData standard)
        2. Convert to RDF camera coordinates only for projection
        3. Use FThetaCamera to project RDF camera coordinates to pixels

        Args:
            camera_model: FThetaCamera model (expects RDF camera coordinates for projection)
            camera_pose: (4, 4) camera-to-world transformation matrix
            object_info: Dictionary of object information with "object_to_world" (FLU) and "object_lwh"
        """
        if not object_info:
            return []

        # Pre-compute camera culling info
        cam_forward = camera_pose[:3, 2]
        cam_pos = camera_pose[:3, 3]

        # Collect all object vertices and metadata
        # Objects are in FLU world coordinates (from SceneData)
        all_vertices = []
        object_metadata = []
        track_ids = []

        for tracking_id in sorted(object_info.keys()):
            obj = object_info[tracking_id]
            obj_type = self._simplify_object_type(obj.get("object_type", "Others"))

            # object_to_world is in FLU world coordinates
            object_to_world = np.array(obj["object_to_world"])
            object_lwh = np.array(obj["object_lwh"])

            # Build vertices in FLU world coordinates
            vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)

            # Early culling
            relative_pos = vertices - cam_pos
            if np.all(np.dot(relative_pos, cam_forward) < 0):
                continue

            all_vertices.append(vertices)
            object_metadata.append(obj_type)
            track_ids.append(tracking_id)

        if not all_vertices:
            return []

        geometry_objects = []

        # Batch all dynamic objects for single GPU projection
        # all_vertices are in FLU world coordinates
        all_vertices_np = np.array(all_vertices).reshape(-1, 3)  # (N*8, 3)
        all_vertices_torch = torch.from_numpy(all_vertices_np).to(self.device, dtype=torch.float32)

        # Convert camera pose to torch
        camera_pose_torch = torch.from_numpy(camera_pose).to(self.device, dtype=torch.float32)
        # world_to_camera transforms FLU world → RDF camera coordinates
        world_to_camera_torch = torch.linalg.inv(camera_pose_torch)

        # Transform from FLU world coordinates to RDF camera coordinates
        # Then project RDF camera coordinates to 2D pixels using FThetaCamera
        all_points_cam = camera_model.transform_points_torch(all_vertices_torch, world_to_camera_torch)
        all_depth = all_points_cam[:, 2:3]
        # FThetaCamera.ray2pixel_torch expects RDF camera coordinates (Right=X, Down=Y, Forward=Z)
        all_xy = camera_model.ray2pixel_torch(all_points_cam)
        all_xy_and_depth = torch.cat([all_xy, all_depth], dim=-1)

        # Reshape back to per-object
        n_objects = len(all_vertices)
        all_xy_and_depth = all_xy_and_depth.reshape(n_objects, 8, 3)

        # Move entire buffer to CPU once (this is the only GPU->CPU transfer)
        all_xy_and_depth_cpu = all_xy_and_depth.cpu().numpy()

        # Validity check on CPU
        valid_x = (all_xy_and_depth_cpu[:, :, 0] >= 0) & (all_xy_and_depth_cpu[:, :, 0] < camera_model.width)
        valid_y = (all_xy_and_depth_cpu[:, :, 1] >= 0) & (all_xy_and_depth_cpu[:, :, 1] < camera_model.height)
        valid_depth = all_xy_and_depth_cpu[:, :, 2] > 0
        valid_mask = ~np.all(~valid_x | ~valid_y | ~valid_depth, axis=1)

        # Get valid objects and their metadata
        valid_xy_and_depth = all_xy_and_depth_cpu[valid_mask]
        valid_metadata = [object_metadata[i] for i in range(n_objects) if valid_mask[i]]
        valid_track_ids = [track_ids[i] for i in range(n_objects) if valid_mask[i]]

        # Create bounding boxes with lines only (no faces, no gray fills)
        # Use vibrant green color for all bboxes
        for xy_and_depth, track_id in zip(valid_xy_and_depth, valid_track_ids, strict=False):
            # Get vibrant green color for this bbox
            green_color = self._get_random_color_for_track_id(track_id)

            # Create per-vertex colors (all vertices same green color for uniform lines)
            per_vertex_color = np.tile(green_color, (8, 1))

            # Use custom BboxLinesOnly2D class that renders only lines, no faces
            geometry_objects.append(
                BboxLinesOnly2D(
                    xy_and_depth=xy_and_depth,
                    per_vertex_color=per_vertex_color,
                    line_width=5,  # Thick solid lines for better visibility
                )
            )

        return geometry_objects

    def _render_geometry_objects(self, geometry_objects: list) -> None:
        """
        Render geometry objects to current viewport.

        Overrides parent method to handle custom BboxLinesOnly2D class.
        """
        # Render each geometry object directly
        ctx = self.render_context.ctx
        polyline_program = self.render_context.polyline_program
        polygon_program = self.render_context.polygon_program

        # Get current image width from shader uniform (set per camera)
        image_width = (
            int(polyline_program["image_width"].value)  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(polyline_program["image_width"], "value")
            else self.cam_width
        )

        # Sort geometries by type
        from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.graphics_utils import (
            BoundingBox2D,
            LineSegment2D,
            Polygon2D,
            PolyLine2D,
            TriangleList2D,
        )

        for geom in geometry_objects:
            if isinstance(geom, BboxLinesOnly2D):
                # Handle our custom lines-only bbox class
                geom.render(ctx, polyline_program, polygon_program, image_width=image_width)
            elif isinstance(geom, (PolyLine2D, LineSegment2D)):
                geom.render(ctx, polyline_program, image_width=image_width)
            elif isinstance(geom, Polygon2D):
                geom.render(ctx, polygon_program)
            elif isinstance(geom, TriangleList2D):
                geom.render(ctx, polygon_program)
            elif isinstance(geom, BoundingBox2D):
                geom.render(ctx, polyline_program, polygon_program, image_width=image_width)


class OverlayRenderer:
    """
    Renderer for overlaying bounding box visualizations on camera video frames.

    This class manages the creation and lifecycle of a custom renderer
    configured for bbox-only line rendering (no HD maps, no filled faces),
    and provides methods for blending rendered bbox frames with camera video frames.
    """

    def __init__(
        self,
        camera_models: Dict[str, FThetaCamera],
        scene_data: SceneData,
        use_persistent_vbos: bool = True,
        multi_sample: int = 4,
        filter_vehicles_only: bool = True,
    ) -> None:
        """
        Initialize the overlay renderer.

        Args:
            camera_models: Dictionary mapping camera names to FThetaCamera models
            scene_data: Scene data containing HD map and dynamic objects
            use_persistent_vbos: Use persistent VBOs for static geometry (improves performance)
            multi_sample: MSAA samples (1 = disabled, 4 = 4x MSAA, default)
            filter_vehicles_only: If True, filter scene_data to only include vehicles (default: True)
        """
        self.camera_models = camera_models
        self.use_persistent_vbos = use_persistent_vbos
        self.multi_sample = multi_sample

        # Create bbox-only scene_data (no HD maps, only bounding boxes)
        logger.info("Creating bbox-only scene data (no HD maps, lines only)...")
        self.scene_data = create_bbox_only_scene_data(scene_data, filter_vehicles_only=filter_vehicles_only)

        # Create custom tiled multi-camera renderer that renders bboxes as lines only
        logger.info("Creating overlay renderer (bbox lines only, no filled faces)...")
        self.renderer = BboxLineOnlyRenderer(
            camera_models=camera_models,
            scene_data=self.scene_data,
            hdmap_color_version="v3",
            bbox_color_version="v3",
            enable_height_filter=False,
            use_persistent_vbos=use_persistent_vbos,
            multi_sample=multi_sample,
        )

    def render_frame(
        self,
        camera_poses: Dict[str, np.ndarray],
        frame_id: int,
    ) -> Dict[str, np.ndarray]:
        """
        Render bounding box frames for all cameras at a given frame.

        Renders only bounding boxes (no HD map elements) using the coordinates
        from the scene data.

        Coordinate System Flow (same as local_project_annotations.py):
        1. SceneData.dynamic_objects are in FLU world coordinates
        2. Renderer transforms FLU world → RDF camera coordinates (via world_to_camera)
        3. FThetaCamera projects RDF camera coordinates → 2D pixel coordinates

        Args:
            camera_poses: Dictionary mapping camera names to (4, 4) camera-to-world pose matrices
                         (FLU world coordinates)
            frame_id: Frame index for rendering

        Returns:
            Dictionary mapping camera names to rendered frames (H, W, 3) uint8 arrays
        """
        return self.renderer.render_all_cameras(camera_poses, frame_id)

    def blend_with_video(
        self,
        rendered_frame: np.ndarray,
        video_frame: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Blend a rendered bbox frame with a camera video frame.

        Base image (video_frame) is kept fully opaque (no alpha).
        Only the overlay (rendered_frame bboxes) is blended on top with alpha transparency.

        Args:
            rendered_frame: Rendered bbox frame (H, W, 3) uint8 array (no HD map, only bboxes)
            video_frame: Camera video frame (H, W, 3) uint8 array
            alpha: Alpha blending value for overlay (0.0 = no overlay, 1.0 = full bbox overlay)

        Returns:
            Blended frame (H, W, 3) uint8 array
        """
        # Base image is fully visible (no alpha)
        # Overlay is blended on top with alpha
        # Where rendered_frame has bboxes (non-black pixels), blend them
        # Where rendered_frame is black/empty, keep base image

        # Create mask for non-black pixels in rendered_frame (where bboxes are)
        rendered_float = rendered_frame.astype(np.float32)
        video_float = video_frame.astype(np.float32)

        # Mask: pixels where rendered_frame has content (sum of RGB > threshold)
        # This identifies where bboxes are drawn
        bbox_mask = np.sum(rendered_float, axis=2) > 10.0  # Threshold to detect non-black pixels
        bbox_mask = bbox_mask[:, :, np.newaxis]  # Expand to (H, W, 1)

        # Blend: base image + alpha * overlay only where overlay has content
        overlay_frame = video_float + alpha * bbox_mask * (rendered_float - video_float)

        return np.clip(overlay_frame, 0, 255).astype(np.uint8)

    def cleanup(self) -> None:
        """Clean up renderer resources."""
        if self.renderer is not None:
            self.renderer.cleanup()


def load_camera_videos(
    camera_names: list[str],
    scene_id: str,
    data_path: Path,
    camera_models: Dict[str, FThetaCamera],
) -> Dict[str, np.ndarray]:
    """
    Load camera video files into memory as numpy arrays.

    Supports multiple video path patterns:
    - ClipGT format: <scene_id>.<camera_name>.mp4
    - RDS-HQ/MADS format: ftheta_<camera_name>/<scene_id>.mp4

    Args:
        camera_names: List of camera names to load videos for
        scene_id: Scene/clip identifier
        data_path: Path to data directory containing video files
        camera_models: Dictionary mapping camera names to FThetaCamera models

    Returns:
        Dictionary mapping camera names to video arrays (N, H, W, 3) uint8 arrays
    """
    video_arrays: Dict[str, np.ndarray] = {}

    for camera_name in camera_names:
        mapped_name = camera_name.replace(":", "_")
        video_path_patterns = [
            data_path / f"{scene_id}.{mapped_name}.mp4",  # Original ClipGT format
            data_path / f"ftheta_{mapped_name}" / f"{scene_id}.mp4",  # RDS-HQ/MADS format
        ]
        video_path = None
        for path in video_path_patterns:
            if path.exists():
                video_path = path
                break

        if video_path is not None:
            camera_model = camera_models[camera_name]
            h, w = camera_model.height, camera_model.width
            video_arrays[camera_name] = read_video_simple(video_path.as_posix(), h, w)
            logger.info(f"Loaded video for {camera_name} with {len(video_arrays[camera_name])} frames")
        else:
            logger.warning(f"No video found for {camera_name}, tried: {[str(p) for p in video_path_patterns]}")

    return video_arrays


def create_overlay_renderer(
    camera_models: Dict[str, FThetaCamera],
    scene_data: SceneData,
    use_persistent_vbos: bool = True,
    multi_sample: int = 4,
    filter_vehicles_only: bool = True,
) -> OverlayRenderer:
    """
    Create an overlay renderer with the specified configuration.

    Convenience function for creating an OverlayRenderer instance.

    Args:
        camera_models: Dictionary mapping camera names to FThetaCamera models
        scene_data: Scene data containing HD map and dynamic objects
        use_persistent_vbos: Use persistent VBOs for static geometry (default: True)
        multi_sample: MSAA samples (default: 4)
        filter_vehicles_only: If True, filter scene_data to only include vehicles (default: True)

    Returns:
        OverlayRenderer instance
    """
    return OverlayRenderer(
        camera_models=camera_models,
        scene_data=scene_data,
        use_persistent_vbos=use_persistent_vbos,
        multi_sample=multi_sample,
        filter_vehicles_only=filter_vehicles_only,
    )
