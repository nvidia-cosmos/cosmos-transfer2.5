# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Tiled multi-camera renderer that renders all cameras in a single OpenGL context.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import moderngl  # type: ignore[reportMissingImports]
import numpy as np
import torch
from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import TrafficLightState
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.data_utils import (
    coerce_traffic_light_state,
    normalize_traffic_light_state_sequence,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.bbox_utils import (
    build_cuboid_bounding_box,
    load_bbox_colors,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.graphics_utils import (
    BoundingBox2D,
    LineSegment2D,
    Polygon2D,
    RenderContext,
    TriangleList2D,
    create_context,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.laneline_utils import (
    prepare_laneline_geometry_data,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.minimap_utils import (
    cuboid3d_to_polyline,
    get_type_from_name,
    load_hdmap_colors,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.pcd_utils import (
    interpolate_polyline_to_points,
    triangulate_polygon_3d,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.traffic_light_utils import (
    create_traffic_light_status_geometry_objects_from_data,
    load_traffic_light_colors,
)

if TYPE_CHECKING:
    from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import SceneData, TrafficLight


@dataclass
class StaticMapCache:
    """Cached static map data after preprocessing."""

    polylines: Dict[str, List[np.ndarray]]  # Preprocessed polylines by type
    polygons: Dict[str, List[Tuple[str, np.ndarray]]]  # Preprocessed polygons
    triangulated: Dict[str, List[np.ndarray]]  # Pre-triangulated crosswalks etc
    colors: Dict[str, np.ndarray]  # Color mappings
    line_widths: Dict[str, float]  # Line width settings
    laneline_data: List[Dict]  # Preprocessed laneline data with patterns


@dataclass
class CameraView:
    """Configuration for a single camera view in the tiled layout."""

    name: str
    model: FThetaCamera
    tile_x: int  # X position in tile grid
    tile_y: int  # Y position in tile grid
    offset_x: int  # Pixel offset in combined image
    offset_y: int  # Pixel offset in combined image


class TiledMultiCameraRenderer:
    """
    High-performance multi-camera renderer using a single OpenGL context.

    Renders all camera views in a tiled layout within a single framebuffer,
    then extracts individual camera images. This approach:
    - Uses a single OpenGL context (no threading issues)
    - Minimizes GPU state changes
    - Enables batch rendering of all cameras
    - Supports up to 7 cameras efficiently
    """

    def __init__(
        self,
        camera_models: Dict[str, FThetaCamera],
        scene_data: "SceneData",
        hdmap_color_version: str = "v3",
        bbox_color_version: str = "v3",
        traffic_light_color_version: str = "v2",
        enable_height_filter: bool = False,
        grid_layout: Optional[Tuple[int, int]] = None,
        use_persistent_vbos: bool = False,
        windowless: bool = True,
        gpu_index: int = 0,
        render_context: Optional[RenderContext] = None,
        multi_sample: int = 4,
        fps_report_interval: float = 60.0,
    ):
        """
        Initialize the tiled multi-camera renderer.

        Args:
            camera_models: Dictionary of camera_name -> FThetaCamera model
            scene_data: Scene data containing map elements and dynamic objects
            hdmap_color_version: HD map color scheme
            bbox_color_version: Bounding box color scheme
            traffic_light_color_version: Traffic light color scheme
            enable_height_filter: Whether to filter by height
            grid_layout: Optional (rows, cols) for tile layout. Auto-computed if None.
            use_persistent_vbos: Whether to use persistent VBOs for static geometry (optimization)
            windowless: Whether to create a windowless context
            gpu_index: Index of the GPU to use
            render_context: Optional render context to use
            multi_sample: Number of samples for multisampling anti-aliasing (MSAA). Default 4.
            fps_report_interval: Interval in seconds to report FPS. Default 60.0.
        """
        self.camera_models = camera_models
        self.scene_data = scene_data
        self.hdmap_color_version = hdmap_color_version
        self.bbox_color_version = bbox_color_version
        self.traffic_light_color_version = traffic_light_color_version
        self.enable_height_filter = enable_height_filter
        self.use_gpu_projection = True
        self.use_persistent_vbos = use_persistent_vbos
        self.windowless = windowless
        self.gpu_index = gpu_index
        self.multi_sample = multi_sample
        self.fps_report_interval = fps_report_interval
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device {self.device}")

        if enable_height_filter:
            raise NotImplementedError("Height filter is not implemented yet")

        # Determine grid layout
        num_cameras = len(camera_models)
        if grid_layout is None:
            # Auto-compute optimal grid layout
            cols = math.ceil(math.sqrt(num_cameras))
            rows = math.ceil(num_cameras / cols)
            self.grid_layout = (rows, cols)
        else:
            self.grid_layout = grid_layout

        logger.debug(f"Using grid layout {self.grid_layout} for {num_cameras} cameras")

        # Log camera names for debugging
        logger.debug(f"Camera order: {list(camera_models.keys())}")

        # Get max camera dimensions
        self.cam_width = max(cam.width for cam in camera_models.values())
        self.cam_height = max(cam.height for cam in camera_models.values())

        # Calculate combined framebuffer size
        self.combined_width = self.cam_width * self.grid_layout[1]
        self.combined_height = self.cam_height * self.grid_layout[0]

        logger.debug(f"Combined framebuffer size: {self.combined_width}x{self.combined_height}")

        # Create camera views with tile positions
        self.camera_views = self._create_camera_views()

        # Create single large OpenGL context for all cameras
        # We'll update the shader uniforms per camera, so use the individual camera size here
        if render_context is None:
            self.render_context = create_context(self.gpu_index, True, self.cam_height, self.cam_width, 200, windowless)
        else:
            self.render_context = render_context

        self._fbo: moderngl.Framebuffer | None = None
        self._color_rb: moderngl.Renderbuffer | None = None
        self._depth_rb: moderngl.Renderbuffer | None = None
        self._resolve_fbo: moderngl.Framebuffer | None = None

        # Pre-process static map once (shared across all cameras)
        logger.debug("Pre-processing static map data once for all cameras...")
        self.static_cache = self.preprocess_scene_data(scene_data, hdmap_color_version)
        self.traffic_light_colors = load_traffic_light_colors(self.traffic_light_color_version)
        (
            self.traffic_light_polylines,
            self.traffic_light_status_dict,
        ) = self._prepare_traffic_light_assets()

        # Initialize bbox colors
        self._init_bbox_colors()

        # Pre-compute world-space geometry
        self._precompute_world_geometry()

        # Initialize persistent VBOs for static geometry if enabled
        self.persistent_static_vbos = {}
        if self.use_persistent_vbos:
            # New: initialize batched global VBOs per primitive type
            self._init_batched_static_vbos()
            logger.debug("Initialized batched persistent VBOs for static geometry")

        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.last_fps_report = time.time()
        self.frame_count = 0

    def _create_camera_views(self) -> Dict[str, CameraView]:
        """Create camera view configurations with tile positions."""
        views = {}

        camera_names = list(self.camera_models.keys())
        for idx, camera_name in enumerate(camera_names):
            row = idx // self.grid_layout[1]
            col = idx % self.grid_layout[1]

            views[camera_name] = CameraView(
                name=camera_name,
                model=self.camera_models[camera_name],
                tile_x=col,
                tile_y=row,
                offset_x=col * self.cam_width,
                offset_y=row * self.cam_height,
            )

            logger.debug(
                f"Camera {idx}: {camera_name} -> tile({col}, {row}), offset({col * self.cam_width}, {row * self.cam_height})"
            )

        return views

    def _init_bbox_colors(self) -> None:
        """Initialize bounding box color mappings."""
        gradient_colors = load_bbox_colors(self.bbox_color_version)
        self.bbox_vertex_colors = {}

        for obj_type, colors in gradient_colors.items():
            if isinstance(colors, list) and len(colors) == 2 and isinstance(colors[0], list):
                # Gradient color
                per_vertex = np.zeros((8, 3))
                per_vertex[[0, 1, 4, 5]] = np.array(colors[0]) / 255.0
                per_vertex[[2, 3, 6, 7]] = np.array(colors[1]) / 255.0
            else:
                # Uniform color
                per_vertex = np.tile(np.array(colors) / 255.0, (8, 1))
            self.bbox_vertex_colors[obj_type] = per_vertex

        self.edge_color = np.array([200, 200, 200]) / 255.0

    def _prepare_traffic_light_assets(self) -> Tuple[List[np.ndarray], Optional[Dict[str, Dict[str, List[str]]]]]:
        """Precompute traffic light geometry and status sequences."""

        if not self.scene_data.traffic_lights:
            return [], None

        num_frames = max(1, self.scene_data.num_frames)
        polylines: List[np.ndarray] = []
        status_dict: Dict[str, Dict[str, List[str]]] = {}

        for idx, light in enumerate(self.scene_data.traffic_lights):
            sequence = self._build_light_state_sequence(light, num_frames)

            # Build cuboid polyline for rendering
            dims = np.asarray(light.dimensions, dtype=np.float32)
            transform = light.transformation_matrix
            cuboid_vertices = build_cuboid_bounding_box(dims[0], dims[1], dims[2], transform)
            polyline = cuboid3d_to_polyline(cuboid_vertices).astype(np.float32)
            polylines.append(polyline)

            status_dict[str(idx)] = {"state": sequence}

        return polylines, status_dict

    def _build_light_state_sequence(self, light: "TrafficLight", num_frames: int) -> List[str]:
        """Construct per-frame state sequence for a traffic light."""

        feature_id = light.metadata.get("feature_id")

        metadata_sequence = light.metadata.get("state_sequence")
        if isinstance(metadata_sequence, (list, tuple)):
            normalized_states = normalize_traffic_light_state_sequence(
                metadata_sequence,
                num_frames,
                feature_id=feature_id,
            )
        else:
            normalized_states = [TrafficLightState.UNKNOWN] * num_frames
            for frame_idx, state_enum in light.states.items():
                if 0 <= frame_idx < num_frames:
                    normalized_states[frame_idx] = coerce_traffic_light_state(
                        state_enum,
                        feature_id=feature_id,
                        frame_idx=frame_idx,
                    )

        return [state.value.lower() for state in normalized_states]

    def _project_traffic_lights(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
        frame_id: int,
    ) -> List[Polygon2D]:
        """Project and colorize traffic lights for the given frame."""

        if not self.traffic_light_polylines:
            return []

        return create_traffic_light_status_geometry_objects_from_data(
            self.traffic_light_polylines,
            self.traffic_light_status_dict,
            frame_id,
            camera_pose,
            camera_model,
            self.traffic_light_colors,
        )

    def _init_persistent_vbos(self) -> None:
        """Pre-allocate persistent VBOs for all static geometry elements."""

        ctx = self.render_context.ctx

        # Pre-allocate VBOs for polylines
        for element_name, world_segments in self.world_polylines.items():
            if element_name not in self.static_cache.colors:
                continue

            # Pre-compute colors and line widths ONCE
            color = np.array(self.static_cache.colors[element_name]) / 255.0
            line_width = self.static_cache.line_widths[element_name]

            # Flatten segments to get max vertices
            flat_points = world_segments.reshape(-1, 3)
            n_vertices = len(flat_points)

            # Pre-allocate vertex color buffer (won't change)
            vertex_colors = np.tile(color, (n_vertices, 1)).astype("f4")

            # Create persistent VBO structures
            self.persistent_static_vbos[("polyline", element_name)] = {
                "world_segments": world_segments,
                "n_segments": len(world_segments),
                "color": color,
                "line_width": line_width,
                "color_vbo": ctx.buffer(vertex_colors.tobytes()),
                "position_vbo": ctx.buffer(reserve=n_vertices * 3 * 4),  # Reserve space for f4 positions
                "vao": None,  # Will be created per camera on first use
                "visibility_mask": None,  # Will be updated per frame
                "valid_segments": None,  # Cached valid segments after projection
            }

        # Pre-allocate VBOs for lanelines
        for ll_idx, laneline_info in enumerate(self.static_cache.laneline_data):
            for pattern_idx, pattern_segments in enumerate(laneline_info["pattern_segments_list"]):
                if len(pattern_segments) == 0:
                    continue

                # Flatten segments
                flat_points = pattern_segments.reshape(-1, 3)
                n_vertices = len(flat_points)

                # Pre-allocate vertex color buffer
                color = laneline_info["rgb_float"]
                vertex_colors = np.tile(color, (n_vertices, 1)).astype("f4")

                # Use unique key for each laneline and pattern
                key = ("laneline", f"ll{ll_idx}_p{pattern_idx}")
                self.persistent_static_vbos[key] = {
                    "world_segments": pattern_segments,
                    "n_segments": len(pattern_segments),
                    "color": color,
                    "line_width": laneline_info["line_width"],
                    "color_vbo": ctx.buffer(vertex_colors.tobytes()),
                    "position_vbo": ctx.buffer(reserve=n_vertices * 3 * 4),
                    "vao": None,
                    "visibility_mask": None,
                    "valid_segments": None,
                }

        # Pre-allocate VBOs for triangles
        for element_name, triangles_list in self.world_triangles.items():
            if element_name not in self.static_cache.colors:
                continue

            color = np.array(self.static_cache.colors[element_name]) / 255.0

            for idx, triangles in enumerate(triangles_list):
                if len(triangles) == 0:
                    continue

                flat_points = triangles.reshape(-1, 3)
                n_vertices = len(flat_points)
                vertex_colors = np.tile(color, (n_vertices, 1)).astype("f4")

                key = ("triangle", f"{element_name}_{idx}")
                self.persistent_static_vbos[key] = {
                    "world_triangles": triangles,
                    "n_triangles": len(triangles),
                    "color": color,
                    "color_vbo": ctx.buffer(vertex_colors.tobytes()),
                    "position_vbo": ctx.buffer(reserve=n_vertices * 3 * 4),
                    "vao": None,
                    "visibility_mask": None,
                    "valid_triangles": None,
                }

        # Pre-allocate VBOs for polygons
        for element_name, polygons in self.world_polygons.items():
            if element_name not in self.static_cache.colors:
                continue

            color = np.array(self.static_cache.colors[element_name]) / 255.0

            for idx, poly_data in enumerate(polygons):
                if len(poly_data) == 0:
                    continue

                n_vertices = len(poly_data)
                vertex_colors = np.tile(color, (n_vertices, 1)).astype("f4")

                key = ("polygon", f"{element_name}_{idx}")
                self.persistent_static_vbos[key] = {
                    "world_polygon": poly_data,
                    "n_vertices": n_vertices,
                    "color": color,
                    "color_vbo": ctx.buffer(vertex_colors.tobytes()),
                    "position_vbo": ctx.buffer(reserve=n_vertices * 3 * 4),
                    "vao": None,
                }

        logger.debug(f"Pre-allocated {len(self.persistent_static_vbos)} persistent VBOs for static geometry")

        self.edge_color = np.array([200, 200, 200]) / 255.0

    def _init_batched_static_vbos(self) -> None:
        """Create global persistent VBOs for static lines and triangles with grouped draws.

        This builds a packing plan using the existing self.static_metadata mapping that
        indexes into self.static_points_gpu. Colors are baked once into a persistent VBO.
        Per-frame, only the position VBOs are updated from the single torch projection.
        """

        # Ensure GPU buffers exist (points + metadata)
        if not hasattr(self, "static_points_gpu"):
            # Fall back to per-element persistent VBOs
            self._init_persistent_vbos()
            return

        ctx = self.render_context.ctx

        # Lines (polylines + lanelines)
        line_items = []  # list of dicts with: start, end, n_vertices, width, color
        # Triangles (pre-triangulated elements)
        tri_items = []  # list of dicts with: start, end, n_vertices, color

        # Build items from metadata (in the same order as static_points_gpu)
        for meta in getattr(self, "static_metadata", []):
            mtype = meta.get("type")
            start = meta.get("start")
            end = meta.get("end")
            if mtype == "polyline":
                name = meta.get("name")
                if name not in self.static_cache.colors:
                    continue
                color = np.array(self.static_cache.colors[name]) / 255.0
                line_width = float(self.static_cache.line_widths.get(name, 12.0))
                n_segments = int(meta.get("n_segments", 0))
                if n_segments <= 0:
                    continue
                n_vertices = n_segments * 2
                line_items.append(
                    {
                        "start": start,
                        "end": end,
                        "n_vertices": n_vertices,
                        "width": line_width,
                        "color": color,
                    }
                )
            elif mtype == "laneline":
                info = meta.get("info", {})
                color = np.array(info.get("rgb_float", [0.5, 0.5, 0.5]), dtype=np.float32)
                line_width = float(info.get("line_width", 12.0))
                n_segments = int(meta.get("n_segments", 0))
                if n_segments <= 0:
                    continue
                n_vertices = n_segments * 2
                line_items.append(
                    {
                        "start": start,
                        "end": end,
                        "n_vertices": n_vertices,
                        "width": line_width,
                        "color": color,
                    }
                )
            elif mtype == "triangle":
                name = meta.get("name")
                if name not in self.static_cache.colors:
                    continue
                color = np.array(self.static_cache.colors[name]) / 255.0
                n_triangles = int(meta.get("n_triangles", 0))
                if n_triangles <= 0:
                    continue
                n_vertices = n_triangles * 3
                tri_items.append(
                    {
                        "start": start,
                        "end": end,
                        "n_vertices": n_vertices,
                        "color": color,
                    }
                )

        # Group line items by line width to minimize uniform changes
        width_to_items = {}
        for it in line_items:
            width_to_items.setdefault(it["width"], []).append(it)

        # Create a fixed packing order: concatenate groups by width
        line_packed_order = []
        line_draw_groups = []  # list of dicts: first, count, width
        total_line_vertices = 0
        for width, items in width_to_items.items():
            group_first = total_line_vertices
            group_count = 0
            for it in items:
                line_packed_order.append(it)
                total_line_vertices += it["n_vertices"]
                group_count += it["n_vertices"]
            if group_count > 0:
                line_draw_groups.append(
                    {
                        "first": group_first,
                        "count": group_count,
                        "width": width,
                    }
                )

        # Build static color buffer for lines following the packed order
        if total_line_vertices > 0:
            line_colors = np.zeros((total_line_vertices, 3), dtype="f4")
            cursor = 0
            for it in line_packed_order:
                n = it["n_vertices"]
                color = it["color"].astype("f4")
                line_colors[cursor : cursor + n] = color
                cursor += n

            self._batched_lines = {
                "packed_order": line_packed_order,
                "draw_groups": line_draw_groups,
                "n_vertices": total_line_vertices,
                "color_vbo": ctx.buffer(line_colors.tobytes()) if total_line_vertices > 0 else None,
                "position_vbo": ctx.buffer(reserve=total_line_vertices * 3 * 4) if total_line_vertices > 0 else None,
                "vao": None,
            }
        else:
            self._batched_lines = {
                "packed_order": [],
                "draw_groups": [],
                "n_vertices": 0,
                "color_vbo": None,
                "position_vbo": None,
                "vao": None,
            }

        # Build triangles packed order (single draw)
        total_tri_vertices = 0
        tri_packed_order = []
        for it in tri_items:
            tri_packed_order.append(it)
            total_tri_vertices += it["n_vertices"]

        if total_tri_vertices > 0:
            tri_colors = np.zeros((total_tri_vertices, 3), dtype="f4")
            cursor = 0
            for it in tri_packed_order:
                n = it["n_vertices"]
                color = it["color"].astype("f4")
                tri_colors[cursor : cursor + n] = color
                cursor += n

            self._batched_triangles = {
                "packed_order": tri_packed_order,
                "n_vertices": total_tri_vertices,
                "color_vbo": ctx.buffer(tri_colors.tobytes()),
                "position_vbo": ctx.buffer(reserve=total_tri_vertices * 3 * 4),
                "vao": None,
            }
        else:
            self._batched_triangles = {
                "packed_order": [],
                "n_vertices": 0,
                "color_vbo": None,
                "position_vbo": None,
                "vao": None,
            }

        # Note: polygons are expected to be triangulated during preprocessing so they
        # can be rendered through the triangles batched path above.

    def _precompute_world_geometry(self) -> None:
        """Pre-compute world-space geometry for faster projection."""
        # Store world-space coordinates for all static elements
        self.world_polylines = {}
        self.world_polygons = {}
        self.world_triangles = {}

        # Flatten polylines into segments
        for element_name, polylines in self.static_cache.polylines.items():
            all_segments = []
            for polyline in polylines:
                if len(polyline) >= 2:
                    segments = np.stack([polyline[:-1], polyline[1:]], axis=1)
                    all_segments.append(segments)
            if all_segments:
                self.world_polylines[element_name] = np.concatenate(all_segments, axis=0)

        # Store polygon vertices
        for element_name, polygons in self.static_cache.polygons.items():
            self.world_polygons[element_name] = [poly_data for _, poly_data in polygons]

        # Store triangulated data
        for element_name, triangles_list in self.static_cache.triangulated.items():
            self.world_triangles[element_name] = triangles_list

        # Create single concatenated buffers for all static geometry
        self._create_gpu_buffers()

    def _create_gpu_buffers(self) -> None:
        """Create unified GPU buffers for all static geometry."""
        import torch

        # Collect all static points and metadata
        all_points = []
        point_metadata = []  # (start_idx, end_idx, element_type, element_name)

        current_idx = 0

        # Process polylines (as line segments - 2 points each)
        for element_name, segments in self.world_polylines.items():
            if len(segments) > 0:
                flat_points = segments.reshape(-1, 3)
                all_points.append(flat_points)
                n_points = len(flat_points)
                point_metadata.append(
                    {
                        "start": current_idx,
                        "end": current_idx + n_points,
                        "type": "polyline",
                        "name": element_name,
                        "n_segments": len(segments),
                    }
                )
                current_idx += n_points

        # Process lanelines
        for laneline_info in self.static_cache.laneline_data:
            for pattern_segments in laneline_info["pattern_segments_list"]:
                if len(pattern_segments) > 0:
                    flat_points = pattern_segments.reshape(-1, 3)
                    all_points.append(flat_points)
                    n_points = len(flat_points)
                    point_metadata.append(
                        {
                            "start": current_idx,
                            "end": current_idx + n_points,
                            "type": "laneline",
                            "info": laneline_info,
                            "n_segments": len(pattern_segments),
                        }
                    )
                    current_idx += n_points

        # Process triangles (3 points each)
        for element_name, triangles_list in self.world_triangles.items():
            for triangles in triangles_list:
                if len(triangles) > 0:
                    flat_points = triangles.reshape(-1, 3)
                    all_points.append(flat_points)
                    n_points = len(flat_points)
                    point_metadata.append(
                        {
                            "start": current_idx,
                            "end": current_idx + n_points,
                            "type": "triangle",
                            "name": element_name,
                            "n_triangles": len(triangles),
                        }
                    )
                    current_idx += n_points

        # Process polygons
        for element_name, polygons in self.world_polygons.items():
            for poly_data in polygons:
                if len(poly_data) > 0:
                    all_points.append(poly_data)
                    n_points = len(poly_data)
                    point_metadata.append(
                        {
                            "start": current_idx,
                            "end": current_idx + n_points,
                            "type": "polygon",
                            "name": element_name,
                            "n_vertices": n_points,
                        }
                    )
                    current_idx += n_points

        # Create single GPU buffer for all static geometry
        if all_points:
            self.static_points_gpu = (
                torch.from_numpy(np.concatenate(all_points, axis=0)).to(self.device, dtype=torch.float32).contiguous()
            )
            self.static_metadata = point_metadata
            logger.debug(f"Created GPU buffer with {len(self.static_points_gpu):,} static points")
        else:
            self.static_points_gpu = None
            self.static_metadata = []

    def render_all_cameras(
        self,
        camera_poses: Dict[str, np.ndarray],
        frame_id: int,
        object_info: Optional[Dict] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Render all cameras in a single pass using tiled layout.

        Args:
            camera_poses: Dictionary of camera_name -> 4x4 pose matrix
            frame_id: Frame ID to render
            object_info: Optional dictionary of dynamic objects to render.
                If None, objects are loaded from scene_data.get_objects_at_frame(frame_id).
                Format: {tracking_id: {"object_type", "object_to_world", "object_lwh"}}

        Returns:
            Dictionary of camera_name -> rendered frame (H, W, 3)
        """
        start_time = time.time()

        # Setup framebuffer for rendering
        ctx = self.render_context.ctx

        # Create framebuffer if not exists
        if self._fbo is None:
            if self.multi_sample > 1:
                # Create MSAA framebuffer for rendering
                color_rb = ctx.renderbuffer((self.combined_width, self.combined_height), samples=self.multi_sample)
                depth_rb = ctx.depth_renderbuffer(
                    (self.combined_width, self.combined_height), samples=self.multi_sample
                )
                self._fbo = ctx.framebuffer(color_attachments=[color_rb], depth_attachment=depth_rb)

                # Create resolve framebuffer for reading
                resolve_color_rb = ctx.renderbuffer((self.combined_width, self.combined_height))
                self._resolve_fbo = ctx.framebuffer(color_attachments=[resolve_color_rb])

                logger.debug(
                    f"Created MSAA framebuffer ({self.multi_sample}x): {self.combined_width}x{self.combined_height}"
                )
            else:
                # No multisampling
                color_rb = ctx.renderbuffer((self.combined_width, self.combined_height))
                depth_rb = ctx.depth_renderbuffer((self.combined_width, self.combined_height))
                self._fbo = ctx.framebuffer(color_attachments=[color_rb], depth_attachment=depth_rb)
                self._color_rb = color_rb
                self._depth_rb = depth_rb
                logger.debug(f"Created framebuffer: {self.combined_width}x{self.combined_height}")

        # Use our framebuffer
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0, 1.0)

        # Get dynamic objects for this frame (use provided object_info if available)
        if object_info is None:
            object_info = self.scene_data.get_objects_at_frame(frame_id)

        # Render each camera view to its tile position
        for camera_name, camera_pose in camera_poses.items():
            if camera_name not in self.camera_views:
                continue

            view = self.camera_views[camera_name]

            # Set viewport for this camera's tile
            # OpenGL viewport origin is bottom-left, so we need to adjust y
            viewport_y = self.combined_height - view.offset_y - view.model.height
            self.render_context.ctx.viewport = (view.offset_x, viewport_y, view.model.width, view.model.height)

            # CRITICAL: Update shader uniforms to match the current camera's resolution
            # The shaders use image_width and image_height for calculations
            if hasattr(self.render_context, "polyline_program"):
                self.render_context.polyline_program["image_width"].value = view.model.width  # pyright: ignore[reportAttributeAccessIssue]
                self.render_context.polyline_program["image_height"].value = view.model.height  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(self.render_context, "polygon_program"):
                self.render_context.polygon_program["image_width"].value = view.model.width  # pyright: ignore[reportAttributeAccessIssue]
                self.render_context.polygon_program["image_height"].value = view.model.height  # pyright: ignore[reportAttributeAccessIssue]

            # Build and render geometry for this camera
            geometry_objects = []

            # Project static geometry
            # If using persistent VBOs, this will render directly and return empty list
            static_geom = self._project_static_geometry(view.model, camera_pose)
            geometry_objects.extend(static_geom)

            # Project traffic lights with per-frame coloring
            geometry_objects.extend(self._project_traffic_lights(view.model, camera_pose, frame_id))

            # Project dynamic geometry
            geometry_objects.extend(self._project_dynamic_geometry(view.model, camera_pose, object_info))

            # Render remaining geometry objects (dynamic + non-persistent static)
            if geometry_objects:
                self._render_geometry_objects(geometry_objects)

        # Read the entire combined framebuffer once
        ctx.viewport = (0, 0, self.combined_width, self.combined_height)

        # Read from framebuffer (resolve MSAA if needed)
        if self.multi_sample > 1 and self._resolve_fbo is not None:
            # Copy from MSAA framebuffer to resolve framebuffer
            ctx.copy_framebuffer(self._resolve_fbo, self._fbo)
            data = self._resolve_fbo.read(components=3, alignment=1)
        else:
            # Read directly from framebuffer
            data = self._fbo.read(components=3, alignment=1)

        combined_frame = np.frombuffer(data, dtype=np.uint8).reshape(self.combined_height, self.combined_width, 3)

        # Extract individual camera frames from the combined image
        results = {}
        for camera_name in camera_poses.keys():
            if camera_name not in self.camera_views:
                continue

            view = self.camera_views[camera_name]

            # Calculate the correct position after flip
            # Since we need to flip vertically, we need to adjust the y position
            flipped_y = self.combined_height - view.offset_y - view.model.height

            # Extract this camera's portion from the combined frame
            camera_frame = combined_frame[
                flipped_y : flipped_y + view.model.height, view.offset_x : view.offset_x + view.model.width, :
            ].copy()

            # Flip individual frame
            camera_frame = np.flipud(camera_frame)

            results[camera_name] = camera_frame

        # Track performance
        render_time = time.time() - start_time
        self.frame_times.append(render_time)
        self.frame_count += len(results)

        # Report FPS periodically
        if time.time() - self.last_fps_report > self.fps_report_interval:
            self._report_performance()

        return results

    def _project_static_geometry_gpu(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
    ) -> List:
        """Project all static geometry in a single GPU operation with GPU-based filtering."""
        geometry_objects = []

        if self.static_points_gpu is None or len(self.static_points_gpu) == 0:
            return geometry_objects

        # Convert camera pose to GPU once
        camera_pose_torch = torch.from_numpy(camera_pose).to(self.device, dtype=torch.float32)

        # Project ALL static points in one GPU call
        all_xy_and_depth_gpu = camera_model.get_xy_and_depth_batch_torch(self.static_points_gpu, camera_pose_torch)

        # Move entire buffer to CPU once (this is the only GPU->CPU transfer)
        all_xy_and_depth = all_xy_and_depth_gpu.cpu().numpy()

        # Now reconstruct geometry objects from the projected points
        for metadata in self.static_metadata:
            start_idx = metadata["start"]
            end_idx = metadata["end"]

            # Get the slice from CPU buffer
            xy_and_depth = all_xy_and_depth[start_idx:end_idx]

            if metadata["type"] == "polyline":
                # Reshape back to segments
                n_segments = metadata["n_segments"]
                xy_and_depth = xy_and_depth.reshape(n_segments, 2, 3)

                # Filter valid segments
                valid_mask = np.all(xy_and_depth[:, :, 2] >= 0, axis=1)
                valid_segments = xy_and_depth[valid_mask]

                if len(valid_segments) > 0 and metadata["name"] in self.static_cache.colors:
                    color = np.array(self.static_cache.colors[metadata["name"]]) / 255.0
                    line_width = self.static_cache.line_widths[metadata["name"]]
                    geometry_objects.append(
                        LineSegment2D(
                            valid_segments,
                            base_color=color,
                            line_width=line_width,
                        )
                    )

            elif metadata["type"] == "laneline":
                # Reshape back to segments
                n_segments = metadata["n_segments"]
                xy_and_depth = xy_and_depth.reshape(n_segments, 2, 3)

                # Filter valid segments
                valid_mask = np.all(xy_and_depth[:, :, 2] >= 0, axis=1)
                valid_segments = xy_and_depth[valid_mask]

                if len(valid_segments) > 0:
                    laneline_info = metadata["info"]
                    geometry_objects.append(
                        LineSegment2D(
                            valid_segments,
                            base_color=laneline_info["rgb_float"],
                            line_width=laneline_info["line_width"],
                        )
                    )

            elif metadata["type"] == "triangle":
                # Reshape back to triangles
                n_triangles = metadata["n_triangles"]
                xy_and_depth = xy_and_depth.reshape(n_triangles, 3, 3)

                # Filter valid triangles (at least one vertex in front)
                invalid_mask = np.all(xy_and_depth[:, :, 2] < 0, axis=1)
                valid_mask = ~invalid_mask
                valid_triangles = xy_and_depth[valid_mask]

                if len(valid_triangles) > 0 and metadata["name"] in self.static_cache.colors:
                    color = np.array(self.static_cache.colors[metadata["name"]]) / 255.0
                    geometry_objects.append(
                        TriangleList2D(
                            valid_triangles,
                            base_color=color,
                        )
                    )

            elif metadata["type"] == "polygon":
                # Check if valid (at least one vertex in front)
                if not np.all(xy_and_depth[:, 2] < 0) and metadata["name"] in self.static_cache.colors:
                    color = np.array(self.static_cache.colors[metadata["name"]]) / 255.0
                    geometry_objects.append(Polygon2D(xy_and_depth, base_color=color))

        return geometry_objects

    def _render_static_with_persistent_vbos(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
    ) -> None:
        """Render static geometry using persistent VBOs with visibility masking."""

        ctx = self.render_context.ctx
        polyline_program = self.render_context.polyline_program
        polygon_program = self.render_context.polygon_program

        # Get current viewport dimensions for shader uniforms
        viewport = ctx.viewport
        image_width = viewport[2]

        # Count rendered elements for debugging
        rendered_counts = {"polyline": 0, "laneline": 0, "triangle": 0, "polygon": 0}

        camera_pose_torch = torch.from_numpy(camera_pose).to(self.device, dtype=torch.float32)
        world_to_camera_torch = torch.linalg.inv(camera_pose_torch)
        world_to_cam_rot = world_to_camera_torch[:3, :3].T
        world_to_cam_trans = world_to_camera_torch[:3, 3]

        # Process each persistent VBO
        for key, vbo_data in self.persistent_static_vbos.items():
            geom_type, _element_name = key

            if geom_type == "polyline" or geom_type == "laneline":
                # Project world segments to screen space
                world_segments = vbo_data["world_segments"]
                flat_points = world_segments.reshape(-1, 3)

                flat_points_torch = torch.from_numpy(flat_points).to(self.device, dtype=torch.float32)
                points_in_cam = torch.matmul(flat_points_torch, world_to_cam_rot) + world_to_cam_trans

                if torch.all(points_in_cam[:, 2] < 0):
                    continue

                if hasattr(camera_model, "get_xy_and_depth_batch_torch"):
                    xy_and_depth = camera_model.get_xy_and_depth_batch_torch(flat_points_torch, camera_pose_torch)
                    if not isinstance(xy_and_depth, np.ndarray):
                        xy_and_depth = xy_and_depth.cpu().numpy()
                else:
                    xy_and_depth = camera_model.get_xy_and_depth(flat_points, camera_pose)

                if not isinstance(xy_and_depth, np.ndarray):
                    xy_and_depth = xy_and_depth.numpy()

                # Update the position buffer
                flat_xy_and_depth = xy_and_depth.astype("f4")
                vbo_data["position_vbo"].write(flat_xy_and_depth.tobytes())

                # Create or reuse VAO
                if vbo_data["vao"] is None:
                    vbo_data["vao"] = ctx.vertex_array(
                        polyline_program,
                        [
                            (vbo_data["position_vbo"], "3f", "in_pos"),
                            (vbo_data["color_vbo"], "3f", "in_color"),
                        ],
                        mode=moderngl.LINES,
                    )

                # Set shader uniforms
                polyline_program["u_line_width"].value = 2 * vbo_data["line_width"] / image_width  # pyright: ignore[reportAttributeAccessIssue]

                # Render using persistent VBOs - GPU will clip segments automatically
                vbo_data["vao"].render()
                rendered_counts[geom_type] += 1

            elif geom_type == "triangle":
                # Similar logic for triangles
                world_triangles = vbo_data["world_triangles"]
                flat_points = world_triangles.reshape(-1, 3)

                flat_points_torch = torch.from_numpy(flat_points).to(self.device, dtype=torch.float32)
                points_in_cam = torch.matmul(flat_points_torch, world_to_cam_rot) + world_to_cam_trans

                if torch.all(points_in_cam[:, 2] < 0):
                    continue

                if hasattr(camera_model, "get_xy_and_depth_batch_torch"):
                    xy_and_depth = camera_model.get_xy_and_depth_batch_torch(flat_points_torch, camera_pose_torch)
                    if not isinstance(xy_and_depth, np.ndarray):
                        xy_and_depth = xy_and_depth.cpu().numpy()
                else:
                    xy_and_depth = camera_model.get_xy_and_depth(flat_points, camera_pose)

                if not isinstance(xy_and_depth, np.ndarray):
                    xy_and_depth = xy_and_depth.numpy()

                # Update the position buffer with ALL triangles
                flat_xy_and_depth = xy_and_depth.astype("f4")
                vbo_data["position_vbo"].write(flat_xy_and_depth.tobytes())

                # Create or reuse VAO
                if vbo_data["vao"] is None:
                    vbo_data["vao"] = ctx.vertex_array(
                        polygon_program,
                        [
                            (vbo_data["position_vbo"], "3f", "in_pos"),
                            (vbo_data["color_vbo"], "3f", "in_color"),
                        ],
                        mode=moderngl.TRIANGLES,
                    )

                # Render using persistent VBOs
                vbo_data["vao"].render()
                rendered_counts["triangle"] += 1

            elif geom_type == "polygon":
                # Handle regular polygons (traffic signs, traffic lights, road markings)
                world_polygon = vbo_data["world_polygon"]

                world_polygon_torch = torch.from_numpy(world_polygon).to(self.device, dtype=torch.float32)
                points_in_cam = torch.matmul(world_polygon_torch, world_to_cam_rot) + world_to_cam_trans

                if torch.all(points_in_cam[:, 2] < 0):
                    continue

                if hasattr(camera_model, "get_xy_and_depth_batch_torch"):
                    xy_and_depth = camera_model.get_xy_and_depth_batch_torch(world_polygon_torch, camera_pose_torch)
                    if not isinstance(xy_and_depth, np.ndarray):
                        xy_and_depth = xy_and_depth.cpu().numpy()
                else:
                    xy_and_depth = camera_model.get_xy_and_depth(world_polygon, camera_pose)

                if not isinstance(xy_and_depth, np.ndarray):
                    xy_and_depth = xy_and_depth.numpy()

                # Pre-triangulate if not done yet
                if "triangulated_indices" not in vbo_data:
                    n_vertices = len(world_polygon)
                    if n_vertices >= 3:
                        # Fan triangulation indices (works for convex polygons)
                        indices = []
                        for i in range(1, n_vertices - 1):
                            indices.extend([0, i, i + 1])
                        vbo_data["triangulated_indices"] = np.array(indices, dtype="i4")

                        # Pre-allocate triangulated color buffer
                        n_triangle_vertices = len(indices)
                        triangle_colors = np.tile(vbo_data["color"], (n_triangle_vertices, 1)).astype("f4")
                        vbo_data["triangulated_color_vbo"] = ctx.buffer(triangle_colors.tobytes())
                        vbo_data["triangulated_position_vbo"] = ctx.buffer(reserve=n_triangle_vertices * 3 * 4)

                if "triangulated_indices" in vbo_data and len(vbo_data["triangulated_indices"]) > 0:
                    # Create triangulated positions using indices
                    indices = vbo_data["triangulated_indices"]
                    triangulated_points = xy_and_depth[indices].astype("f4")  # type: ignore[reportAttributeAccessIssue]

                    # Update position buffer
                    vbo_data["triangulated_position_vbo"].write(triangulated_points.tobytes())

                    # Create or reuse VAO
                    if vbo_data["vao"] is None:
                        vbo_data["vao"] = ctx.vertex_array(
                            polygon_program,
                            [
                                (vbo_data["triangulated_position_vbo"], "3f", "in_pos"),
                                (vbo_data["triangulated_color_vbo"], "3f", "in_color"),
                            ],
                            mode=moderngl.TRIANGLES,
                        )

                    # Render using persistent VBOs
                    vbo_data["vao"].render()
                    rendered_counts["polygon"] += 1

    def _project_static_geometry(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
    ) -> List:
        """Project static world geometry to camera view."""
        # Use batched persistent VBO path if enabled
        if self.use_persistent_vbos and hasattr(self, "_batched_lines") and hasattr(self, "_batched_triangles"):
            self._render_static_batched(camera_model, camera_pose)
            return []

        if self.static_points_gpu is not None:
            return self._project_static_geometry_gpu(camera_model, camera_pose)

        return []

    def _render_static_batched(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
    ) -> None:
        """Render static geometry using global persistent VBOs with grouped draws.

        Performs a single torch projection over all static points and updates
        the position VBOs for lines and triangles. Then draws grouped lines
        (grouped by line width) and a single batch for triangles.
        """

        if self.static_points_gpu is None or len(self.static_points_gpu) == 0:
            return

        ctx = self.render_context.ctx
        polyline_program = self.render_context.polyline_program
        polygon_program = self.render_context.polygon_program

        # Current viewport width for line width normalization
        viewport = ctx.viewport
        image_width = max(1, int(viewport[2]))

        # 1) Single GPU projection for all static points
        camera_pose_torch = torch.from_numpy(camera_pose).to(self.device, dtype=torch.float32)
        all_xy_and_depth_gpu = camera_model.get_xy_and_depth_batch_torch(self.static_points_gpu, camera_pose_torch)
        all_xy_and_depth = all_xy_and_depth_gpu.cpu().numpy().astype("f4")

        # 2) Update lines VBO and render grouped draws
        if self._batched_lines and self._batched_lines["n_vertices"] > 0:
            packed = np.zeros((self._batched_lines["n_vertices"], 3), dtype="f4")
            cursor = 0
            for it in self._batched_lines["packed_order"]:
                start = it["start"]
                end = it["end"]
                segment_xyd = all_xy_and_depth[start:end]  # shape [n_vertices, 3]
                n = it["n_vertices"]
                # segment_xyd is already flattened as [n_vertices, 3] (since we stored flat points)
                packed[cursor : cursor + n] = segment_xyd
                cursor += n

            # Initialize/reuse VAO
            if self._batched_lines["vao"] is None:
                self._batched_lines["vao"] = ctx.vertex_array(
                    polyline_program,
                    [
                        (self._batched_lines["position_vbo"], "3f", "in_pos"),
                        (self._batched_lines["color_vbo"], "3f", "in_color"),
                    ],
                    mode=moderngl.LINES,
                )

            # Write positions once per frame
            assert self._batched_lines["position_vbo"] is not None
            self._batched_lines["position_vbo"].write(packed.tobytes())

            # Render each width group
            for group in self._batched_lines["draw_groups"]:
                width = float(group["width"])
                # match shader convention: u_line_width is normalized by image width and multiplied by 2
                polyline_program["u_line_width"].value = 2 * width / image_width  # pyright: ignore[reportAttributeAccessIssue]
                first = int(group["first"])  # vertex start offset
                count = int(group["count"])  # number of vertices to draw
                self._batched_lines["vao"].render(vertices=count, first=first)

        # 3) Update triangles VBO and render (single draw)
        if self._batched_triangles and self._batched_triangles["n_vertices"] > 0:
            packed_tri = np.zeros((self._batched_triangles["n_vertices"], 3), dtype="f4")
            cursor = 0
            for it in self._batched_triangles["packed_order"]:
                start = it["start"]
                end = it["end"]
                tri_xyd = all_xy_and_depth[start:end]
                n = it["n_vertices"]
                packed_tri[cursor : cursor + n] = tri_xyd
                cursor += n

            # Initialize/reuse VAO
            if self._batched_triangles["vao"] is None:
                self._batched_triangles["vao"] = ctx.vertex_array(
                    polygon_program,
                    [
                        (self._batched_triangles["position_vbo"], "3f", "in_pos"),
                        (self._batched_triangles["color_vbo"], "3f", "in_color"),
                    ],
                    mode=moderngl.TRIANGLES,
                )

            # Write positions once per frame
            assert self._batched_triangles["position_vbo"] is not None
            self._batched_triangles["position_vbo"].write(packed_tri.tobytes())

            # Single draw for all triangles
            self._batched_triangles["vao"].render(vertices=self._batched_triangles["n_vertices"])

        # Polygons should have been triangulated and covered by the batched triangles above.

    def _project_dynamic_geometry(
        self,
        camera_model: FThetaCamera,
        camera_pose: np.ndarray,
        object_info: Dict,
    ) -> List[BoundingBox2D]:
        """Project dynamic objects to camera view."""
        if not object_info:
            return []

        # Pre-compute camera culling info
        cam_forward = camera_pose[:3, 2]
        cam_pos = camera_pose[:3, 3]

        # Collect all object vertices and metadata
        all_vertices = []
        object_metadata = []

        for tracking_id in sorted(object_info.keys()):
            obj = object_info[tracking_id]
            obj_type = self._simplify_object_type(obj.get("object_type", "Others"))

            object_to_world = np.array(obj["object_to_world"])
            object_lwh = np.array(obj["object_lwh"])

            vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)

            # Early culling
            relative_pos = vertices - cam_pos
            if np.all(np.dot(relative_pos, cam_forward) < 0):
                continue

            all_vertices.append(vertices)
            object_metadata.append(obj_type)

        if not all_vertices:
            return []

        geometry_objects = []

        # Batch all dynamic objects for single GPU projection
        all_vertices_np = np.array(all_vertices).reshape(-1, 3)  # (N*8, 3)
        all_vertices_torch = torch.from_numpy(all_vertices_np).to(self.device, dtype=torch.float32)

        # Convert camera pose to torch
        camera_pose_torch = torch.from_numpy(camera_pose).to(self.device, dtype=torch.float32)
        world_to_camera_torch = torch.linalg.inv(camera_pose_torch)

        # Single GPU call for all objects
        all_points_cam = camera_model.transform_points_torch(all_vertices_torch, world_to_camera_torch)
        all_depth = all_points_cam[:, 2:3]
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

        # Create bounding boxes
        for xy_and_depth, obj_type in zip(valid_xy_and_depth, valid_metadata, strict=False):
            geometry_objects.append(
                BoundingBox2D(
                    xy_and_depth=xy_and_depth,
                    base_color_or_per_vertex_color=self.bbox_vertex_colors[obj_type],
                    fill_face="all",
                    fill_face_style="solid",
                    line_width=4,
                    edge_color=self.edge_color,
                )
            )

        return geometry_objects

    def _render_geometry_objects(self, geometry_objects: List) -> None:
        """Render geometry objects to current viewport."""
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
            if isinstance(geom, (PolyLine2D, LineSegment2D)):
                geom.render(ctx, polyline_program, image_width=image_width)
            elif isinstance(geom, Polygon2D):
                geom.render(ctx, polygon_program)
            elif isinstance(geom, TriangleList2D):
                geom.render(ctx, polygon_program)
            elif isinstance(geom, BoundingBox2D):
                geom.render(ctx, polyline_program, polygon_program, image_width=image_width)

    def _read_framebuffer(self) -> np.ndarray:
        """Read the entire combined framebuffer."""
        # Create a framebuffer to read from
        ctx = self.render_context.ctx

        # Read from the default framebuffer (screen)
        fbo = ctx.detect_framebuffer()
        data = fbo.read(components=3, alignment=1)

        image = np.frombuffer(data, dtype=np.uint8).reshape(self.combined_height, self.combined_width, 3)
        return np.flipud(image)

    def _simplify_object_type(self, object_type: str) -> str:
        """Map object type to standard categories for v3 color scheme.

        Uses exactly 5 object categories: Car, Truck, Pedestrian, Cyclist, Others.

        Should be consistent with cosmos-av-sample-toolkits/utils/bbox_utils.py:simplify_type_in_object_info
        """
        obj_type_normalized = object_type.replace("_", " ").replace("-", " ").title().replace(" ", "_")
        if obj_type_normalized in ["Bus", "Heavy_Truck", "Train_Or_Tram_Car", "Trolley_Bus", "Trailer"]:
            return "Truck"
        elif obj_type_normalized in ["Vehicle", "Automobile", "Other_Vehicle", "Car"]:
            return "Car"
        elif obj_type_normalized in ["Person", "Pedestrian"]:
            return "Pedestrian"
        elif obj_type_normalized in ["Rider", "Cyclist", "Motorcycle", "Bicycle"]:
            return "Cyclist"
        elif obj_type_normalized in ["Other", "Others"]:  # Explicitly handle both singular and plural
            return "Others"
        else:
            return "Others"

    def _report_performance(self) -> None:
        """Report rendering performance metrics."""
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0

            logger.info(
                f"Tiled multi-camera rendering performance: {fps:.1f} FPS,"
                f"avg: {avg_time * 1000:.2f}ms per batch, frames: {self.frame_count}"
            )

        self.last_fps_report = time.time()

    @classmethod
    def preprocess_scene_data(cls, scene_data: "SceneData", hdmap_color_version: str = "v3") -> StaticMapCache:
        """Class method to preprocess scene data that can be shared across instances."""
        return cls._preprocess_scene_data_impl(scene_data, hdmap_color_version)

    @classmethod
    def preprocess_static_map(cls, map_data: Dict, hdmap_color_version: str = "v3") -> StaticMapCache:
        """Class method to preprocess static map that can be shared across instances."""
        return cls._preprocess_static_map_impl(map_data, hdmap_color_version)

    @staticmethod
    def _preprocess_scene_data_impl(scene_data: "SceneData", hdmap_color_version: str) -> StaticMapCache:
        """Preprocess scene data elements for efficient rendering."""
        cache = StaticMapCache(
            polylines={},
            polygons={},
            triangulated={},
            colors=load_hdmap_colors(hdmap_color_version),
            line_widths={},
            laneline_data=[],
        )

        def add_polylines(
            cache_key: str,
            elements: Iterable,
            line_width: float,
            get_points: Callable[[Any], Optional[np.ndarray]],
            subdivide: Optional[float] = None,
        ) -> None:
            polylines: List[np.ndarray] = []
            for element in elements:
                points = get_points(element)
                if points is None:
                    continue
                arr = np.asarray(points, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                if subdivide is not None:
                    arr = interpolate_polyline_to_points(arr, segment_interval=subdivide)
                polylines.append(arr)
            if polylines:
                cache.polylines[cache_key] = polylines
                cache.line_widths[cache_key] = line_width

        def add_polygon_tris(
            cache_key: str,
            elements: Iterable,
            get_vertices: Callable[[Any], Optional[np.ndarray]],
            subdivide: Optional[float] = None,
        ) -> None:
            triangles_list: List[np.ndarray] = []
            for element in elements:
                vertices = get_vertices(element)
                if vertices is None:
                    continue
                arr = np.asarray(vertices, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] < 3:
                    continue
                if subdivide is not None:
                    arr = interpolate_polyline_to_points(arr, segment_interval=subdivide)
                tris = triangulate_polygon_3d(arr)
                if tris.size > 0:
                    triangles_list.append(tris.astype(np.float32))
            if triangles_list:
                cache.triangulated[cache_key] = triangles_list

        cuboid_face_tris = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        )

        def oriented_box_to_tris(dimensions: np.ndarray, transform: np.ndarray) -> np.ndarray:
            cube = build_cuboid_bounding_box(dimensions[0], dimensions[1], dimensions[2], transform)
            return cube[cuboid_face_tris]

        # Lane lines retain type/color information for dashed patterns
        if scene_data.lane_lines:
            lane_tuples = [
                (np.asarray(lane_line.points, dtype=np.float32), lane_line.lane_type.canonical_name)
                for lane_line in scene_data.lane_lines
                if lane_line.points.shape[0] >= 2
            ]
            if lane_tuples:
                cache.laneline_data = prepare_laneline_geometry_data(lane_tuples)

        add_polylines(
            "lanes",
            scene_data.lane_boundaries,
            line_width=12,
            get_points=lambda boundary: boundary.points,
            subdivide=0.8,
        )
        add_polylines(
            "road_boundaries",
            scene_data.road_boundaries,
            line_width=12,
            get_points=lambda boundary: boundary.points,
            subdivide=0.8,
        )
        add_polylines(
            "poles",
            scene_data.poles,
            line_width=5,
            get_points=lambda pole: pole.points,
            subdivide=0.8,
        )
        add_polylines(
            "wait_lines",
            scene_data.wait_lines,
            line_width=12,
            get_points=lambda wait: wait.points,
            subdivide=0.8,
        )

        add_polygon_tris(
            "crosswalks",
            scene_data.crosswalks,
            get_vertices=lambda crosswalk: crosswalk.vertices,
            subdivide=0.8,
        )
        add_polygon_tris(
            "road_markings",
            scene_data.road_markings,
            get_vertices=lambda marking: marking.vertices,
            subdivide=0.8,
        )
        add_polygon_tris(
            "intersection_areas",
            scene_data.intersection_areas,
            get_vertices=lambda area: area.vertices,
            subdivide=0.8,
        )
        add_polygon_tris(
            "road_islands",
            scene_data.road_islands,
            get_vertices=lambda island: island.vertices,
            subdivide=0.8,
        )

        if scene_data.traffic_signs:
            sign_tris = []
            for sign in scene_data.traffic_signs:
                dims = np.asarray(sign.dimensions, dtype=np.float32)
                if dims.size != 3:
                    continue
                sign_tris.append(oriented_box_to_tris(dims, sign.transformation_matrix))
            if sign_tris:
                cache.triangulated["traffic_signs"] = [tris.astype(np.float32) for tris in sign_tris]

        return cache

    @staticmethod
    def _preprocess_static_map_impl(map_data: Dict, hdmap_color_version: str) -> StaticMapCache:
        """Preprocess static map elements for efficient rendering."""
        cache = StaticMapCache(
            polylines={},
            polygons={},
            triangulated={},
            colors=load_hdmap_colors(hdmap_color_version),
            line_widths={},
            laneline_data=[],
        )

        for element_name, elements in map_data.items():
            if not elements:
                continue

            # Skip lanes (no visual representation needed) and elements not in color mapping
            if element_name == "lanes":
                continue
            if element_name not in cache.colors:
                logger.debug(f"Skipping {element_name} - not in color mapping")
                continue

            try:
                element_type = get_type_from_name(element_name)
            except ValueError:
                logger.debug(f"Skipping {element_name} - unknown type")
                continue

            if element_type == "polyline":
                # Special handling for lanelines with type information
                if element_name == "lanelines" and elements and isinstance(elements[0], tuple):
                    # Process lanelines with their types to create proper patterns
                    cache.laneline_data = prepare_laneline_geometry_data(elements)
                    # Don't add to regular polylines since we have special rendering
                    continue

                # Regular polyline processing for other elements
                processed = []
                for polyline in elements:
                    # Ensure polyline is numpy array
                    if not isinstance(polyline, np.ndarray):
                        polyline = np.array(polyline)  # noqa: PLW2901

                    # Subdivide for smooth rendering
                    if element_name in ["road_boundaries"]:
                        subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.8)
                    else:
                        subdivided = polyline

                    if len(subdivided) >= 2:
                        processed.append(subdivided)

                cache.polylines[element_name] = processed
                cache.line_widths[element_name] = 5 if element_name == "poles" else 12

            elif element_type == "polygon":
                # Triangulate polygonal types so they can use the batched triangles path.
                # Special handling for traffic signs/lights represented as cuboid edge polylines.
                for polygon in elements:
                    if element_name in ["traffic_signs", "traffic_lights"] and len(polygon) >= 9:
                        # Reconstruct cuboid vertices from the known polyline connectivity used upstream:
                        # [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
                        v0, v1, v2, v3 = polygon[0], polygon[1], polygon[2], polygon[3]
                        v4, v5, v6, v7 = polygon[5], polygon[6], polygon[7], polygon[8]
                        cube = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=0)

                        # Create 12 triangles covering the 6 faces of the cuboid
                        face_tris_idx = np.array(
                            [
                                [0, 1, 2],
                                [0, 2, 3],  # bottom
                                [4, 5, 6],
                                [4, 6, 7],  # top
                                [0, 1, 5],
                                [0, 5, 4],  # side 1
                                [1, 2, 6],
                                [1, 6, 5],  # side 2
                                [2, 3, 7],
                                [2, 7, 6],  # side 3
                                [3, 0, 4],
                                [3, 4, 7],  # side 4
                            ],
                            dtype=np.int32,
                        )
                        tris = cube[face_tris_idx]
                        if element_name not in cache.triangulated:
                            cache.triangulated[element_name] = []
                        cache.triangulated[element_name].append(tris)
                    else:
                        subdivided = interpolate_polyline_to_points(polygon, segment_interval=0.8)
                        triangles = triangulate_polygon_3d(subdivided)
                        if triangles.size > 0:
                            if element_name not in cache.triangulated:
                                cache.triangulated[element_name] = []
                            cache.triangulated[element_name].append(triangles)

            elif element_type == "cuboid3d":
                # Elements may arrive as polylines describing cuboid edges (length ~16).
                # Reconstruct faces and triangulate for fast batched rendering.
                for _poly in elements:
                    poly = _poly
                    if not isinstance(poly, np.ndarray):
                        poly = np.array(poly)
                    if len(poly) >= 9:
                        v0, v1, v2, v3 = poly[0], poly[1], poly[2], poly[3]
                        v4, v5, v6, v7 = poly[5], poly[6], poly[7], poly[8]
                        cube = np.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=0)

                        face_tris_idx = np.array(
                            [
                                [0, 1, 2],
                                [0, 2, 3],
                                [4, 5, 6],
                                [4, 6, 7],
                                [0, 1, 5],
                                [0, 5, 4],
                                [1, 2, 6],
                                [1, 6, 5],
                                [2, 3, 7],
                                [2, 7, 6],
                                [3, 0, 4],
                                [3, 4, 7],
                            ],
                            dtype=np.int32,
                        )
                        tris = cube[face_tris_idx]
                        if element_name not in cache.triangulated:
                            cache.triangulated[element_name] = []
                        cache.triangulated[element_name].append(tris)
                    else:
                        # Fallback: treat as polygonal loop and triangulate
                        subdivided = interpolate_polyline_to_points(poly, segment_interval=0.8)
                        triangles = triangulate_polygon_3d(subdivided)
                        if triangles.size > 0:
                            if element_name not in cache.triangulated:
                                cache.triangulated[element_name] = []
                            cache.triangulated[element_name].append(triangles)

        logger.debug(
            f"Cached {len(cache.polylines)} polyline types, "
            f"{len(cache.polygons)} polygon types, "
            f"{len(cache.triangulated)} triangulated types, "
            f"{len(cache.laneline_data)} laneline patterns"
        )
        logger.debug(f"Polyline types: {list(cache.polylines.keys())}")
        logger.debug(f"Polygon types: {list(cache.polygons.keys())}")
        logger.debug(f"Triangulated types: {list(cache.triangulated.keys())}")

        return cache

    def cleanup(self) -> None:
        """Clean up resources."""
        # Release persistent VBOs
        if self.use_persistent_vbos and self.persistent_static_vbos:
            logger.debug(f"Releasing {len(self.persistent_static_vbos)} persistent VBOs")
            for _, vbo_data in self.persistent_static_vbos.items():
                if vbo_data["position_vbo"]:
                    vbo_data["position_vbo"].release()
                if vbo_data["color_vbo"]:
                    vbo_data["color_vbo"].release()
                if vbo_data["vao"]:
                    # VAOs are automatically released with context
                    pass
            self.persistent_static_vbos.clear()

        if hasattr(self.render_context, "ctx"):
            self.render_context.ctx.finish()
