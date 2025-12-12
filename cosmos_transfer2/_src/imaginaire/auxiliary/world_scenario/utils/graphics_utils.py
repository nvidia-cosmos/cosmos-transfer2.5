# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Graphics utilities for control video rendering.

Heavily modified from cosmos-av-sample-toolkits.
"""

from typing import Any, Final, List, Tuple, Union

import attrs
import moderngl  # pyright: ignore[reportMissingImports]
import numpy as np

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.shader_utils import (
    fragment_shader_code,
    geometry_shader_code,
    vertex_shader_constant_color_code,
    vertex_shader_graident_color_code,
)

EDGE_INDICES: Final = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

FRONT_DIAGONAL_INDICES: Final = [(0, 5), (1, 4)]

BACK_DIAGONAL_INDICES: Final = [
    (2, 7),
    (3, 6),
]

ALL_DIAGONAL_INDICES: Final = [
    (0, 5),
    (1, 4),
    (2, 7),
    (3, 6),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 7),
    (3, 4),
    (1, 6),
    (2, 5),
]

ALL_FACE_INDICES: Final = [
    (0, 1, 5),
    (0, 4, 5),
    (2, 3, 7),
    (2, 6, 7),
    (0, 2, 3),
    (0, 1, 2),
    (4, 6, 7),
    (4, 5, 6),
    (0, 4, 7),
    (0, 3, 7),
    (1, 5, 6),
    (1, 2, 6),
]

FRONT_FACE_INDICES: Final = [
    (0, 1, 5),
    (5, 4, 0),
]

BACK_FACE_INDICES: Final = [
    (2, 3, 7),
    (2, 6, 7),
]


def get_remaining_face_indices(
    all_face_indices: list[tuple[int, int, int]], exclude_face_indices: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """
    Get the remaining face indices from the all face indices by excluding the exclude_face_indices.

    Args:
        all_face_indices: List[Tuple[int, int, int]]
            the full face indices
        exclude_face_indices: List[Tuple[int, int, int]]
            the face indices to exclude

    Returns:
        remaining_face_indices: List[Tuple[int, int, int]]
            the remaining face indices
    """
    remaining_face_indices = []
    for face_indices in all_face_indices:
        if face_indices not in exclude_face_indices:
            remaining_face_indices.append(face_indices)
    return remaining_face_indices


class Geometry2D:
    def __init__(self) -> None:
        raise NotImplementedError("Derived class must implement __init__ method")

    def render(self) -> Any:
        raise NotImplementedError("Derived class must implement render method")


class LineSegment2D(Geometry2D):
    """
    This object includes N line segments, each line segment is defined by two vertices.

    Args:
        xy_and_depth: np.ndarray, [N, 2, 3]
            pixel coordinate and depth of N line segments (two vertices each) after projection
        base_color: np.ndarray, [N, 3]
            base color of the N line segments, it will faded according to the depth
        line_width: float
            line width of the N line segments
    """

    def __init__(
        self, xy_and_depth: np.ndarray, base_color: Union[Tuple[float, float, float], np.ndarray], line_width: float
    ) -> None:
        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if any(base_color > 1.0):
            raise ValueError("color must be in the range [0, 1]")

        self.xy_and_depth = xy_and_depth
        self.base_color = base_color
        self.line_width = line_width

    def render(self, ctx: moderngl.Context, program: moderngl.Program, **kwargs: Any) -> Any:
        """
        Render the line segment.

        Args:
            ctx: moderngl.Context
                the opengl context to render the line segment
            program: moderngl.Program
                the shader program to render the line segment
            **kwargs: dict
                additional arguments, including:
                    - image_width: int
                        the width of the image
        """
        program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]  # pyright: ignore[reportAttributeAccessIssue]

        # prepare vertex position
        vertex_position = self.xy_and_depth.reshape(-1, 3).astype("f4")

        # prepare vertex color
        vertex_base_color = np.tile(self.base_color, (vertex_position.shape[0], 1)).astype("f4")

        # prepare vbo
        vbo = ctx.buffer(vertex_position.tobytes())
        vbo_base_color = ctx.buffer(vertex_base_color.tobytes())
        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
        )
        vao.render()


class PolyLine2D(Geometry2D):
    """
    After projecting 3D vertices to 2D, the vertices are stored in this class.

    Args:
        xy_and_depth: np.ndarray, [N, 3]
            pixel coordinate and depth of N vertices of the polyline after projection
        base_color: np.ndarray, [3]
            base color of the polyline, it will faded according to the depth
        line_width: float
            line width of the polyline
    """

    def __init__(
        self,
        xy_and_depth: Union[List[Tuple[float, float, float]], np.ndarray],
        base_color: Union[Tuple[float, float, float], np.ndarray],
        line_width: float,
    ) -> None:
        if isinstance(xy_and_depth, list):
            xy_and_depth = np.array(xy_and_depth)

        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if any(base_color > 1.0):
            raise ValueError("color must be in the range [0, 1]")

        self.xy_and_depth = xy_and_depth
        self.base_color = base_color
        self.line_width = line_width

    def render(self, ctx: moderngl.Context, program: moderngl.Program, **kwargs: Any) -> Any:
        """
        Render the polyline.

        Args:
            ctx: moderngl.Context
                the opengl context to render the polyline
            program: moderngl.Program
                the shader program to render the polyline
            **kwargs: dict
                additional arguments, including:
                    - image_width: int
                        the width of the image
        """
        program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]  # pyright: ignore[reportAttributeAccessIssue]

        # prepare vertex position
        xy_and_depth = self.xy_and_depth.astype("f4")
        start_vertex_position = xy_and_depth[:-1]  # [N, 3]
        end_vertex_position = xy_and_depth[1:]  # [N, 3]
        vertex_position = np.stack([start_vertex_position, end_vertex_position], axis=-2)  # [N, 2, 3]
        vertex_position = vertex_position.reshape(-1, 3)  # [2N, 3]

        # prepare vertex color
        vertex_base_color = np.tile(self.base_color, (vertex_position.shape[0], 1)).astype("f4")

        # prepare vbo
        vbo = ctx.buffer(vertex_position.tobytes())
        vbo_base_color = ctx.buffer(vertex_base_color.tobytes())
        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
        )
        vao.render()


class TriangleList2D(Geometry2D):
    """
    Render a list of triangles. Useful for rendering triangulated polygons.

    Args:
        triangles: np.ndarray, [M, 3, 3]
            M triangles, each with 3 vertices of shape [x, y, depth]
        base_color: np.ndarray, [3] or [M, 3]
            base color(s) for the triangles, will be faded according to depth
    """

    def __init__(
        self,
        triangles: np.ndarray,
        base_color: Union[Tuple[float, float, float], np.ndarray],
    ) -> None:
        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if (base_color > 1.0).any():
            raise ValueError("color must be in the range [0, 1]")

        self.triangles = triangles
        self.base_color = base_color

    def render(self, ctx: moderngl.Context, program: moderngl.Program, **kwargs: Any) -> Any:
        """
        Render the triangles.

        Args:
            ctx: moderngl.Context
                the opengl context to render the triangles
            program: moderngl.Program
                the shader program to render the triangles
            **kwargs: dict
                not used.
        """
        # Flatten triangles to vertices: [M, 3, 3] -> [M*3, 3]
        vertices = self.triangles.reshape(-1, 3).astype("f4")

        # Prepare colors for each vertex
        if self.base_color.ndim == 1:  # Single color for all triangles
            vertex_colors = np.tile(self.base_color, (len(vertices), 1)).astype("f4")
        else:  # Per-triangle colors: [M, 3] -> [M*3, 3]
            vertex_colors = np.repeat(self.base_color, 3, axis=0).astype("f4")

        vbo = ctx.buffer(vertices.tobytes())
        vbo_base_color = ctx.buffer(vertex_colors.tobytes())

        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.TRIANGLES
        )
        vao.render()


class Polygon2D(Geometry2D):
    """
    After projecting 3D vertices to 2D, the vertices are stored in this class.

    Args:
        xy_and_depth: np.ndarray, [N, 3]
            pixel coordinate and depth of N vertices of the polygon after projection
        base_color: np.ndarray, [3]
            base color of the polygon, it will faded according to the depth
    """

    def __init__(
        self,
        xy_and_depth: Union[List[Tuple[float, float, float]], np.ndarray],
        base_color: Union[Tuple[float, float, float], np.ndarray],
    ) -> None:
        if isinstance(xy_and_depth, list):
            xy_and_depth = np.array(xy_and_depth)

        # if first vertex is not the same as the last vertex, add the first vertex to the end
        if not np.allclose(xy_and_depth[0], xy_and_depth[-1]):
            xy_and_depth = np.concatenate([xy_and_depth, [xy_and_depth[0]]], axis=0)

        if isinstance(base_color, tuple):
            base_color = np.array(base_color)

        if any(base_color > 1.0):
            raise ValueError("color must be in the range [0, 1]")

        self.xy_and_depth = xy_and_depth
        self.base_color = base_color

    def render(self, ctx: moderngl.Context, program: moderngl.Program, **kwargs: Any) -> Any:
        """
        Render the polygon.

        Args:
            ctx: moderngl.Context
                the opengl context to render the polygon
            program: moderngl.Program
                the shader program to render the polygon
            **kwargs: dict
                not used.
        """
        # use all vertices as a triangle fan (ensure the vertex order is correct and closed)
        triangles = self.xy_and_depth[:-1].astype("f4")  # last vertex is the same as the first vertex
        vertex_base_color = np.tile(self.base_color, (len(triangles), 1)).astype("f4")

        vbo = ctx.buffer(triangles.tobytes())
        vbo_base_color = ctx.buffer(vertex_base_color.tobytes())

        # Use the TRIANGLE_FAN mode (more suitable for polygon filling)
        vao = ctx.vertex_array(
            program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.TRIANGLE_FAN
        )
        vao.render()


class BoundingBox2D(Geometry2D):
    """
    3D Bounding box in 2D. xy_and_depth refers to 8 vertices of the bounding box.

    The order of the vertices is:

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

    Args:
        xy_and_depth: np.ndarray, [8, 3]
            pixel coordinate and depth of 8 vertices of the bounding box after projection
        base_color_or_per_vertex_color: tuple or np.ndarray
            - if a tuple, (r, g, b), this is base color shared by all vertices
            - if a list of tuples, [(r, g, b), (r, g, b), ...], this is the color of each vertex
            - if a np.ndarray, [3], this is the base color shared by all vertices
            - if a np.ndarray, [8, 3], this is the color of each vertex

        fill_face: str = 'front', # none, front, all or front_and_back
            fill the front face, back face or both faces
        fill_face_style: str = 'solid',
            - solid: fill the face with color
            - diagonal: connect the diagonal of the face
        line_width: float = 4.0
            line width of the bounding box
        force_all_edge_on_top: bool = False
            if true, make all edges (including diagonal edges) on top of rendered face polygons
        edge_color: np.ndarray, shape (3,), dtype=np.float32, optional edge color
    """

    def __init__(
        self,
        xy_and_depth: Union[List[Tuple[float, float, float]], np.ndarray],
        base_color_or_per_vertex_color: Union[Tuple[float, float, float], List[Tuple[float, float, float]], np.ndarray],
        fill_face: str = "front",  # none, front or all, front_and_back
        fill_face_style: str = "solid",
        line_width: float = 4.0,
        force_all_edge_on_top: bool = False,
        edge_color: Union[Tuple[float, float, float], np.ndarray, None] = None,
    ) -> None:
        if isinstance(xy_and_depth, list):
            xy_and_depth = np.array(xy_and_depth)

        if isinstance(base_color_or_per_vertex_color, tuple) or isinstance(base_color_or_per_vertex_color, list):
            base_color_or_per_vertex_color = np.array(base_color_or_per_vertex_color)

        if (base_color_or_per_vertex_color > 1.0).any():
            raise ValueError("color must be in the range [0, 1]")

        if base_color_or_per_vertex_color.shape == (3,):
            self.per_vertex_color = np.tile(base_color_or_per_vertex_color, (8, 1))
        elif base_color_or_per_vertex_color.shape == (8, 3):
            self.per_vertex_color = base_color_or_per_vertex_color
        else:
            raise ValueError(f"Invalid base_color_or_per_vertex_color: {base_color_or_per_vertex_color.shape}")

        self.xy_and_depth = xy_and_depth
        self.fill_face = fill_face
        self.fill_face_style = fill_face_style
        self.edge_color = edge_color
        self.line_width = line_width
        self.force_all_edge_on_top = force_all_edge_on_top

    def render(
        self,
        ctx: moderngl.Context,
        polyline_program: moderngl.Program,
        polygon_program: moderngl.Program,
        **kwargs: Any,
    ) -> Any:
        polyline_program["u_line_width"].value = 2 * self.line_width / kwargs["image_width"]  # pyright: ignore[reportAttributeAccessIssue]

        # if fill_face_style is solid (not diagonal), draw the solid face
        if self.fill_face == "front" and self.fill_face_style == "solid":
            solid_face_indices = FRONT_FACE_INDICES
            black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, FRONT_FACE_INDICES)
        elif self.fill_face == "back" and self.fill_face_style == "solid":
            solid_face_indices = BACK_FACE_INDICES
            black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, BACK_FACE_INDICES)
        elif self.fill_face == "front_and_back" and self.fill_face_style == "solid":
            solid_face_indices = FRONT_FACE_INDICES + BACK_FACE_INDICES
            black_face_indices = get_remaining_face_indices(ALL_FACE_INDICES, solid_face_indices)
        elif self.fill_face == "all" and self.fill_face_style == "solid":
            solid_face_indices = ALL_FACE_INDICES
            black_face_indices = None
        elif self.fill_face == "none" and self.fill_face_style == "solid":
            solid_face_indices = None
            black_face_indices = ALL_FACE_INDICES
        elif self.fill_face_style == "diagonal":
            solid_face_indices = None
            black_face_indices = ALL_FACE_INDICES  # make all faces black
        else:
            raise ValueError(
                f"Invalid fill_face_style x fill_face combination: {self.fill_face_style} x {self.fill_face}"
            )

        draw_triangles_vertices = []
        draw_triangles_colors = []

        if solid_face_indices is not None:
            for i0, i1, i2 in solid_face_indices:
                draw_triangles_vertices.append(self.xy_and_depth[i0])
                draw_triangles_vertices.append(self.xy_and_depth[i1])
                draw_triangles_vertices.append(self.xy_and_depth[i2])
                draw_triangles_colors.append(self.per_vertex_color[i0])
                draw_triangles_colors.append(self.per_vertex_color[i1])
                draw_triangles_colors.append(self.per_vertex_color[i2])

        if black_face_indices is not None:
            for i0, i1, i2 in black_face_indices:
                draw_triangles_vertices.append(self.xy_and_depth[i0])
                draw_triangles_vertices.append(self.xy_and_depth[i1])
                draw_triangles_vertices.append(self.xy_and_depth[i2])
                draw_triangles_colors.append(np.ones(3) * 0.25)
                draw_triangles_colors.append(np.ones(3) * 0.25)
                draw_triangles_colors.append(np.ones(3) * 0.25)

        if len(draw_triangles_vertices) > 0:
            draw_triangles_vertices = np.concatenate(draw_triangles_vertices, axis=0).astype("f4")
            draw_triangles_colors = np.concatenate(draw_triangles_colors, axis=0).astype("f4")

            vbo = ctx.buffer(draw_triangles_vertices.tobytes())
            vbo_base_color = ctx.buffer(draw_triangles_colors.tobytes())
            vao = ctx.vertex_array(
                polygon_program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.TRIANGLES
            )
            vao.render()

        # draw edges, make small depth shift to make sure edges are on top of the faces
        line_vertices = []
        line_colors = []

        for i0, i1 in EDGE_INDICES:
            line_vertices.append(self.xy_and_depth[i0])
            line_vertices.append(self.xy_and_depth[i1])
            if self.edge_color is not None:
                line_colors.append(self.edge_color)
                line_colors.append(self.edge_color)
            else:
                line_colors.append(self.per_vertex_color[i0])
                line_colors.append(self.per_vertex_color[i1])

        # draw diagonal lines if needed
        if self.fill_face_style == "diagonal" and self.fill_face != "none":
            if self.fill_face == "front":
                diagonal_line_indices = FRONT_DIAGONAL_INDICES
            elif self.fill_face == "front_and_back":
                diagonal_line_indices = FRONT_DIAGONAL_INDICES + BACK_DIAGONAL_INDICES
            elif self.fill_face == "all":
                diagonal_line_indices = ALL_DIAGONAL_INDICES
            else:
                raise ValueError(f"Invalid fill_face: {self.fill_face}")

            for i0, i1 in diagonal_line_indices:
                line_vertices.append(self.xy_and_depth[i0])
                line_vertices.append(self.xy_and_depth[i1])
                line_colors.append(self.per_vertex_color[i0])
                line_colors.append(self.per_vertex_color[i1])

                line_vertices[-1][2] -= 1e-4  # make sure diagonal lines are on top of the faces
                line_vertices[-2][2] -= 1e-4  # make sure diagonal lines are on top of the faces

        line_vertices = np.array(line_vertices, dtype="f4")
        if self.force_all_edge_on_top:
            line_vertices[:, 2] = np.min(self.xy_and_depth[:, 2]) - 1e-4

        line_colors = np.array(line_colors, dtype="f4")

        vbo = ctx.buffer(line_vertices.tobytes())
        vbo_base_color = ctx.buffer(line_colors.tobytes())
        vao = ctx.vertex_array(
            polyline_program, [(vbo, "3f", "in_pos"), (vbo_base_color, "3f", "in_color")], mode=moderngl.LINES
        )
        vao.render()


@attrs.define
class RenderContext:
    ctx: moderngl.Context
    polyline_program: moderngl.Program
    polygon_program: moderngl.Program


def create_polyline_program(ctx: moderngl.Context, depth_gradient: bool) -> moderngl.Program:
    """
    Create a program for rendering polylines.

    Args:
        ctx: moderngl.Context
            the context to create the program
        depth_gradient: bool
            whether to use variable color according to the depth

    Returns:
        program: moderngl.Program
            the program for rendering polyline
    """
    vertex_shader = vertex_shader_graident_color_code if depth_gradient else vertex_shader_constant_color_code
    geometry_shader = geometry_shader_code
    fragment_shader = fragment_shader_code.replace("v_color", "g_color")
    return ctx.program(vertex_shader=vertex_shader, geometry_shader=geometry_shader, fragment_shader=fragment_shader)


def create_polygon_program(ctx: moderngl.Context, depth_gradient: bool) -> moderngl.Program:
    """
    Args:
        ctx: moderngl.Context
            the context to create the program
        depth_gradient: bool
            whether to use variable color according to the depth

    Returns:
        program: moderngl.Program
            the program for rendering polygon
    """
    vertex_shader = vertex_shader_graident_color_code if depth_gradient else vertex_shader_constant_color_code
    fragment_shader = fragment_shader_code
    return ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)


def create_context(
    device_index: int,
    depth_gradient: bool,
    image_height: int,
    image_width: int,
    depth_max: float,
    windowless: bool = True,
) -> RenderContext:
    if windowless:
        ctx = moderngl.create_context(
            standalone=True,
            backend="egl",  # type: ignore
            libgl="libGL.so.1",  # type: ignore
            libegl="libEGL.so.1",  # type: ignore
            device_index=device_index,  # type: ignore
        )
    else:
        ctx = moderngl.get_context()

    ctx.enable(moderngl.DEPTH_TEST)

    polyline_program = create_polyline_program(ctx, depth_gradient)
    polyline_program["image_height"].value = image_height  # type: ignore
    polyline_program["image_width"].value = image_width  # type: ignore
    polyline_program["depth_max"].value = depth_max  # type: ignore

    polygon_program = create_polygon_program(ctx, depth_gradient)
    polygon_program["image_height"].value = image_height  # type: ignore
    polygon_program["image_width"].value = image_width  # type: ignore
    polygon_program["depth_max"].value = depth_max  # type: ignore

    return RenderContext(ctx, polyline_program, polygon_program)
