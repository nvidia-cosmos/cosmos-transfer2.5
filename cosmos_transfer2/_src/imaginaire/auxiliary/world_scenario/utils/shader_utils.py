# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Shader utilities for rendering 2D geometries.

Modified from cosmos-av-sample-toolkits.
"""

vertex_shader_graident_color_code = """
#version 330
in vec3 in_pos;
in vec3 in_color;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
out vec3 v_color;
void main() {
    float x_ndc = 2.0 * in_pos.x / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * in_pos.y / image_height;
    float z_ndc = (in_pos.z / depth_max) * 2.0 - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);

    float r = clamp((1.0 - z_ndc) / 2.0, 0.0, 1.0);
    v_color = in_color * r;
}
"""

vertex_shader_constant_color_code = """
#version 330
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
void main() {
    float x_ndc = 2.0 * in_pos.x / image_width - 1.0;
    float y_ndc = 1.0 - 2.0 * in_pos.y / image_height;
    float z_ndc = (in_pos.z / depth_max) * 2.0 - 1.0;
    gl_Position = vec4(x_ndc, y_ndc, z_ndc, 1.0);
    v_color = in_color;
}
"""

geometry_shader_code = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 42) out;

uniform float u_line_width;

in vec3 v_color[];
out vec3 g_color;

void createRoundCap(vec4 center, vec2 direction, vec3 color, float line_width) {
    const int segments = 8;
    const float PI = 3.1415926;

    vec2 perpendicular = vec2(-direction.y, direction.x);

    for (int i = 0; i <= segments; i++) {
        float angle = (float(i) / float(segments)) * PI;
        vec2 offset = cos(angle) * perpendicular + sin(angle) * direction;
        offset *= (line_width / 2.0);

        g_color = color;
        gl_Position = center + vec4(offset, 0.0, 0.0);
        EmitVertex();

        if (i > 0) {
            g_color = color;
            gl_Position = center;
            EmitVertex();
        }
    }
    EndPrimitive();
}

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // p.z is z_ndc in [-1, 1], -1 is near, 1 is far
    float avg_z_ndc = (p0.z + p1.z) / 2.0;
    // Remap z from [-1, 1] to [1, 0] for scaling factor
    float depth_scale = (1.0 - avg_z_ndc) / 2.0;

    depth_scale = clamp(depth_scale, 0.0, 1.0);
    float dynamic_line_width = u_line_width * depth_scale;

    vec2 p0_ndc = p0.xy / p0.w;
    vec2 p1_ndc = p1.xy / p1.w;

    vec2 dir = normalize(p1_ndc - p0_ndc);
    vec2 normal = vec2(-dir.y, dir.x);

    float half_width = dynamic_line_width / 2.0;

    vec4 offset = vec4(normal * half_width, 0.0, 0.0);

    g_color = v_color[0];
    gl_Position = p0 + offset;
    EmitVertex();

    g_color = v_color[0];
    gl_Position = p0 - offset;
    EmitVertex();

    g_color = v_color[1];
    gl_Position = p1 + offset;
    EmitVertex();

    g_color = v_color[1];
    gl_Position = p1 - offset;
    EmitVertex();

    EndPrimitive();

    createRoundCap(p0, -dir, v_color[0], dynamic_line_width);

    createRoundCap(p1, dir, v_color[1], dynamic_line_width);
}
"""

fragment_shader_code = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

# no color as output, only update the position for updating depth buffer
depth_only_vertex_shader_code = """
#version 330
in vec3 in_pos;
uniform float image_height;
uniform float image_width;
uniform float depth_max;
void main() {
    float x_ndc = 2 * in_pos.x / image_width - 1;
    float y_ndc = 1 - 2 * in_pos.y / image_height;
    gl_Position = vec4(x_ndc, y_ndc, in_pos.z / depth_max, 1.0);
}
"""

# no color as output, only update the position for updating depth buffer
depth_only_fragment_shader_code = """
#version 330
void main() {
}
"""
