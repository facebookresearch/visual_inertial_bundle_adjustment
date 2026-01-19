/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Geometry.h"

#include "sokol_gfx.h"

#include <algorithm>

namespace viba_gui {
namespace geometry {

// Shader sources
static const char* vs_trajectory_src = R"(
#version 330
uniform mat4 u_mvp;
layout(location=0) in vec3 a_position;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = vec3(1.0, 0.5, 0.0); // Orange
}
)";

static const char* fs_trajectory_src = R"(
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
)";

static const char* vs_points_src = R"(
#version 330
uniform mat4 u_mvp;
layout(location=0) in vec3 a_position;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = 3.0;
    v_color = vec3(0.3, 0.7, 0.3); // Green
}
)";

static const char* fs_points_src = R"(
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 0.6);
}
)";

// Internal module state
namespace detail {
sg_pipeline g_points_pipeline{0};
sg_pipeline g_lines_pipeline{0};
sg_shader g_points_shader{0};
sg_shader g_lines_shader{0};
bool g_initialized = false;

void ensureInitialized() {
  if (g_initialized)
    return;

  // Create trajectory (lines) shader
  sg_shader_desc lines_shader_desc = {};
  lines_shader_desc.vertex_func.source = vs_trajectory_src;
  lines_shader_desc.fragment_func.source = fs_trajectory_src;
  lines_shader_desc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
  lines_shader_desc.uniform_blocks[0].size = sizeof(float) * 16;
  lines_shader_desc.uniform_blocks[0].glsl_uniforms[0].glsl_name = "u_mvp";
  lines_shader_desc.uniform_blocks[0].glsl_uniforms[0].type = SG_UNIFORMTYPE_MAT4;
  lines_shader_desc.label = "lines_shader";
  g_lines_shader = sg_make_shader(&lines_shader_desc);

  // Create points shader
  sg_shader_desc points_shader_desc = {};
  points_shader_desc.vertex_func.source = vs_points_src;
  points_shader_desc.fragment_func.source = fs_points_src;
  points_shader_desc.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
  points_shader_desc.uniform_blocks[0].size = sizeof(float) * 16;
  points_shader_desc.uniform_blocks[0].glsl_uniforms[0].glsl_name = "u_mvp";
  points_shader_desc.uniform_blocks[0].glsl_uniforms[0].type = SG_UNIFORMTYPE_MAT4;
  points_shader_desc.label = "points_shader";
  g_points_shader = sg_make_shader(&points_shader_desc);

  // Create lines pipeline
  sg_pipeline_desc lines_pip_desc = {};
  lines_pip_desc.shader = g_lines_shader;
  lines_pip_desc.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
  lines_pip_desc.primitive_type = SG_PRIMITIVETYPE_LINE_STRIP;
  lines_pip_desc.index_type = SG_INDEXTYPE_NONE;
  lines_pip_desc.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
  lines_pip_desc.depth.write_enabled = true;
  lines_pip_desc.colors[0].blend.enabled = false;
  lines_pip_desc.label = "lines_pipeline";
  g_lines_pipeline = sg_make_pipeline(&lines_pip_desc);

  // Create points pipeline
  sg_pipeline_desc points_pip_desc = {};
  points_pip_desc.shader = g_points_shader;
  points_pip_desc.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
  points_pip_desc.primitive_type = SG_PRIMITIVETYPE_POINTS;
  points_pip_desc.index_type = SG_INDEXTYPE_NONE;
  points_pip_desc.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
  points_pip_desc.depth.write_enabled = true;
  points_pip_desc.colors[0].blend.enabled = true;
  points_pip_desc.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA;
  points_pip_desc.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
  points_pip_desc.label = "points_pipeline";
  g_points_pipeline = sg_make_pipeline(&points_pip_desc);

  g_initialized = true;
}
} // namespace detail

GeometryBuffer createBuffer(const std::vector<Eigen::Vector3f>& vertices) {
  if (vertices.empty()) {
    return GeometryBuffer{};
  }

  sg_buffer_desc vbuf_desc = {};
  vbuf_desc.data.ptr = vertices.data();
  vbuf_desc.data.size = vertices.size() * sizeof(Eigen::Vector3f);
  vbuf_desc.label = "geometry_vertices";

  GeometryBuffer buf;
  buf.buffer = sg_make_buffer(&vbuf_desc);
  buf.count = vertices.size();

  return buf;
}

void updateBuffer(GeometryBuffer& buf, const std::vector<Eigen::Vector3f>& vertices) {
  if (buf.buffer.id != 0) {
    sg_destroy_buffer(buf.buffer);
  }

  buf = createBuffer(vertices);
}

void destroyBuffer(GeometryBuffer& buf) {
  if (buf.buffer.id != 0) {
    sg_destroy_buffer(buf.buffer);
    buf.buffer.id = 0;
    buf.count = 0;
  }
}

void drawPoints(
    const GeometryBuffer& buf,
    const Eigen::Matrix4f& mvp,
    const Eigen::Vector3f& color,
    float alpha) {
  if (buf.buffer.id == 0 || buf.count == 0) {
    return;
  }

  detail::ensureInitialized();

  sg_apply_pipeline(detail::g_points_pipeline);

  sg_bindings bindings = {};
  bindings.vertex_buffers[0] = buf.buffer;
  sg_apply_bindings(&bindings);

  sg_range mvp_data = SG_RANGE_REF(mvp);
  sg_apply_uniforms(0, &mvp_data);

  sg_draw(0, buf.count, 1);
}

void drawLineStrip(
    const GeometryBuffer& buf,
    const Eigen::Matrix4f& mvp,
    const Eigen::Vector3f& color) {
  if (buf.buffer.id == 0 || buf.count < 2) {
    return;
  }

  detail::ensureInitialized();

  sg_apply_pipeline(detail::g_lines_pipeline);

  sg_bindings bindings = {};
  bindings.vertex_buffers[0] = buf.buffer;
  sg_apply_bindings(&bindings);

  sg_range mvp_data = SG_RANGE_REF(mvp);
  sg_apply_uniforms(0, &mvp_data);

  sg_draw(0, buf.count, 1);
}

void shutdown() {
  if (detail::g_initialized) {
    sg_destroy_pipeline(detail::g_points_pipeline);
    sg_destroy_pipeline(detail::g_lines_pipeline);
    sg_destroy_shader(detail::g_points_shader);
    sg_destroy_shader(detail::g_lines_shader);
    detail::g_initialized = false;
  }
}

} // namespace geometry
} // namespace viba_gui
