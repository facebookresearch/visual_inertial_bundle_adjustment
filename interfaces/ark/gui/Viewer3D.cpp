/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Viewer3D.h"
#include <algorithm>
#include <cmath>

// OpenGL headers for direct GL calls (depth reading)
#if defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

namespace viba_gui {

Viewer3D::Viewer3D() {
  pass_action_ = new sg_pass_action();
  pass_action_->colors[0].load_action = SG_LOADACTION_CLEAR;
  pass_action_->colors[0].clear_value = {0.1f, 0.1f, 0.12f, 1.0f};
  pass_action_->depth.load_action = SG_LOADACTION_CLEAR;
  pass_action_->depth.clear_value = 1.0f;

  // Initialize IDs
  color_img_.id = 0;
  depth_img_.id = 0;
  color_attachment_.id = 0;
  depth_attachment_.id = 0;
  color_texture_.id = 0;
}

Viewer3D::~Viewer3D() {
  if (color_img_.id != 0) {
    sg_destroy_view(color_attachment_);
    sg_destroy_view(depth_attachment_);
    sg_destroy_view(color_texture_);
    sg_destroy_image(color_img_);
    sg_destroy_image(depth_img_);
  }

  delete pass_action_;
}

void Viewer3D::render(
    const char* windowName,
    const std::function<void(const Eigen::Matrix4f&)>& draw) {
  ImGui::Begin(windowName, nullptr, ImGuiWindowFlags_NoCollapse);

  ImVec2 window_pos = ImGui::GetCursorScreenPos();
  ImVec2 window_size = ImGui::GetContentRegionAvail();

  if (window_size.x < 1 || window_size.y < 1) {
    ImGui::End();
    return;
  }

  // Ensure render target matches window size
  if (render_width_ != (int)window_size.x || render_height_ != (int)window_size.y) {
    createRenderTarget((int)window_size.x, (int)window_size.y);
  }

  if (color_attachment_.id == 0) {
    ImGui::End();
    return;
  }

  // Compute MVP matrix
  float aspect = window_size.x / window_size.y;
  Eigen::Matrix4f mvp = camera_.getProjectionMatrix(aspect) * camera_.getViewMatrix();

  // Begin offscreen pass (WHERE to render)
  sg_pass pass = {};
  pass.action = *pass_action_;
  pass.attachments.colors[0] = color_attachment_;
  pass.attachments.depth_stencil = depth_attachment_;
  sg_begin_pass(&pass);

  // User callback draws geometry (HOW to render)
  draw(mvp);

  sg_end_pass();

  // Display the rendered texture in ImGui
  if (color_texture_.id != 0) {
    ImTextureID tex_id = (ImTextureID)simgui_imtextureid(color_texture_);

    // Flip texture vertically (OpenGL has origin at bottom-left, ImGui at top-left)
    ImGui::Image(tex_id, window_size, ImVec2(0, 1), ImVec2(1, 0));

    // Check if image is hovered or active
    bool image_hovered = ImGui::IsItemHovered();
    bool image_active = ImGui::IsItemActive();
    bool currently_interacting = is_dragging_ || is_panning_ || is_scrolling_;

    // Add debug info overlay
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    char info[512];
    if (interaction_state_.empty()) {
      snprintf(
          info,
          sizeof(info),
          "Camera pos: (%.1f, %.1f, %.1f)\\nPivot: (%.1f, %.1f, %.1f)",
          camera_.position.x(),
          camera_.position.y(),
          camera_.position.z(),
          controls_.pivot.x(),
          controls_.pivot.y(),
          controls_.pivot.z());
    } else {
      snprintf(
          info,
          sizeof(info),
          "Camera pos: (%.1f, %.1f, %.1f)\\nPivot: (%.1f, %.1f, %.1f)\\nState: %s",
          camera_.position.x(),
          camera_.position.y(),
          camera_.position.z(),
          controls_.pivot.x(),
          controls_.pivot.y(),
          controls_.pivot.z(),
          interaction_state_.c_str());
    }
    draw_list->AddText(
        ImVec2(window_pos.x + 10, window_pos.y + 10), IM_COL32(200, 200, 200, 255), info);

    // Handle mouse input if image is hovered/active OR if we're currently in an interaction
    if (image_hovered || image_active || currently_interacting) {
      handleMouseInput(window_pos, window_size);
    } else {
      // Reset interaction state only when truly not interacting
      is_dragging_ = false;
      is_panning_ = false;
      is_scrolling_ = false;
      interaction_state_ = "";
    }
  }

  ImGui::End();
}

void Viewer3D::createRenderTarget(int width, int height) {
  if (width <= 0 || height <= 0)
    return;

  // Destroy old render target if it exists
  if (color_img_.id != 0) {
    sg_destroy_view(color_attachment_);
    sg_destroy_view(depth_attachment_);
    sg_destroy_view(color_texture_);
    sg_destroy_image(color_img_);
    sg_destroy_image(depth_img_);
  }

  // Create color texture
  sg_image_desc color_img_desc = {};
  color_img_desc.usage.color_attachment = true;
  color_img_desc.width = width;
  color_img_desc.height = height;
  color_img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
  color_img_desc.label = "viewer3d_color_target";
  color_img_ = sg_make_image(&color_img_desc);

  // Create depth texture
  sg_image_desc depth_img_desc = {};
  depth_img_desc.usage.depth_stencil_attachment = true;
  depth_img_desc.width = width;
  depth_img_desc.height = height;
  depth_img_desc.pixel_format = SG_PIXELFORMAT_DEPTH;
  depth_img_desc.label = "viewer3d_depth_target";
  depth_img_ = sg_make_image(&depth_img_desc);

  // Create attachment views (for rendering TO the images)
  sg_view_desc color_att_desc = {};
  color_att_desc.color_attachment.image = color_img_;
  color_att_desc.label = "viewer3d_color_attachment_view";
  color_attachment_ = sg_make_view(&color_att_desc);

  sg_view_desc depth_att_desc = {};
  depth_att_desc.depth_stencil_attachment.image = depth_img_;
  depth_att_desc.label = "viewer3d_depth_attachment_view";
  depth_attachment_ = sg_make_view(&depth_att_desc);

  // Create texture view (for sampling FROM the color image - displaying in ImGui)
  sg_view_desc tex_desc = {};
  tex_desc.texture.image = color_img_;
  tex_desc.label = "viewer3d_color_texture_view";
  color_texture_ = sg_make_view(&tex_desc);

  render_width_ = width;
  render_height_ = height;
}

float Viewer3D::readDepthAtPixel(int x, int y) {
  if (depth_img_.id == 0) {
    return 1.0f; // Far plane if no depth buffer
  }

  float min_depth = 1.0f;

// OpenGL-specific depth reading
#if defined(__APPLE__) || defined(__linux__)
  // Define a small window around the pixel (hwin=1 -> 3x3)
  const int hwin = 1;
  const int zl = (hwin * 2 + 1);
  const int zsize = zl * zl;

  // Clamp the read region to the render target bounds
  int x0 = std::max(0, x - hwin);
  int y0 = std::max(0, y - hwin);
  int x1 = std::min(render_width_ - 1, x + hwin);
  int y1 = std::min(render_height_ - 1, y + hwin);

  int read_w = x1 - x0 + 1;
  int read_h = y1 - y0 + 1;

  if (read_w > 0 && read_h > 0) {
    // Flip Y coordinate (OpenGL has origin at bottom-left)
    int gl_y = render_height_ - (y0 + read_h); // bottom-left of the block in GL coords

    // Get the OpenGL texture ID
    sg_gl_image_info gl_depth_info = sg_gl_query_image_info(depth_img_);
    GLuint depth_tex = gl_depth_info.tex[gl_depth_info.active_slot];

    // Bind the framebuffer to read from
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);

    // Attach depth texture
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);

    // Read a small block of depth values
    std::vector<float> zs(read_w * read_h, 1.0f);
    glReadPixels(x0, gl_y, read_w, read_h, GL_DEPTH_COMPONENT, GL_FLOAT, zs.data());

    // Find minimum valid depth in the block
    for (int i = 0; i < read_w * read_h; ++i) {
      float d = zs[i];
      if (d < min_depth) {
        min_depth = d;
      }
    }

    // Cleanup
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
  }
#endif

  return min_depth;
}

Eigen::Vector3f
Viewer3D::unprojectDepth(const ImVec2& screenPos, float depth, const ImVec2& windowSize) const {
  float aspect = windowSize.x / windowSize.y;
  Eigen::Matrix4f proj = camera_.getProjectionMatrix(aspect);
  Eigen::Matrix4f view = camera_.getViewMatrix();
  Eigen::Matrix4f mvp = proj * view;

  // Convert screen coordinates to NDC
  float ndc_x = (screenPos.x / windowSize.x) * 2.0f - 1.0f;
  float ndc_y = 1.0f - (screenPos.y / windowSize.y) * 2.0f; // Flip Y
  float ndc_z = depth * 2.0f - 1.0f; // Convert [0,1] to [-1,1]

  // Unproject from NDC to world space
  Eigen::Vector4f clip_pos(ndc_x, ndc_y, ndc_z, 1.0f);
  Eigen::Matrix4f inv_mvp = mvp.inverse();
  Eigen::Vector4f world_pos = inv_mvp * clip_pos;

  // Perspective divide
  if (std::abs(world_pos.w()) > 0.0001f) {
    world_pos /= world_pos.w();
  }

  return world_pos.head<3>();
}

Eigen::Vector3f Viewer3D::findPivot(const ImVec2& screenPos, const ImVec2& windowSize) {
  // Try GPU-based picking
  if (depth_img_.id != 0) {
    float depth = readDepthAtPixel((int)screenPos.x, (int)screenPos.y);

    // If depth is valid (not at far plane), unproject it
    if (depth < 0.9999f) {
      return unprojectDepth(screenPos, depth, windowSize);
    }
  }

  // Fallback to current pivot if no geometry hit
  return controls_.pivot;
}

void Viewer3D::handleMouseInput(const ImVec2& windowPos, const ImVec2& windowSize) {
  // Only handle input if the window is hovered (mouse is over it)
  if (!ImGui::IsWindowHovered()) {
    is_dragging_ = false;
    is_panning_ = false;
    is_scrolling_ = false;
    interaction_state_ = "";
    return;
  }

  ImGuiIO& io = ImGui::GetIO();
  ImVec2 mouse_pos = ImGui::GetMousePos();
  ImVec2 relative_pos(mouse_pos.x - windowPos.x, mouse_pos.y - windowPos.y);

  // Check if mouse is in window
  bool in_window = relative_pos.x >= 0 && relative_pos.x < windowSize.x && relative_pos.y >= 0 &&
      relative_pos.y < windowSize.y;

  if (!in_window) {
    is_dragging_ = false;
    is_panning_ = false;
    is_scrolling_ = false;
    interaction_state_ = "";
    return;
  }

  // Clear interaction state at start of frame (will be set by active interactions below)
  if (!is_dragging_ && !is_panning_ && !is_scrolling_) {
    interaction_state_ = "";
  }

  // Left mouse button: rotate around point under cursor
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && in_window) {
    controls_.pivot = findPivot(relative_pos, windowSize);
    is_dragging_ = true;
    last_mouse_pos_ = mouse_pos;
    interaction_state_ = "ROTATING";
  }

  if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && is_dragging_) {
    interaction_state_ = "ROTATING";
    ImVec2 delta(mouse_pos.x - last_mouse_pos_.x, mouse_pos.y - last_mouse_pos_.y);

    Eigen::Vector3f offset = camera_.position - controls_.pivot;

    if (offset.norm() > 0.01f) {
      // Horizontal rotation: rotate around world Z axis through pivot
      float angle_h = -delta.x * 0.01f;
      Eigen::AngleAxisf rot_h(angle_h, Eigen::Vector3f::UnitZ());

      // Vertical rotation: rotate around the "right" vector
      Eigen::Vector3f view_dir = camera_.getViewDirection();
      Eigen::Vector3f right(std::sin(camera_.azimuth), -std::cos(camera_.azimuth), 0.0f);
      float angle_v = -delta.y * 0.01f;

      // Check if we're at elevation limits to prevent unwanted panning
      const float elev_min = -M_PI / 2.0f + 0.1f;
      const float elev_max = M_PI / 2.0f - 0.1f;
      const float elev_threshold = 0.05f;

      bool at_upper_limit = (camera_.elevation >= elev_max - elev_threshold) && (angle_v > 0);
      bool at_lower_limit = (camera_.elevation <= elev_min + elev_threshold) && (angle_v < 0);
      bool skip_vertical = at_upper_limit || at_lower_limit;

      // Apply rotations to camera position offset
      Eigen::Vector3f new_offset;
      Eigen::Vector3f new_view_dir;

      if (skip_vertical) {
        new_offset = rot_h * offset;
        new_view_dir = rot_h * view_dir;
      } else {
        Eigen::AngleAxisf rot_v(angle_v, right);
        new_offset = rot_h * rot_v * offset;
        new_view_dir = rot_h * rot_v * view_dir;
      }

      camera_.position = controls_.pivot + new_offset;

      // Convert back to azimuth/elevation
      float horizontal_dist =
          std::sqrt(new_view_dir.x() * new_view_dir.x() + new_view_dir.y() * new_view_dir.y());
      camera_.azimuth = std::atan2(new_view_dir.y(), new_view_dir.x());
      camera_.elevation = std::atan2(new_view_dir.z(), horizontal_dist);
      camera_.elevation = std::max(
          float(-M_PI / 2.0f + 0.1f), std::min(float(M_PI / 2.0f - 0.1f), camera_.elevation));
    }

    last_mouse_pos_ = mouse_pos;
  }

  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    is_dragging_ = false;
    interaction_state_ = "";
  }

  // Middle or Right mouse button: pan
  if ((ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
       ImGui::IsMouseClicked(ImGuiMouseButton_Right)) &&
      in_window) {
    is_panning_ = true;
    last_mouse_pos_ = mouse_pos;
    interaction_state_ = "PANNING";
  }

  if ((ImGui::IsMouseDown(ImGuiMouseButton_Middle) || ImGui::IsMouseDown(ImGuiMouseButton_Right)) &&
      is_panning_) {
    interaction_state_ = "PANNING";
    ImVec2 delta(mouse_pos.x - last_mouse_pos_.x, mouse_pos.y - last_mouse_pos_.y);

    // Get camera coordinate frame
    Eigen::Vector3f forward = camera_.getViewDirection();
    Eigen::Vector3f right = forward.cross(Eigen::Vector3f(0, 0, 1)).normalized();
    Eigen::Vector3f up = right.cross(forward).normalized();

    // Move camera and pivot together in screen plane
    float pan_scale = (camera_.position - controls_.pivot).norm() * 0.001f;
    Eigen::Vector3f pan_offset = -right * delta.x * pan_scale + up * delta.y * pan_scale;

    camera_.position += pan_offset;
    controls_.pivot += pan_offset;

    last_mouse_pos_ = mouse_pos;
  }

  if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle) ||
      ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
    is_panning_ = false;
    interaction_state_ = "";
  }

  // Scroll: zoom toward point under mouse
  if (in_window && io.MouseWheel != 0.0f) {
    interaction_state_ = "ZOOMING";

    // Only find pivot when scroll STARTS
    if (!is_scrolling_) {
      controls_.pivot = findPivot(relative_pos, windowSize);
      is_scrolling_ = true;
    }

    // Zoom factor
    float zoom_scale = 1.0f - io.MouseWheel * 0.1f;

    // Move camera toward/away from zoom target
    Eigen::Vector3f offset = camera_.position - controls_.pivot;
    camera_.position = controls_.pivot + offset * zoom_scale;
  } else if (is_scrolling_) {
    // Reset scrolling state when wheel stops
    is_scrolling_ = false;
  }
}

} // namespace viba_gui
