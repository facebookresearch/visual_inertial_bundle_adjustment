/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Camera.h"

#include <imgui.h>
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "util/sokol_imgui.h"
#undef Success // X11 is leaking this macro

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace viba_gui {

// this deals with where to render a 3d view, maintain the view matrix and handle the view control.
class Viewer3D {
 public:
  Viewer3D();
  ~Viewer3D();

  // Immediate-mode render with callback
  void render(const char* windowName, const std::function<void(const Eigen::Matrix4f&)>& draw);

  // Camera access
  CameraState& camera() {
    return camera_;
  }
  const CameraState& camera() const {
    return camera_;
  }

 private:
  void createRenderTarget(int width, int height);
  void handleMouseInput(const ImVec2& windowPos, const ImVec2& windowSize);
  float readDepthAtPixel(int x, int y);
  Eigen::Vector3f unprojectDepth(const ImVec2& screenPos, float depth, const ImVec2& windowSize)
      const;
  Eigen::Vector3f findPivot(const ImVec2& screenPos, const ImVec2& windowSize);

  // State
  CameraState camera_;
  CameraControls controls_;
  bool camera_initialized_ = false;

  // Render target
  sg_image color_img_;
  sg_image depth_img_;
  sg_view color_attachment_;
  sg_view depth_attachment_;
  sg_view color_texture_;
  int render_width_ = 0;
  int render_height_ = 0;
  sg_pass_action* pass_action_ = nullptr; // Allocated in cpp

  // Interaction state
  bool is_dragging_ = false;
  bool is_panning_ = false;
  bool is_scrolling_ = false;
  ImVec2 last_mouse_pos_;
  std::string interaction_state_;
};

} // namespace viba_gui
