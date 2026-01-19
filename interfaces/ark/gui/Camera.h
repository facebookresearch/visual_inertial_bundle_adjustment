/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace viba_gui {

struct CameraState {
  Eigen::Vector3f position{0.0f, 0.0f, 5.0f};
  float azimuth = 0.0f; // Horizontal rotation (radians)
  float elevation = 0.0f; // Vertical rotation (radians)

  Eigen::Vector3f getViewDirection() const;
  Eigen::Matrix4f getViewMatrix() const;
  Eigen::Matrix4f getProjectionMatrix(
      float aspect,
      float fov = 45.0f,
      float near = 0.1f,
      float far = 1000.0f) const;
};

struct CameraControls {
  Eigen::Vector3f pivot{0.0f, 0.0f, 0.0f}; // Point to rotate/zoom around
  bool is_dragging = false;
  bool is_panning = false;
  bool is_scrolling = false;
};

} // namespace viba_gui
