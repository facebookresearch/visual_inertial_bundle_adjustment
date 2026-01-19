/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Camera.h"
#include <cmath>

namespace viba_gui {

Eigen::Vector3f CameraState::getViewDirection() const {
  return Eigen::Vector3f(
      std::cos(elevation) * std::cos(azimuth),
      std::cos(elevation) * std::sin(azimuth),
      std::sin(elevation));
}

Eigen::Matrix4f CameraState::getViewMatrix() const {
  Eigen::Vector3f forward = getViewDirection();
  Eigen::Vector3f right = forward.cross(Eigen::Vector3f(0, 0, 1)).normalized();
  Eigen::Vector3f up = right.cross(forward).normalized();

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  view.block<1, 3>(0, 0) = right.transpose();
  view.block<1, 3>(1, 0) = up.transpose();
  view.block<1, 3>(2, 0) = -forward.transpose();
  view.block<3, 1>(0, 3) = -(view.block<3, 3>(0, 0) * position);

  return view;
}

Eigen::Matrix4f CameraState::getProjectionMatrix(float aspect, float fov, float near, float far)
    const {
  float f = 1.0f / std::tan(fov * 0.5f * M_PI / 180.0f);

  Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
  proj(0, 0) = f / aspect;
  proj(1, 1) = f;
  proj(2, 2) = (far + near) / (near - far);
  proj(2, 3) = (2.0f * far * near) / (near - far);
  proj(3, 2) = -1.0f;

  return proj;
}

} // namespace viba_gui
