/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Geometry>

namespace visual_inertial_ba {

struct PointObservation {
  // unique point track id
  int64_t pointId{};

  int64_t captureTimestampUs{};
  int cameraIndex{};

  // image projection and sqrt information
  Eigen::Vector2f projectionBaseRes;
  Eigen::Matrix2f sqrtH_BaseRes;
};

using PointObservations = std::vector<PointObservation>;

} // namespace visual_inertial_ba
