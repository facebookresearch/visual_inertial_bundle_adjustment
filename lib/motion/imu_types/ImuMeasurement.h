/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <filesystem>
#include <fstream>

#include <Eigen/Geometry>
#include <vector>

namespace imu_types {

struct ImuMeasurement {
  int64_t timestampNs;
  double temperatureC;
  Eigen::Vector3d accelMSec2;
  Eigen::Vector3d gyroRadSec;
};

using ImuMeasurements = std::vector<ImuMeasurement>;

} // namespace imu_types
