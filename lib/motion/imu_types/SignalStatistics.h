/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

namespace imu_types {

struct SignalStatistics {
  // The average signal value.
  Eigen::Vector3d averageSignal;

  // The rate of change (temporal derivative) of the signal.
  Eigen::Vector3d rate;
};

} // namespace imu_types
