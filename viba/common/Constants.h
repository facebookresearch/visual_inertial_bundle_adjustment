/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <limits>

namespace visual_inertial_ba {

constexpr int kNumRigsForIterative = 20000;

constexpr double kDefaultGravityMagnitude = 9.81;

constexpr double kMultiImuOmegaPriorStdRadSec = 10.0 * M_PI / 180;

constexpr double kReprojectionErrorHuberLossWidth = 1.0;
constexpr double kReprojectionErrorHuberLossCutoff = 3.0;

constexpr double kImuErrorHuberLossWidth = std::numeric_limits<double>::infinity();
constexpr double kImuErrorHuberLossCutoff = std::numeric_limits<double>::infinity();

} // namespace visual_inertial_ba
