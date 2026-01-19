/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <imu_types/ImuNoiseModelParameters.h>

namespace visual_inertial_ba {

using ImuNoiseModelParameters = ::imu_types::ImuNoiseModelParameters;

Eigen::Vector<double, 6> imuExtrRandomWalkCov(
    const ImuNoiseModelParameters& noiseParams,
    double dtSec);

Eigen::Vector<double, 6> camExtrRandomWalkCov(double dtSec);

} // namespace visual_inertial_ba
