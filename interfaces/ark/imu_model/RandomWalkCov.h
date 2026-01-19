/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <imu_model/ImuCalibParam.h>
#include <imu_types/ImuNoiseModelParameters.h>

namespace visual_inertial_ba {

using ImuCalibrationJacobianIndices = ::imu_types::ImuCalibrationJacobianIndices;
using ImuNoiseModelParameters = ::imu_types::ImuNoiseModelParameters;

Eigen::VectorXd imuCalibRandomWalkCov(
    const ImuNoiseModelParameters& noiseParams,
    const ImuCalibrationJacobianIndices& jacInd,
    double dtSec);

Eigen::VectorXd imuCalibTurnonStdDev(
    const ImuNoiseModelParameters& noiseParams,
    const ImuCalibrationJacobianIndices& jacInd);

} // namespace visual_inertial_ba
