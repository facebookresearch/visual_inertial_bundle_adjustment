/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <imu_types/ImuMeasurementModelParameters.h>
#include <projectaria_tools/core/calibration/ImuMagnetometerCalibration.h>

namespace visual_inertial_ba {

using namespace projectaria::tools::calibration;

using ImuMeasurementModelParameters = ::imu_types::ImuMeasurementModelParameters;

ImuMeasurementModelParameters fromProjectAriaCalibration(const ImuCalibration& calib);

ImuCalibration toProjectAriaCalibration(
    const ImuMeasurementModelParameters& calib,
    const std::string& label,
    const Sophus::SE3d& T_Device_Imu);

} // namespace visual_inertial_ba
