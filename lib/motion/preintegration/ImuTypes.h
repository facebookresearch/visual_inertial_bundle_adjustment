/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <imu_types/ImuCalibrationJacobianIndices.h>
#include <imu_types/ImuMeasurement.h>
#include <imu_types/ImuMeasurementModelParameters.h>
#include <imu_types/ImuNoiseModelParameters.h>
#include <imu_types/SignalStatistics.h>

namespace visual_inertial_ba::preintegration {

using ImuMeasurement = ::imu_types::ImuMeasurement;
using ImuMeasurementModelParameters = ::imu_types::ImuMeasurementModelParameters;
using ImuNoiseModelParameters = ::imu_types::ImuNoiseModelParameters;
using ImuCalibrationOptions = ::imu_types::ImuCalibrationOptions;
using ImuCalibrationJacobianIndices = ::imu_types::ImuCalibrationJacobianIndices;
using SignalStatistics = ::imu_types::SignalStatistics;

// those interface functions allow to swap the ImuMeasurement type
// with a type having different timestamp types (eg std::chrono::nanoseconds)

// get timestamp from ImuMeasurement
inline int64_t timestampNs(const ImuMeasurement& m) {
  return m.timestampNs;
}

// create ImuMeasurement from timestamp and other fields
inline ImuMeasurement newImuMeasurement(
    const Eigen::Vector3d& accelMSec2_,
    const Eigen::Vector3d& gyroRadSec_,
    int64_t timestampNs) {
  return ImuMeasurement{
      .timestampNs = timestampNs,
      .temperatureC = -800,
      .accelMSec2 = accelMSec2_,
      .gyroRadSec = gyroRadSec_,
  };
}

} // namespace visual_inertial_ba::preintegration
