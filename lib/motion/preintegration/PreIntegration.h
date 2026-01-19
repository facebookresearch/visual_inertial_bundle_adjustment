/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <preintegration/ImuTypes.h>
#include <preintegration/MotionIntegral.h>
#include <optional>

namespace visual_inertial_ba::preintegration {

using Mat99 = Eigen::Matrix<double, 9, 9>;

struct PreIntegration {
  RotVelPos rvp;
  Mat9X J;
  Mat99 rvpCov;
  Vec3 omegaAtEnd; // in imu
  ImuMeasurementModelParameters calibEvalPoint;
};

// compute full preintegration
PreIntegration computePreIntegration(
    const ImuCalibrationJacobianIndices& jacInd,
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    const ImuNoiseModelParameters& noiseModel,
    int64_t timeStartUs,
    int64_t timeEndUs);

// only compute RVP
RotVelPos integrateMeasurements(
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    int64_t timeStartUs,
    int64_t timeEndUs);

// callback `enumFunc` called for each step:
//   enumFunc(rvp, atAccelBoundary, atGyroBoundary)
void forEachIntegratedMeasurement(
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    int64_t timeStartUs,
    int64_t timeEndUs,
    const std::function<void(const RotVelPos&, bool, bool)>& enumFunc);

} // namespace visual_inertial_ba::preintegration
