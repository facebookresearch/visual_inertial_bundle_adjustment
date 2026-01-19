/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <preintegration/ImuTypes.h>

namespace visual_inertial_ba::preintegration {

using Vec3 = Eigen::Vector3d;
using VecX = Eigen::VectorXd;
using Mat3 = Eigen::Matrix3d;
using MatX = Eigen::MatrixXd;
using Mat66 = Eigen::Matrix<double, 6, 6>;
using Mat6X = Eigen::Matrix<double, 6, Eigen::Dynamic>;
template <typename T>
using Ref = Eigen::Ref<T>;
using SO3 = Sophus::SO3d;

// apply a correction to the state (aka boxPlus)
void boxPlus(
    ImuMeasurementModelParameters& modelParams,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<const VecX> correction);

// compute the residual to another state (aka boxMinus)
void boxMinus(
    const ImuMeasurementModelParameters& modelParams,
    const ImuMeasurementModelParameters& refModelParams,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<VecX> res);

// compute compensated gyro/accel and jacobian wrt full calibration
void getCompensatedImuMeasurementAndJac(
    const ImuMeasurementModelParameters& modelParams,
    const SignalStatistics& uncompensatedGyroRadSec,
    const SignalStatistics& uncompensatedAccelMSec2,
    Vec3& compensatedGyroRadSec,
    Vec3& compensatedAccelMSec2,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<Mat6X> calibJac,
    Ref<Mat66> measJac);

} // namespace visual_inertial_ba::preintegration
