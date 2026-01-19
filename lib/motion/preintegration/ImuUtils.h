/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <preintegration/ImuTypes.h>

namespace visual_inertial_ba::preintegration {

using MatX = Eigen::MatrixXd;
using MatX6 = Eigen::Matrix<double, Eigen::Dynamic, 6>;
template <typename T>
using Ref = Eigen::Ref<T>;

// apply normalization as after `boxPlus`, enforcing constraints
void normalizeImuParams(ImuMeasurementModelParameters& modelParams);

// return a factory calibration value
ImuMeasurementModelParameters factoryImuParams();

// return printable string with jacobian blocks
std::string printableImuCalibJacDelta(
    Ref<const MatX> J,
    const ImuCalibrationJacobianIndices& jacInd,
    const std::vector<std::tuple<std::string, int, int>>& arrivalRanges,
    bool printBlocks = false,
    double epsilonCheck = 1e-5);

std::string printableImuMeasJacDelta(
    Ref<const MatX6> J,
    const std::vector<std::tuple<std::string, int, int>>& arrivalRanges,
    bool printBlocks = false,
    double epsilonCheck = 1e-5);

} // namespace visual_inertial_ba::preintegration
