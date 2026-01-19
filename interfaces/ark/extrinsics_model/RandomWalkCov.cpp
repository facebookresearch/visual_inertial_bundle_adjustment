/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <extrinsics_model/RandomWalkCov.h>

namespace visual_inertial_ba {

Eigen::Vector<double, 6> imuExtrRandomWalkCov(
    const ImuNoiseModelParameters& noiseParams,
    double dtSec) {
  Eigen::Vector<double, 6> qDiag;
  qDiag.head<3>() = dtSec * noiseParams.imuBodyImuPosRandomWalkVarM2PerSec;
  qDiag.tail<3>() = dtSec * noiseParams.imuBodyImuRotRandomWalkVarRad2PerSec;
  return qDiag;
}

constexpr double kCamExtrRWVarRad2PerSec = 1e-11;
constexpr double kCamExtrRWVarM2PerSec = (1e-3 * M_PI / 180) * (1e-3 * M_PI / 180);

Eigen::Vector<double, 6> camExtrRandomWalkCov(double dtSec) {
  Eigen::Vector<double, 6> qDiag;
  qDiag.head<3>().array() = dtSec * kCamExtrRWVarM2PerSec;
  qDiag.tail<3>().array() = dtSec * kCamExtrRWVarRad2PerSec;
  return qDiag;
}

} // namespace visual_inertial_ba
