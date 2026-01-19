/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <camera_model/RandomWalkCov.h>

namespace visual_inertial_ba {

constexpr double kCamIntrinsicsProgRWVarPerSec = 1e-6;
constexpr double kCamIntrinsicsDistRWVarPerSec = 1e-10;
constexpr double kReadoutTimeRWVarSec2PerSec = 1e-10;

Eigen::VectorXd cameraModelRandomWalkCov(const CameraModelParam& cameraModelParam, double dtSec) {
  const int nProjParams = cameraModelParam.model.numProjectionParameters();
  const int nDistParams = cameraModelParam.model.numDistortionParameters();
  const int nTimeOffsetParams = (cameraModelParam.estimateReadoutTime ? 1 : 0) +
      (cameraModelParam.estimateTimeOffset ? 1 : 0);
  const int nParams = nProjParams + nDistParams + nTimeOffsetParams;

  Eigen::VectorXd qDiag = Eigen::VectorXd::Zero(nParams);
  qDiag.head(nProjParams).array() = kCamIntrinsicsProgRWVarPerSec * dtSec;
  qDiag.segment(nProjParams, nDistParams).array() = kCamIntrinsicsDistRWVarPerSec * dtSec;
  qDiag.tail(nTimeOffsetParams).array() = kReadoutTimeRWVarSec2PerSec * dtSec;

  return qDiag;
}

// Camera intrinsics turn-on standard deviations
constexpr double kCamIntrinsicsProjectionParamsTurnonStd = 1.0;
constexpr double kCamIntrinsicsDistortionParamsTurnonStd = 1e-3;
constexpr double kCamIntrinsicsReadoutTimeTurnonStdSec = 0.01;
constexpr double kCamIntrinsicsTimeOffsetTurnonStdSec = 0.01;

Eigen::VectorXd cameraModelTurnOnStdDev(const CameraModelParam& cameraModelParam) {
  const int nProjParams = cameraModelParam.model.numProjectionParameters();
  const int nDistParams = cameraModelParam.model.numDistortionParameters();
  const int nTimeOffsetParams = (cameraModelParam.estimateReadoutTime ? 1 : 0) +
      (cameraModelParam.estimateTimeOffset ? 1 : 0);
  const int nParams = nProjParams + nDistParams + nTimeOffsetParams;

  Eigen::VectorXd turnOnStdDev(nParams);
  turnOnStdDev.head(nProjParams).setConstant(kCamIntrinsicsProjectionParamsTurnonStd);
  turnOnStdDev.segment(nProjParams, nDistParams)
      .setConstant(kCamIntrinsicsDistortionParamsTurnonStd);
  if (cameraModelParam.estimateReadoutTime) {
    turnOnStdDev[nProjParams + nDistParams] = kCamIntrinsicsReadoutTimeTurnonStdSec;
  }
  if (cameraModelParam.estimateTimeOffset) {
    turnOnStdDev[nParams - 1] = kCamIntrinsicsTimeOffsetTurnonStdSec;
  }

  return turnOnStdDev;
}

} // namespace visual_inertial_ba
