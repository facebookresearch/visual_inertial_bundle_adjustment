/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <imu_model/RandomWalkCov.h>

namespace visual_inertial_ba {

Eigen::VectorXd imuCalibRandomWalkCov(
    const ImuNoiseModelParameters& noiseParams,
    const ImuCalibrationJacobianIndices& jacInd,
    double dtSec) {
  Eigen::VectorXd qDiag(jacInd.getErrorStateSize());

  if (jacInd.accelBiasIdx() >= 0) {
    qDiag.template segment<3>(jacInd.accelBiasIdx()) =
        dtSec * noiseParams.accelBiasRandomWalkVarM2Sec4PerSec;
  }
  if (jacInd.gyroBiasIdx() >= 0) {
    qDiag.template segment<3>(jacInd.gyroBiasIdx()) =
        dtSec * noiseParams.gyroBiasRandomWalkVarRad2Sec2PerSec;
  }
  if (jacInd.gyroScaleIdx() >= 0) {
    qDiag.template segment<3>(jacInd.gyroScaleIdx()) =
        dtSec * noiseParams.gyroScaleRandomWalkVarPerSec;
  }
  if (jacInd.accelScaleIdx() >= 0) {
    qDiag.template segment<3>(jacInd.accelScaleIdx()) =
        dtSec * noiseParams.accelScaleRandomWalkVarPerSec;
  }
  if (jacInd.accelNonorthIdx() >= 0) {
    qDiag.template segment<3>(jacInd.accelNonorthIdx()) =
        dtSec * noiseParams.accelNonorthRandomWalkVarPerSec;
  }
  if (jacInd.gyroNonorthIdx() >= 0) {
    qDiag.template segment<6>(jacInd.gyroNonorthIdx()) =
        dtSec * noiseParams.gyroNonorthRandomWalkWVarPerSec;
  }
  if (jacInd.referenceImuTimeOffsetIdx() >= 0) {
    qDiag(jacInd.referenceImuTimeOffsetIdx()) =
        noiseParams.refImuTimeOffsetRandomWalkVarSec2PerSec * dtSec;
  }
  if (jacInd.gyroAccelTimeOffsetIdx() >= 0) {
    qDiag(jacInd.gyroAccelTimeOffsetIdx()) =
        noiseParams.gyroAccelTimeOffsetRandomWalkVarSec2PerSec * dtSec;
  }

  return qDiag;
}

Eigen::VectorXd imuCalibTurnonStdDev(
    const ImuNoiseModelParameters& noiseParams,
    const ImuCalibrationJacobianIndices& jacInd) {
  Eigen::VectorXd result(jacInd.getErrorStateSize());

  if (jacInd.gyroBiasIdx() >= 0) {
    result.template segment<3>(jacInd.gyroBiasIdx()) = noiseParams.gyroBiasTurnonStdRadSec;
  }
  if (jacInd.accelBiasIdx() >= 0) {
    result.template segment<3>(jacInd.accelBiasIdx()) = noiseParams.accelBiasTurnonStdMSec2;
  }
  if (jacInd.gyroScaleIdx() >= 0) {
    result.template segment<3>(jacInd.gyroScaleIdx()) = noiseParams.gyroScaleTurnonStd;
  }
  if (jacInd.accelScaleIdx() >= 0) {
    result.template segment<3>(jacInd.accelScaleIdx()) = noiseParams.accelScaleTurnonStd;
  }
  if (jacInd.gyroNonorthIdx() >= 0) {
    result.template segment<6>(jacInd.gyroNonorthIdx()) = noiseParams.gyroNonorthTurnonStd;
  }
  if (jacInd.accelNonorthIdx() >= 0) {
    result.template segment<3>(jacInd.accelNonorthIdx()) = noiseParams.accelNonorthTurnonStd;
  }
  if (jacInd.referenceImuTimeOffsetIdx() >= 0) {
    result(jacInd.referenceImuTimeOffsetIdx()) = noiseParams.refImuTimeOffsetTurnonStdSec;
  }
  if (jacInd.gyroAccelTimeOffsetIdx() >= 0) {
    result(jacInd.gyroAccelTimeOffsetIdx()) = noiseParams.gyroAccelTimeOffsetTurnonStdSec;
  }

  return result;
}

} // namespace visual_inertial_ba
