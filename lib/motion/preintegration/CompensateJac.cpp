/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <preintegration/CompensateJac.h>

namespace visual_inertial_ba::preintegration {

void boxPlus(
    ImuMeasurementModelParameters& modelParams,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<const VecX> correction) {
  if (correction.size() != jacInd.getErrorStateSize()) {
    throw std::runtime_error("boxPlus: incorrect Jacobian size");
  }

  if (jacInd.gyroBiasIdx() >= 0) {
    modelParams.gyroBiasRadSec += correction.template segment<3>(jacInd.gyroBiasIdx());
  }

  if (jacInd.accelBiasIdx() >= 0) {
    modelParams.accelBiasMSec2 += correction.template segment<3>(jacInd.accelBiasIdx());
  }

  if (jacInd.gyroScaleIdx() >= 0) {
    Eigen::Vector3d invScale = modelParams.gyroScaleVec.cwiseInverse();
    invScale += correction.template segment<3>(jacInd.gyroScaleIdx());
    modelParams.gyroScaleVec = invScale.cwiseInverse();
  }

  if (jacInd.accelScaleIdx() >= 0) {
    Eigen::Vector3d invScale = modelParams.accelScaleVec.cwiseInverse();
    invScale += correction.template segment<3>(jacInd.accelScaleIdx());
    modelParams.accelScaleVec = invScale.cwiseInverse();
  }

  if (jacInd.gyroNonorthIdx() >= 0) {
    modelParams.gyroNonorth.row(0).template segment<2>(1) +=
        correction.template segment<2>(jacInd.gyroNonorthIdx());
    modelParams.gyroNonorth(1, 0) += correction(jacInd.gyroNonorthIdx() + 2);
    modelParams.gyroNonorth(1, 2) += correction(jacInd.gyroNonorthIdx() + 3);
    modelParams.gyroNonorth.row(2).template segment<2>(0) +=
        correction.template segment<2>(jacInd.gyroNonorthIdx() + 4);

    modelParams.gyroNonorth(0, 0) =
        std::sqrt(1.0 - modelParams.gyroNonorth.row(0).template segment<2>(1).squaredNorm());
    modelParams.gyroNonorth(1, 1) = std::sqrt(
        1.0 - modelParams.gyroNonorth(1, 0) * modelParams.gyroNonorth(1, 0) -
        modelParams.gyroNonorth(1, 2) * modelParams.gyroNonorth(1, 2));
    modelParams.gyroNonorth(2, 2) =
        std::sqrt(1.0 - modelParams.gyroNonorth.row(2).template segment<2>(0).squaredNorm());
  }

  if (jacInd.accelNonorthIdx() >= 0) {
    modelParams.accelNonorth.row(0).template segment<2>(1) +=
        correction.template segment<2>(jacInd.accelNonorthIdx());
    modelParams.accelNonorth(1, 2) += correction(jacInd.accelNonorthIdx() + 2);

    modelParams.accelNonorth(0, 0) =
        std::sqrt(1.0 - modelParams.accelNonorth.row(0).template segment<2>(1).squaredNorm());

    modelParams.accelNonorth(1, 1) =
        std::sqrt(1.0 - modelParams.accelNonorth(1, 2) * modelParams.accelNonorth(1, 2));
    modelParams.accelNonorth(2, 2) = 1.0;
  }

  if (jacInd.referenceImuTimeOffsetIdx() >= 0) {
    modelParams.dtReferenceGyroSec += correction(jacInd.referenceImuTimeOffsetIdx());
    modelParams.dtReferenceAccelSec += correction(jacInd.referenceImuTimeOffsetIdx());
  }

  if (jacInd.gyroAccelTimeOffsetIdx() >= 0) {
    modelParams.dtReferenceAccelSec += correction(jacInd.gyroAccelTimeOffsetIdx());
  }
}

// compute the residual to another state (aka boxMinus)
void boxMinus(
    const ImuMeasurementModelParameters& modelParams,
    const ImuMeasurementModelParameters& refModelParams,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<VecX> res) {
  if (res.size() != jacInd.getErrorStateSize()) {
    throw std::runtime_error("boxMinus: incorrect Jacobian size");
  }

  if (jacInd.gyroBiasIdx() >= 0) {
    res.template segment<3>(jacInd.gyroBiasIdx()) =
        modelParams.gyroBiasRadSec - refModelParams.gyroBiasRadSec;
  }

  if (jacInd.accelBiasIdx() >= 0) {
    res.template segment<3>(jacInd.accelBiasIdx()) =
        modelParams.accelBiasMSec2 - refModelParams.accelBiasMSec2;
  }

  if (jacInd.gyroScaleIdx() >= 0) {
    res.template segment<3>(jacInd.gyroScaleIdx()) =
        modelParams.gyroScaleVec.cwiseInverse() - refModelParams.gyroScaleVec.cwiseInverse();
  }

  if (jacInd.accelScaleIdx() >= 0) {
    res.template segment<3>(jacInd.accelScaleIdx()) =
        modelParams.accelScaleVec.cwiseInverse() - refModelParams.accelScaleVec.cwiseInverse();
  }

  if (jacInd.gyroNonorthIdx() >= 0) {
    res.template segment<2>(jacInd.gyroNonorthIdx()) =
        modelParams.gyroNonorth.row(0).template segment<2>(1) -
        refModelParams.gyroNonorth.row(0).template segment<2>(1);

    res(jacInd.gyroNonorthIdx() + 2) =
        modelParams.gyroNonorth(1, 0) - refModelParams.gyroNonorth(1, 0);
    res(jacInd.gyroNonorthIdx() + 3) =
        modelParams.gyroNonorth(1, 2) - refModelParams.gyroNonorth(1, 2);

    res.template segment<2>(jacInd.gyroNonorthIdx() + 4) =
        modelParams.gyroNonorth.row(2).template segment<2>(0) -
        refModelParams.gyroNonorth.row(2).template segment<2>(0);
  }

  if (jacInd.accelNonorthIdx() >= 0) {
    res.template segment<2>(jacInd.accelNonorthIdx()) =
        modelParams.accelNonorth.row(0).template segment<2>(1) -
        refModelParams.accelNonorth.row(0).template segment<2>(1);

    res(jacInd.accelNonorthIdx() + 2) =
        modelParams.accelNonorth(1, 2) - refModelParams.accelNonorth(1, 2);
  }

  if (jacInd.referenceImuTimeOffsetIdx() >= 0) {
    res(jacInd.referenceImuTimeOffsetIdx()) =
        modelParams.dtReferenceGyroSec - refModelParams.dtReferenceGyroSec;
  }

  if (jacInd.gyroAccelTimeOffsetIdx() >= 0) {
    res(jacInd.gyroAccelTimeOffsetIdx()) =
        (modelParams.dtReferenceAccelSec - modelParams.dtReferenceGyroSec) -
        (refModelParams.dtReferenceAccelSec - refModelParams.dtReferenceGyroSec);
  }
}

void getCompensatedImuMeasurementAndJac(
    const ImuMeasurementModelParameters& p,
    const SignalStatistics& uncompensatedGyroRadSec,
    const SignalStatistics& uncompensatedAccelMSec2,
    Vec3& compensatedGyroRadSec,
    Vec3& compensatedAccelMSec2,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<Mat6X> calibJac,
    Ref<Mat66> measJac) {
  if (calibJac.cols() != jacInd.getErrorStateSize()) {
    throw std::runtime_error("getCompensatedImuMeasurementAndJac: incorrect Jacobian size");
  }
  calibJac.setZero();

  const Vec3 accelScaleInvVec = p.accelScaleVec.cwiseInverse();
  const Mat3 accelNonOrthInv = p.accelNonorth.inverse();
  const Mat3 accelScaleMat = accelNonOrthInv * accelScaleInvVec.asDiagonal();
  const Vec3 gyroScaleInvVec = p.gyroScaleVec.cwiseInverse();
  const Mat3 gyroNonOrthInv = p.gyroNonorth.inverse();
  const Mat3 gyroScaleMat = gyroNonOrthInv * gyroScaleInvVec.asDiagonal();

  Vec3 gyroLinear = uncompensatedGyroRadSec.averageSignal;
  Mat3 dGyroLinear_dGyroRaw = Mat3::Identity();

  Vec3 accelLinear = uncompensatedAccelMSec2.averageSignal;
  Mat3 dAccelLinear_dAccelRaw = Mat3::Identity();

  if (jacInd.gyroScaleIdx() >= 0) {
    calibJac.block<3, 3>(0, jacInd.gyroScaleIdx()).noalias() =
        gyroNonOrthInv * gyroLinear.asDiagonal();
  }

  /*
    N(p) * Ninv(p) * g = g
    DN(p) * Ninv * g + N * DNinv(p) * g = 0
    DNinv(p) * g = -NInv * DN(p) * Ninv * g
  */
  const Vec3 scaledGyro = gyroScaleMat * gyroLinear;
  if (jacInd.gyroNonorthIdx() >= 0) {
    static constexpr int kRows[] = {0, 0, 1, 1, 2, 2};
    static constexpr int kCols[] = {1, 2, 0, 2, 0, 1};

    for (int i = 0; i < 6; i++) {
      const int r = kRows[i], c = kCols[i];

      const double dNrr_dpI = -p.gyroNonorth(r, c) / p.gyroNonorth(r, r);
      calibJac.block<3, 1>(0, jacInd.gyroNonorthIdx() + i).noalias() =
          -gyroNonOrthInv.col(r) * (scaledGyro[r] * dNrr_dpI + scaledGyro[c]);
    }
  }
  const Mat3 dScaleGyro_dGyroRaw = gyroScaleMat * dGyroLinear_dGyroRaw;

  // compensated gyro first
  compensatedGyroRadSec.noalias() = scaledGyro - p.gyroBiasRadSec;

  measJac.block<3, 3>(0, 0).noalias() = dScaleGyro_dGyroRaw;
  measJac.block<3, 3>(0, 3).setZero();

  if (jacInd.gyroBiasIdx() >= 0) {
    calibJac.block<3, 3>(0, jacInd.gyroBiasIdx()).noalias() = -Mat3::Identity();
  }

  if (jacInd.accelScaleIdx() >= 0) {
    // tweak possibly added to tempAccel is accelScale-neutral
    calibJac.block<3, 3>(3, jacInd.accelScaleIdx()).noalias() =
        accelNonOrthInv * accelLinear.asDiagonal();
  }

  // start with the uncompensated signal
  Vec3 tempAccel = accelLinear;

  // tweak to scaledAccel to compute accelNonOrth's calibJac
  Vec3 scaledAccelAdd_dNonOrth2 = Vec3::Zero();

  const Mat3 dScaleAccel_dAccelRaw = accelScaleMat * dAccelLinear_dAccelRaw;

  measJac.block<3, 3>(3, 0).setZero();
  measJac.block<3, 3>(3, 3).noalias() = dScaleAccel_dAccelRaw;

  const Vec3 scaledAccel = accelScaleMat * tempAccel;

  if (jacInd.accelNonorthIdx() >= 0) {
    static constexpr int kRows[] = {0, 0, 1};
    static constexpr int kCols[] = {1, 2, 2};

    for (int i = 0; i < 3; i++) {
      const int r = kRows[i], c = kCols[i];

      Vec3 s = scaledAccel;
      if (i == 2) {
        s += scaledAccelAdd_dNonOrth2;
      }
      const double dNrr_dpI = -p.accelNonorth(r, c) / p.accelNonorth(r, r);
      calibJac.block<3, 1>(3, jacInd.accelNonorthIdx() + i).noalias() =
          -accelNonOrthInv.col(r) * (s[r] * dNrr_dpI + s[c]);
    }
  }

  compensatedAccelMSec2.noalias() = scaledAccel - p.accelBiasMSec2;

  if (jacInd.accelBiasIdx() >= 0) {
    calibJac.block<3, 3>(3, jacInd.accelBiasIdx()).noalias() = -Mat3::Identity();
  }
}

} // namespace visual_inertial_ba::preintegration
