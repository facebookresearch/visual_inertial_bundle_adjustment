/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <imu_types/SignalStatistics.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>

namespace imu_types {

/**
 * IMU model parameters. All parameters describe how a true acceleration/angular velocity is
 * distorted into the measured reading. The model is defined as:
 * "true" = "compensated" = "undistorted"
 * "measured" = "uncompensated" = "distorted"
 *
 * a_measured = getAccelScaleMat() * (a_true + accelBiasMSec2)
 *            = accelScale * accelNonorth * (a_true + accelBiasMSec2)
 * w_measured = getGyroScaleMat() * (w_true + gyroBiasRadSec)
 *            = gyroScale * gyroNonorth * (w_true + gyroBiasRadSec)
 * This is implemented in getCompensatedImuMeasurement().
 */
struct ImuMeasurementModelParameters {
  /// The scale of each of the gyro axes. These factors map from a true angular velocity to the
  /// scaled measured omega.
  Eigen::Vector3d gyroScaleVec;

  /// The scale of each of the accel axes. These factors map from a true acceleration to the scaled
  /// measured acceleration.
  Eigen::Vector3d accelScaleVec;

  /// Additive bias, m/sec^2 in the rectified (aka compensated, calibrated) accel frame.
  Eigen::Vector3d accelBiasMSec2;

  /// Additive bias, rad/sec in the rectified gyro frame.
  Eigen::Vector3d gyroBiasRadSec;

  /// Nonorthogonality of accel axes. Each row of the matrix must be unit norm. This is upper
  /// triangular.
  Eigen::Matrix3d accelNonorth;

  /// Nonorthogonality of gyro axes. Each row of the matrix must be unit norm. This does not have to
  /// be upper triangular in case the gyroscope axis are slightly rotated with respect to the
  /// accelerometer axes.
  Eigen::Matrix3d gyroNonorth;

  // The time offsets of the recorded timestamps. This means that if we record a measurement at t,
  // this actually corresponds to the omega/specific force at t-time_offset.
  // If we estimte the time offset wrt. a reference clock (e.g. camera timestamps), we reconstruct
  // the timestamps
  //   tReference = tAccel - dtReferenceAccel
  //   tAccel = tReference + dtReferenceAccel
  double dtReferenceAccelSec;
  double dtReferenceGyroSec;

  ImuMeasurementModelParameters() {
    reset();
  }

  // Default copy & move ctors
  ImuMeasurementModelParameters(const ImuMeasurementModelParameters& other) = default;
  ImuMeasurementModelParameters& operator=(const ImuMeasurementModelParameters& other) = default;
  ImuMeasurementModelParameters(ImuMeasurementModelParameters&& other) noexcept = default;
  ImuMeasurementModelParameters& operator=(ImuMeasurementModelParameters&& other) noexcept =
      default;

  void reset() {
    // default values
    gyroNonorth = Eigen::Matrix3d::Identity();
    accelNonorth = Eigen::Matrix3d::Identity();

    accelBiasMSec2.setZero();
    gyroBiasRadSec.setZero();

    gyroScaleVec.setConstant(1.0);
    accelScaleVec.setConstant(1.0);

    dtReferenceAccelSec = 0.0;
    dtReferenceGyroSec = 0.0;
  }

  inline void getCompensatedImuMeasurement(
      const SignalStatistics& uncompensatedGyroRadSec,
      const SignalStatistics& uncompensatedAccelMSec2,
      Eigen::Vector3d& compensatedGyroRadSec,
      Eigen::Vector3d& compensatedAccelMSec2) const {
    const Eigen::Matrix3d accelScaleInv = getAccelScaleMat().inverse();
    const Eigen::Matrix3d gyroScaleInv = getGyroScaleMat().inverse();

    // compensated gyro first
    compensatedGyroRadSec.noalias() =
        gyroScaleInv * uncompensatedGyroRadSec.averageSignal - gyroBiasRadSec;
    compensatedAccelMSec2.noalias() =
        accelScaleInv * uncompensatedAccelMSec2.averageSignal - accelBiasMSec2;
  }

  /// Set the distortion matrices. accelScaleMat must be upper triangular.
  void setScaleMatrices(const Eigen::Matrix3d& gyroScaleMat, const Eigen::Matrix3d& accelScaleMat) {
    for (int i = 0; i < 3; i++) {
      gyroScaleVec(i) = gyroScaleMat.row(i).norm();
      gyroNonorth.row(i) = gyroScaleMat.row(i).normalized();

      accelScaleVec(i) = accelScaleMat.row(i).norm();
      accelNonorth.row(i) = accelScaleMat.row(i).normalized();
    }

    // make sure the accel nonorthogonality is upper triangular
    const double kEpsilon = 1e-14;
    if (std::abs(accelNonorth(1, 0)) > kEpsilon || std::abs(accelNonorth(2, 0)) > kEpsilon ||
        std::abs(accelNonorth(2, 1)) > kEpsilon) {
      throw std::runtime_error(
          "ImuMeasurementModelParameters::setScaleMatrices: accel should be upper triangular");
    }
  }

  /// These return the linear 'distortion' matrices. See documentation at the top for
  /// ImuMeasurementModelParameters.
  Eigen::Matrix3d getAccelScaleMat() const {
    return accelScaleVec.asDiagonal() * accelNonorth;
  }

  /// These return the linear 'distortion' matrices. See documentation at the top for
  /// ImuMeasurementModelParameters.
  Eigen::Matrix3d getGyroScaleMat() const {
    return gyroScaleVec.asDiagonal() * gyroNonorth;
  }
};

} // namespace imu_types
