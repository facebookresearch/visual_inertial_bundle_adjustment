/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

namespace imu_types {

struct ImuNoiseModelParameters {
  // Initial bias standard deviations: accel, bias
  Eigen::Vector3d accelBiasTurnonStdMSec2;
  Eigen::Vector3d gyroBiasTurnonStdRadSec;

  // Initial scale standard deviations: accel, bias
  Eigen::Vector3d accelScaleTurnonStd;
  Eigen::Vector3d gyroScaleTurnonStd;

  // Initial nonorthogonality standard deviations: accel, bias
  // These are the standard deviations of the off-diagonal elements of the nonorthogonality matrices
  // These elements equal sin(nonorthogonality angle).
  Eigen::Vector3d accelNonorthTurnonStd;
  Eigen::Matrix<double, 6, 1> gyroNonorthTurnonStd;

  // the initial time offset uncertainty
  double gyroAccelTimeOffsetTurnonStdSec;
  double refImuTimeOffsetTurnonStdSec;

  // the uncertainty of the imu-imu transform (standard deviations in position and orientation)
  Eigen::Vector3d imuBodyImuTurnonPosStdM;
  Eigen::Vector3d imuBodyImuTurnonRotStdRad;

  // ---------------------------------------------------------
  // Time - dependent changes in the biases and scale factors
  // ---------------------------------------------------------
  // The power of the noise driving the bias random walk.
  // These values are the variance accumulated per unit time.
  Eigen::Vector3d gyroBiasRandomWalkVarRad2Sec2PerSec;
  Eigen::Vector3d accelBiasRandomWalkVarM2Sec4PerSec;

  // The power of the noise driving the scale random walk.
  // These values are the variance accumulated per unit time.
  Eigen::Vector3d accelScaleRandomWalkVarPerSec;
  Eigen::Vector3d gyroScaleRandomWalkVarPerSec;

  // the power of the noise driving the nonorthogonality random walk.
  // These values are the variance accumulated per unit time.
  Eigen::Vector3d accelNonorthRandomWalkVarPerSec;
  Eigen::Matrix<double, 6, 1> gyroNonorthRandomWalkWVarPerSec;

  // power of the noise driving the time offset random walks
  // These values are the variance accumulated per unit time.
  double refImuTimeOffsetRandomWalkVarSec2PerSec;
  double gyroAccelTimeOffsetRandomWalkVarSec2PerSec;

  // sample variance of the accel measurements noise. This is the variance of the noise affecting
  // each sample ("discrete time noise")
  Eigen::Vector3d accelSampleVarianceM2Sec4;
  // sample variance of the gyro measurement noise. This is the variance of the noise affecting each
  // sample ("discrete time noise")
  Eigen::Vector3d gyroSampleVarianceRad2Sec2;

  // the noise that we use if we model the imu-imu transform as a random walk
  Eigen::Vector3d imuBodyImuPosRandomWalkVarM2PerSec;
  Eigen::Vector3d imuBodyImuRotRandomWalkVarRad2PerSec;

  ImuNoiseModelParameters() {
    // set default values
    reset();
  }

  void reset() {
    // values fitting Aria glasses
    // accelSampleVarianceM2Sec4.setConstant(7.7951241e-3); // left-imu
    accelSampleVarianceM2Sec4.setConstant(6.6297049e-3); // right-imu
    gyroSampleVarianceRad2Sec2.setConstant(2.7415568e-05);

    // bias (turn-on std and random walk)
    accelBiasTurnonStdMSec2.setConstant(0.03);
    gyroBiasTurnonStdRadSec.setConstant(0.5 * 3.14159 / 180);
    accelBiasRandomWalkVarM2Sec4PerSec.setConstant(1e-8);
    gyroBiasRandomWalkVarRad2Sec2PerSec.setConstant(1e-10);

    // scale (turn-on std and random walk)
    accelScaleTurnonStd.setConstant(1e-3);
    gyroScaleTurnonStd.setConstant(1e-3);
    accelScaleRandomWalkVarPerSec.setConstant(1e-10);
    gyroScaleRandomWalkVarPerSec.setConstant(1e-10);

    // non-orth (turn-on std and random walk)
    accelNonorthTurnonStd.setConstant(0.2 * 3.14159 / 180);
    gyroNonorthTurnonStd.setConstant(0.2 * 3.14159 / 180);
    accelNonorthRandomWalkVarPerSec.setConstant(1e-12);
    gyroNonorthRandomWalkWVarPerSec.setConstant(1e-12);

    // time offsets (turn-on std and random walk)
    gyroAccelTimeOffsetTurnonStdSec = 0.001;
    refImuTimeOffsetTurnonStdSec = 0.001;
    gyroAccelTimeOffsetRandomWalkVarSec2PerSec = 1e-10;
    refImuTimeOffsetRandomWalkVarSec2PerSec = 1e-10;

    // imu-to-imu transform (turn-on std and random walk)
    imuBodyImuTurnonPosStdM.setConstant(0.001);
    imuBodyImuTurnonRotStdRad.setConstant(0.2 * 3.14159 / 180);
    imuBodyImuPosRandomWalkVarM2PerSec.setConstant(1e-10);
    imuBodyImuRotRandomWalkVarRad2PerSec.setConstant(1e-10 * 3.14159 / 180);
  }
};

} // namespace imu_types
