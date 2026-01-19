/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <imu_types/ImuCalibrationOptions.h>

namespace imu_types {

// Structure which stores the indices of the different Imu Calibration Variables in the Jacobian. -1
// values correspond to unused variables.
struct ImuCalibrationJacobianIndices {
  // Max columns of the jacobian.
  static constexpr int kMaxCalibrationStateSize = 23;

  // The calibration param indices (e.g. gyroBiasIdx()) correspond to columns in the jacobian.
  // The following three constants correspond to the estimated values, i.e. the rows of the
  // jacobian.
  static constexpr int kRotationRowIdx = 0;
  static constexpr int kVelocityRowIdx = 3;
  static constexpr int kPositionRowIdx = 6;

  ImuCalibrationJacobianIndices() = default;

  explicit ImuCalibrationJacobianIndices(const ImuCalibrationOptions& estimationOptions) {
    computeIndices(estimationOptions);
  }

  // Computes the indices for the Jacobians given the ImuCalibrationOptions.
  void computeIndices(const ImuCalibrationOptions& estimationOptions) {
    int runningIdx = 0;
    if (estimationOptions.gyroBias) {
      gyroBiasIdx_ = runningIdx;
      runningIdx += 3;
    } else {
      gyroBiasIdx_ = -1;
    }

    if (estimationOptions.accelBias) {
      accelBiasIdx_ = runningIdx;
      runningIdx += 3;
    } else {
      accelBiasIdx_ = -1;
    }

    if (estimationOptions.gyroScale) {
      gyroScaleIdx_ = runningIdx;
      runningIdx += 3;
    } else {
      gyroScaleIdx_ = -1;
    }

    if (estimationOptions.accelScale) {
      accelScaleIdx_ = runningIdx;
      runningIdx += 3;
    } else {
      accelScaleIdx_ = -1;
    }

    if (estimationOptions.gyroNonOrth) {
      gyroNonorthIdx_ = runningIdx;
      runningIdx += 6;
    } else {
      gyroNonorthIdx_ = -1;
    }

    if (estimationOptions.accelNonOrth) {
      accelNonorthIdx_ = runningIdx;
      runningIdx += 3;
    } else {
      accelNonorthIdx_ = -1;
    }

    if (estimationOptions.referenceImuTimeOffset) {
      referenceImuTimeOffsetIdx_ = runningIdx;
      runningIdx += 1;
    } else {
      referenceImuTimeOffsetIdx_ = -1;
    }

    if (estimationOptions.gyroAccelTimeOffset) {
      gyroAccelTimeOffsetIdx_ = runningIdx;
      runningIdx += 1;
    } else {
      gyroAccelTimeOffsetIdx_ = -1;
    }

    errorStateSize_ = runningIdx;
  }

  // Recreeate the estimation options that these Jacobian indices were constructed from.
  ImuCalibrationOptions getEstimationOptions() const {
    ImuCalibrationOptions options(
        accelBiasIdx() >= 0,
        gyroBiasIdx() >= 0,
        accelScaleIdx() >= 0,
        gyroScaleIdx() >= 0,
        gyroNonorthIdx() >= 0,
        accelNonorthIdx() >= 0,
        referenceImuTimeOffsetIdx() >= 0,
        gyroAccelTimeOffsetIdx() >= 0);
    return options;
  }

  // Accessors to the private index variables.

  int gyroBiasIdx() const {
    return gyroBiasIdx_;
  }
  int accelBiasIdx() const {
    return accelBiasIdx_;
  }
  int gyroScaleIdx() const {
    return gyroScaleIdx_;
  }
  int accelScaleIdx() const {
    return accelScaleIdx_;
  }
  int gyroNonorthIdx() const {
    return gyroNonorthIdx_;
  }
  int accelNonorthIdx() const {
    return accelNonorthIdx_;
  }
  int referenceImuTimeOffsetIdx() const {
    return referenceImuTimeOffsetIdx_;
  }
  int gyroAccelTimeOffsetIdx() const {
    return gyroAccelTimeOffsetIdx_;
  }

  int getErrorStateSize() const {
    return errorStateSize_;
  }

  bool operator==(const ImuCalibrationJacobianIndices& other) const {
    return (
        (gyroBiasIdx_ == other.gyroBiasIdx_) && (accelBiasIdx_ == other.accelBiasIdx_) &&
        (accelScaleIdx_ == other.accelScaleIdx_) && (gyroScaleIdx_ == other.gyroScaleIdx_) &&
        (gyroNonorthIdx_ == other.gyroNonorthIdx_) &&
        (accelNonorthIdx_ == other.accelNonorthIdx_) &&
        (referenceImuTimeOffsetIdx_ == other.referenceImuTimeOffsetIdx_) &&
        (gyroAccelTimeOffsetIdx_ == other.gyroAccelTimeOffsetIdx_) &&
        (errorStateSize_ == other.errorStateSize_));
  }

  // Only checking the valid indices to see if everything is equivalent.
  bool isCompatibleWith(const ImuCalibrationJacobianIndices& other) const {
    if (gyroBiasIdx() >= 0 && (other.gyroBiasIdx() != gyroBiasIdx())) {
      return false;
    }
    if (accelBiasIdx() >= 0 && (other.accelBiasIdx() != accelBiasIdx())) {
      return false;
    }
    if (accelScaleIdx() >= 0 && (other.accelScaleIdx() != accelScaleIdx())) {
      return false;
    }
    if (gyroScaleIdx() >= 0 && (other.gyroScaleIdx() != gyroScaleIdx())) {
      return false;
    }
    if (accelNonorthIdx() >= 0 && (other.accelNonorthIdx() != accelNonorthIdx())) {
      return false;
    }
    if (gyroNonorthIdx() >= 0 && (other.gyroNonorthIdx() != gyroNonorthIdx())) {
      return false;
    }

    if (referenceImuTimeOffsetIdx() >= 0 &&
        (other.referenceImuTimeOffsetIdx() != referenceImuTimeOffsetIdx())) {
      return false;
    }

    if (gyroAccelTimeOffsetIdx() >= 0 &&
        (other.gyroAccelTimeOffsetIdx() != gyroAccelTimeOffsetIdx())) {
      return false;
    }

    return true;
  }

 private:
  int gyroBiasIdx_ = -1;
  int accelBiasIdx_ = -1;
  int gyroScaleIdx_ = -1;
  int accelScaleIdx_ = -1;
  int gyroNonorthIdx_ = -1;
  int accelNonorthIdx_ = -1;
  int referenceImuTimeOffsetIdx_ = -1;
  int gyroAccelTimeOffsetIdx_ = -1;

  int errorStateSize_ = 0;
};

} // namespace imu_types
