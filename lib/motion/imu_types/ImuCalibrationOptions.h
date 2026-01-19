/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace imu_types {

// Structure that indicates which Imu calibration options are used.
struct ImuCalibrationOptions {
  ImuCalibrationOptions() = default;

  // total number of estimation option combinations (used for tests): 2^8
  static constexpr size_t kNumTotalCombinations = 256;

  explicit ImuCalibrationOptions(bool option)
      : accelBias(option),
        gyroBias(option),
        accelScale(option),
        gyroScale(option),
        gyroNonOrth(option),
        accelNonOrth(option),
        referenceImuTimeOffset(option),
        gyroAccelTimeOffset(option) {}

  ImuCalibrationOptions(
      bool accelB,
      bool gyroB,
      bool accelSc,
      bool gyroSc,
      bool gyroNonO,
      bool accelNonO,
      bool refImuTimeOffset,
      bool gyroAccelTimeOffset)
      : accelBias(accelB),
        gyroBias(gyroB),
        accelScale(accelSc),
        gyroScale(gyroSc),
        gyroNonOrth(gyroNonO),
        accelNonOrth(accelNonO),
        referenceImuTimeOffset(refImuTimeOffset),
        gyroAccelTimeOffset(gyroAccelTimeOffset) {}

  // returns ImuCalibrationOptions with the complementary (opposite) settings
  ImuCalibrationOptions complementary() const {
    return ImuCalibrationOptions(
        !accelBias,
        !gyroBias,
        !accelScale,
        !gyroScale,
        !gyroNonOrth,
        !accelNonOrth,
        !referenceImuTimeOffset,
        !gyroAccelTimeOffset);
  }

  // helper function used for tests
  static ImuCalibrationOptions onlyBiases() {
    return ImuCalibrationOptions(true, true, false, false, false, false, false, false);
  }

  // helper function used for tests
  static ImuCalibrationOptions allExceptTimeOffsets() {
    return ImuCalibrationOptions(true, true, true, true, true, true, false, false);
  }

  // Helper function for sweeping through all estimation options for testing. It treats an integer i
  // as a bitfield, where the integer ranges from 0 to kNumTotalCombinations
  static ImuCalibrationOptions getTestEstimationOptions(size_t i) {
    return ImuCalibrationOptions(
        (i / 1 % 2),
        (i / 2 % 2),
        (i / 4 % 2),
        (i / 8 % 2),
        (i / 16 % 2),
        (i / 32 % 2),
        (i / 64 % 2),
        (i / 128 % 2));
  }

  // Helper function to get the error state size for a given option
  int errorStateSize() const {
    int errorStateSize = 0;

    errorStateSize += gyroBias ? 3 : 0;
    errorStateSize += accelBias ? 3 : 0;
    errorStateSize += gyroScale ? 3 : 0;
    errorStateSize += accelScale ? 3 : 0;
    errorStateSize += gyroNonOrth ? 6 : 0;
    errorStateSize += accelNonOrth ? 3 : 0;
    errorStateSize += referenceImuTimeOffset ? 1 : 0;
    errorStateSize += gyroAccelTimeOffset ? 1 : 0;

    return errorStateSize;
  }

  bool accelBias = false;
  bool gyroBias = false;
  bool accelScale = false;
  bool gyroScale = false;
  bool gyroNonOrth = false;
  bool accelNonOrth = false;
  bool referenceImuTimeOffset = false;
  bool gyroAccelTimeOffset = false;
};

} // namespace imu_types
