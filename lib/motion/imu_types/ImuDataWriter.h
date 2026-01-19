/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <filesystem>
#include <fstream>

#include <imu_types/ImuMeasurement.h>

namespace imu_types {

class ImuDataWriter {
 public:
  explicit ImuDataWriter(const std::filesystem::path& path);

  void write(const ImuMeasurement& meas);

 private:
  void writeHeader();

  std::ofstream writer_;
};

} // namespace imu_types
