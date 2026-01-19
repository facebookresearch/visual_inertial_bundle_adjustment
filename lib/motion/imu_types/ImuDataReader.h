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

struct ImuDataReader {
  static ImuMeasurements read(const std::filesystem::path& path);
  static ImuMeasurements read(const std::string& filename, std::istream& is);
};

} // namespace imu_types
