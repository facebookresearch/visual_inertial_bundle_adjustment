/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <imu_types/ImuDataWriter.h>

#include <imu_types/ImuDataFormat.h>

namespace imu_types {
ImuDataWriter::ImuDataWriter(const std::filesystem::path& path) : writer_(path) {
  writeHeader();
}

void ImuDataWriter::writeHeader() {
  const auto& columns = ImuDataFormat;

  writer_ << columns.front();
  for (int i = 1; i < columns.size(); ++i) {
    writer_ << ", " << columns[i];
  }
  writer_ << std::endl;
}

void ImuDataWriter::write(const ImuMeasurement& meas) {
  writer_ << std::fixed;
  writer_.precision(7);

  writer_ << meas.timestampNs;
  writer_ << ", " << meas.temperatureC;
  writer_ << ", " << meas.gyroRadSec[0];
  writer_ << ", " << meas.gyroRadSec[1];
  writer_ << ", " << meas.gyroRadSec[2];
  writer_ << ", " << meas.accelMSec2[0];
  writer_ << ", " << meas.accelMSec2[1];
  writer_ << ", " << meas.accelMSec2[2];
  writer_ << std::endl;
}

} // namespace imu_types
