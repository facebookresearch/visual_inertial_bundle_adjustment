/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <imu_types/ImuDataReader.h>

#include <imu_types/ImuDataFormat.h>

#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD
#endif
#include <fast-cpp-csv-parser/csv.h>

namespace imu_types {
namespace {
ImuMeasurements readImpl(io::CSVReader<ImuDataFormat.size()>& csv) {
  // Read in the CSV header
  const auto readHeader = [&](auto&&... args) { csv.read_header(io::ignore_no_column, args...); };
  std::apply(readHeader, ImuDataFormat);

  ImuMeasurements imuMeasVec;
  ImuMeasurement meas;
  std::string tempC; // read as string, it might be "nan"
  while (csv.read_row(
      meas.timestampNs,
      tempC,
      meas.gyroRadSec[0],
      meas.gyroRadSec[1],
      meas.gyroRadSec[2],
      meas.accelMSec2[0],
      meas.accelMSec2[1],
      meas.accelMSec2[2])) {
    size_t nProc;
    const double tempC_val = std::stod(tempC, &nProc);
    meas.temperatureC =
        nProc == tempC.size() ? tempC_val : std::numeric_limits<double>::quiet_NaN();
    imuMeasVec.push_back(meas);
  }

  return imuMeasVec;
}
} // namespace

ImuMeasurements ImuDataReader::read(const std::filesystem::path& path) {
  io::CSVReader<ImuDataFormat.size()> csv(path.string());
  return readImpl(csv);
}

ImuMeasurements ImuDataReader::read(const std::string& filename, std::istream& is) {
  io::CSVReader<ImuDataFormat.size()> csv(filename, is);
  return readImpl(csv);
}

} // namespace imu_types
