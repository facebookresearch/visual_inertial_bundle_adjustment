/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <point_observation/PointObservationReader.h>

#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD
#endif
#include <fast-cpp-csv-parser/csv.h>

#include <point_observation/PointObservationFormat.h>

namespace visual_inertial_ba {
namespace {
PointObservations readImpl(io::CSVReader<PointObservationColumns.size()>& csv) {
  // Read in the CSV header
  const auto readHeader = [&](auto&&... args) { csv.read_header(io::ignore_no_column, args...); };
  std::apply(readHeader, PointObservationColumns);

  PointObservations trackingObservations;

  PointObservation tObs;
  while (csv.read_row(
      tObs.pointId,
      tObs.captureTimestampUs,
      tObs.cameraIndex,
      tObs.projectionBaseRes.x(),
      tObs.projectionBaseRes.y(),
      tObs.sqrtH_BaseRes(0, 0),
      tObs.sqrtH_BaseRes(0, 1),
      tObs.sqrtH_BaseRes(1, 0),
      tObs.sqrtH_BaseRes(1, 1))) {
    trackingObservations.push_back(tObs);
  }

  return trackingObservations;
}
} // namespace

PointObservations PointObservationReader::read(const std::filesystem::path& path) {
  io::CSVReader<PointObservationColumns.size()> csv(path.string());
  return readImpl(csv);
}
PointObservations PointObservationReader::read(const std::string& filename, std::istream& is) {
  io::CSVReader<PointObservationColumns.size()> csv(filename, is);
  return readImpl(csv);
}
} // namespace visual_inertial_ba
