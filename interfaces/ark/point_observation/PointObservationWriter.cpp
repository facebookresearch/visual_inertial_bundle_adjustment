/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <point_observation/PointObservationWriter.h>

#include <point_observation/PointObservationFormat.h>

namespace visual_inertial_ba {
PointObservationWriter::PointObservationWriter(const std::filesystem::path& path) : writer_(path) {
  writeHeader();
}

void PointObservationWriter::writeHeader() {
  const auto& columns = PointObservationColumns;

  writer_ << columns.front();
  for (int i = 1; i < columns.size(); ++i) {
    writer_ << ',' << columns[i];
  }
  writer_ << std::endl;
}

void PointObservationWriter::write(const PointObservation& tObs) {
  writer_ << std::fixed;

  writer_ << tObs.pointId;
  writer_ << ',' << tObs.captureTimestampUs;
  writer_ << ',' << tObs.cameraIndex;

  writer_.precision(6);
  writer_ << ',' << tObs.projectionBaseRes.x();
  writer_ << ',' << tObs.projectionBaseRes.y();
  writer_ << ',' << tObs.sqrtH_BaseRes(0, 0);
  writer_ << ',' << tObs.sqrtH_BaseRes(0, 1);
  writer_ << ',' << tObs.sqrtH_BaseRes(1, 0);
  writer_ << ',' << tObs.sqrtH_BaseRes(1, 1);
  writer_ << std::endl;
}

} // namespace visual_inertial_ba
