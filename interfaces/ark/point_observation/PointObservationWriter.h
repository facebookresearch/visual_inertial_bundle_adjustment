/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <filesystem>
#include <fstream>

#include <point_observation/PointObservation.h>

namespace visual_inertial_ba {

class PointObservationWriter {
 public:
  explicit PointObservationWriter(const std::filesystem::path& path);

  void write(const PointObservation& /*tObs*/);

 private:
  void writeHeader();

  std::ofstream writer_;
};

} // namespace visual_inertial_ba
