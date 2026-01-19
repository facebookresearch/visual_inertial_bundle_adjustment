/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <point_observation/PointObservation.h>
#include <filesystem>

namespace visual_inertial_ba {

struct PointObservationReader {
  static PointObservations read(const std::filesystem::path& path);
  static PointObservations read(const std::string& filename, std::istream& is);
};

} // namespace visual_inertial_ba
