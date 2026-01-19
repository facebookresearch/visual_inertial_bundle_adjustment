/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <session_data/SessionData.h>
#include <set>
#include <unordered_map>

namespace visual_inertial_ba {

class Matcher {
 public:
  void buildIndices(SessionData& fData, bool verbose = true);

  std::unordered_map<int64_t, int64_t> timestampToRigIndex;
  std::vector<int64_t> rigIndexToEvolvingStateIndex;
  std::vector<int64_t> rigIndexToCalibStateIndex;
  std::vector<int64_t> processedObsToRigIndex;
  std::set<int64_t> resetRigIndices;

  std::unordered_map<int64_t, int64_t> pointIdToPointIndex;
  std::vector<std::vector<int64_t>> pointIndexToObsIndices;

  std::vector<int> slamCamIndexToFactoryCalib;
  std::vector<int> slamImuIndexToFactoryCalib;
  std::vector<int> slamCamIndexToOnlineCalib;
  std::vector<int> slamImuIndexToOnlineCalib;
};

} // namespace visual_inertial_ba
