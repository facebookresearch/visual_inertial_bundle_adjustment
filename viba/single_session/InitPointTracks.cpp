/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/single_session/SingleSessionAdapter.h>
#include <viba/single_session/Triangulation.h>

#define DEFAULT_LOG_CHANNEL "ViBa::InitPointTracks"
#include <logging/Log.h>

namespace visual_inertial_ba {

// filter observations referencing rigs which will be optimized
void SingleSessionAdapter::filterPointObservations(
    std::vector<int64_t>& filteredObsIndices,
    const std::vector<int64_t>& obsIndices) {
  for (auto obsIndex : obsIndices) {
    const auto& procObs = fData_.trackingObservations[obsIndex];
    const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
    if (prob_.inertialPose_exists(rigIndex)) {
      filteredObsIndices.push_back(obsIndex);
    }
  }
}

void SingleSessionAdapter::initPointsFromObservations() {
  int64_t usedObs = 0, totalObs = 0, triedTracks = 0, skippedTracks = 0;
  std::vector<int64_t> filteredObsIndices;
  for (auto [ptId, ptIndex] : matcher_.pointIdToPointIndex) {
    XR_CHECK_LT(ptIndex, matcher_.pointIndexToObsIndices.size());
    filterPointObservations(filteredObsIndices, matcher_.pointIndexToObsIndices[ptIndex]);
    if (filteredObsIndices.size() < triangulation::kMinInlierObs) {
      skippedTracks += !filteredObsIndices.empty(); // some observations, but still too short track
      filteredObsIndices.clear();
      continue;
    }

    triedTracks++;
    totalObs += filteredObsIndices.size();

    constexpr int kSeedOffset = 1729; // to map point ids to ransac seeds
    auto maybeTriangulationResult = triangulatePoint(filteredObsIndices, ptId + kSeedOffset);
    if (maybeTriangulationResult) {
      usedObs += maybeTriangulationResult->inlierObservationIndices.size();
      int64_t varIndex = prob_.pointTrack_addNew(maybeTriangulationResult->point);
      XR_CHECK_EQ(varIndex, pointTrackObservations_.size());
      pointTrackObservations_.push_back(
          std::move(maybeTriangulationResult->inlierObservationIndices));
    }

    filteredObsIndices.clear();
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Created {} point tracks out of {} attemped ({} skipped / small overlap). Will use {}/{} observations",
        prob_.numPointTrackVariables(),
        triedTracks,
        skippedTracks,
        usedObs,
        totalObs);
  }
}

} // namespace visual_inertial_ba
