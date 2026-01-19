/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::InertialFactors"
#include <logging/Log.h>

namespace visual_inertial_ba {

[[maybe_unused]]
constexpr int64_t kSmallIntervalForOmegaUs = 5'000; // 5ms should give a us a good enough estimate

void SingleSessionAdapter::generatePreintegration(
    std::optional<int> maybePrevRigIndex,
    int nextRigIndex,
    int imuIndex) {
  const auto& imuMeasurements = fData_.allImuMeasurements[imuIndex];

  auto& calibVar = prob_.imuCalib(maybePrevRigIndex ? *maybePrevRigIndex : nextRigIndex, imuIndex);
  const auto& measModel = calibVar.var.value.modelParams;
  const int fIdx = matcher_.slamImuIndexToFactoryCalib[imuIndex];
  const auto& noiseModel = fData_.factoryCalibration.imuNoiseModels[fIdx];
  const int64_t nextRigTimestampUs = prob_.inertialPose(nextRigIndex).timestampUs;
  const int64_t prevRigTimestampUs = maybePrevRigIndex
      ? prob_.inertialPose(*maybePrevRigIndex).timestampUs
      : (nextRigTimestampUs - kSmallIntervalForOmegaUs);

  recomputedPreInts_[{nextRigIndex, imuIndex}] = computePreIntegration(
      *calibVar.var.value.jacInd,
      imuMeasurements,
      measModel,
      noiseModel,
      prevRigTimestampUs,
      nextRigTimestampUs);
}

static constexpr int64_t kMaxTimeDistanceInertialFactorUs = 10'000'000; // 10s

int64_t SingleSessionAdapter::regenerateAllPreintegrationsFromImuMeasurements() {
  const auto sortedRigs = prob_.sortedRigIndices();

  for (int imuIndex = 0; imuIndex < numImus_; imuIndex++) {
    for (size_t i = 0; i < sortedRigs.size(); i++) {
      const int64_t nextRigIndex = sortedRigs[i];
      const int64_t nextRigTimestampUs = prob_.inertialPose(nextRigIndex).timestampUs;
      if (i == 0) {
        generatePreintegration(std::nullopt, nextRigIndex, imuIndex);
        continue;
      }

      const int64_t prevRigIndex = sortedRigs[i - 1];
      const int64_t prevRigTimestampUs = prob_.inertialPose(prevRigIndex).timestampUs;

      if (nextRigTimestampUs - prevRigTimestampUs > kMaxTimeDistanceInertialFactorUs) {
        generatePreintegration(std::nullopt, nextRigIndex, imuIndex);
        continue;
      }

      generatePreintegration(prevRigIndex, nextRigIndex, imuIndex);
    }
  }

  return recomputedPreInts_.size();
}

void SingleSessionAdapter::addInertialFactors(double imuLossRadius, double imuLossCutoff) {
  XR_LOGI("Adding inertial factors (nimus: {})", numImus_);

  // reset loss radius
  prob_.imuErrorLoss().setSize(imuLossRadius, imuLossCutoff);
  const auto sortedRigs = prob_.sortedRigIndices();

  int64_t nAdded = 0;
  for (int imuIndex = 0; imuIndex < numImus_; imuIndex++) {
    for (size_t i = 1; i < sortedRigs.size(); i++) {
      const int64_t prevRigIndex = sortedRigs[i - 1];
      const int64_t nextRigIndex = sortedRigs[i];
      const int64_t prevRigTimestampUs = prob_.inertialPose(prevRigIndex).timestampUs;
      const int64_t nextRigTimestampUs = prob_.inertialPose(nextRigIndex).timestampUs;

      if (nextRigTimestampUs - prevRigTimestampUs > kMaxTimeDistanceInertialFactorUs) {
        continue;
      }

      const auto& preintRef = findOrDie(recomputedPreInts_, {nextRigIndex, imuIndex});
      prob_.addInertialFactor(
          prevRigIndex, nextRigIndex, imuIndex, preintRef, prob_.imuErrorLoss());
      nAdded++;
    }
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added inertial factors: {}", nAdded);
  }
}

} // namespace visual_inertial_ba
