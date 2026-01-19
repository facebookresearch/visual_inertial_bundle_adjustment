/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::VisualFactors"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionAdapter::addVisualFactors(
    double trackingObsLossRadius,
    double trackingObsLossCutoff,
    bool optimizeDetectorBias) {
  // reset loss radius
  prob_.reprojErrorLoss().setSize(trackingObsLossRadius, trackingObsLossCutoff);

  if (optimizeDetectorBias) {
    throw std::runtime_error("Detector bias currently unsupported");
    // prob_.detectorBiases_init(numCameras_);
  }

  int64_t nAdded = 0;
  for (size_t ptVarIndex = 0; ptVarIndex < prob_.numPointTrackVariables(); ptVarIndex++) {
    for (const auto& obsIndex : pointTrackObservations_[ptVarIndex]) {
      const auto& procObs = fData_.trackingObservations[obsIndex];
      const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
      const int64_t cameraIndex = procObs.cameraIndex;
      auto& pointTrackVar = prob_.pointTrack(ptVarIndex);
      Vec2 projBaseRes = procObs.projectionBaseRes.cast<double>();
      Mat22 sqrtH_BaseRes = procObs.sqrtH_BaseRes.cast<double>();

      if (optimizeDetectorBias) {
        prob_.addVisualFactorWithBias(
            rigIndex,
            cameraIndex,
            pointTrackVar,
            prob_.detectorBias(cameraIndex),
            projBaseRes,
            sqrtH_BaseRes,
            prob_.reprojErrorLoss());
      } else {
        prob_.addVisualFactor(
            rigIndex,
            cameraIndex,
            pointTrackVar,
            projBaseRes,
            sqrtH_BaseRes,
            prob_.reprojErrorLoss());
      }
      nAdded++;
    }
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added (tracking) visual factors: {}", nAdded);
  }
}

} // namespace visual_inertial_ba
