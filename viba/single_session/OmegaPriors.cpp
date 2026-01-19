/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::OmegaPriors"
#include <logging/Log.h>

namespace visual_inertial_ba {

// We add priors on omega (angular velocity) coming from all IMUs
// The value we are conditioning gyro's calibration during preintegration, if you want to avoid
// this bias you can recompute preintegration at each optimizer iteration.
void SingleSessionAdapter::addOmegaPriors() {
  if (numImus_ <= 1 && verbosity_ != Muted) {
    XR_LOGI("Num imus={}, not adding omega priors", numImus_);
    return;
  }

  for (const auto& [rigIndex, p] : prob_.inertialPoses()) {
    for (int imuIndex = 0; imuIndex < numImus_; imuIndex++) {
      const auto& preintRef = findOrDie(recomputedPreInts_, {rigIndex, imuIndex});
      prob_.addOmegaPriorFactor(rigIndex, imuIndex, preintRef.omegaAtEnd);
    }
  }
}

} // namespace visual_inertial_ba
