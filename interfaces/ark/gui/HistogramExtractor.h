/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gui/MonitoringState.h>
#include <small_thing/Optimizer.h>
#include <viba/common/Histogram.h>
#include <viba/problem/SingleSessionProblem.h>
#include <map>
#include <string>

namespace viba_gui {

// Helper to extract raw residual data from the optimizer
inline void extractHistograms(
    const visual_inertial_ba::SingleSessionProblem& problem,
    small_thing::Optimizer& opt,
    std::map<std::string, std::vector<double>>& residuals_by_type) {
  // Iterate through factor stores to extract residuals
  for (const auto& [tid, factorStore] : opt.factorStores.stores) {
    std::string name = factorStore->name();

    // Determine factor type
    std::string factor_type;

    if (name.find("VisualFactor") != std::string::npos) {
      factor_type = "Visual (Reprojection)";
    } else if (name.find("InertialFactor") != std::string::npos) {
      factor_type = "Inertial (IMU)";
    } else if (name.find("ImuCalibRWFactor") != std::string::npos) {
      factor_type = "IMU Calib (Random Walk)";
    } else if (name.find("CamIntrRWFactor") != std::string::npos) {
      factor_type = "Camera Intrinsics (Random Walk)";
    } else if (name.find("ExtrRWFactor") != std::string::npos) {
      factor_type = "Extrinsics (Random Walk)";
    } else if (name.find("OmegaPriorFactor") != std::string::npos) {
      factor_type = "Omega Prior";
    } else {
      // Skip unknown factor types
      continue;
    }

    // Collect raw residuals
    std::vector<double>& residuals = residuals_by_type[factor_type];
    const int64_t numFactors = factorStore->numCosts();
    residuals.reserve(numFactors);

    for (int64_t i = 0; i < numFactors; ++i) {
      // Get squared error for this factor
      double sqError = factorStore->unweightedSquaredError(i);
      double error = std::sqrt(sqError);
      residuals.push_back(error);
    }
  }
}

} // namespace viba_gui
