/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/Types.h>

// triangulation settings
namespace visual_inertial_ba::triangulation {

// num ransac iterations in triangulation
constexpr int kNumRansac = 10;

// max observation angular error to validate triangulation candidate
constexpr double kOutlierObservationRads = degreesToRadians(0.4);

// minimum number of inliers to accept a candidate
constexpr int kMinNumInliersInTriangulation = 2;

// min track length
constexpr int kMinInlierObs = 3;

// refinement 1
constexpr double kRefine1_outlierThreshold = 3.0;
constexpr bool kRefine1_skipOutliers = false;
constexpr int kRefine1_maxNumIterations = 3;
constexpr double kRefine1_loss_radius = 1.5;

// refinement 2
constexpr double kRefine2_outlierThreshold = 2.5;
constexpr bool kRefine2_skipOutliers = true;
constexpr int kRefine2_maxNumIterations = 3;
constexpr double kRefine2_loss_radius = 1.0;

// min track length after refinement
constexpr int kMinNumInliersAfterRefinement = 3;

// model rolling shutters (setting only used relevant during triangulation)
constexpr bool kModelRollingShutter = true;

} // namespace visual_inertial_ba::triangulation
