/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>

namespace visual_inertial_ba {

Eigen::VectorXd cameraModelRandomWalkCov(const CameraModelParam& cameraModelParam, double dtSec);

Eigen::VectorXd cameraModelTurnOnStdDev(const CameraModelParam& cameraModelParam);

} // namespace visual_inertial_ba
