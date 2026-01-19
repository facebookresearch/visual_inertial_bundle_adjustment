/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <preintegration/PreIntegration.h>

namespace visual_inertial_ba {

using ImuMeasurement = preintegration::ImuMeasurement;
using PreIntegrationData = preintegration::PreIntegration;
using preintegration::computePreIntegration;

} // namespace visual_inertial_ba
