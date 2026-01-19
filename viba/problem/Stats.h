/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Optimizer.h>
#include <viba/common/StatsValueContainer.h>
#include <viba/problem/Types.h>

namespace visual_inertial_ba {

struct StatsRef {
  StatsValueContainer* rotErrDeg = nullptr;
  StatsValueContainer* velErrCmS = nullptr;
  StatsValueContainer* posErrCm = nullptr;
  StatsValueContainer* imu = nullptr;

  StatsValueContainer* imuCalibRw = nullptr;
  StatsValueContainer* camIntrRw = nullptr;
  StatsValueContainer* imuExtrRw = nullptr;
  StatsValueContainer* camExtrRw = nullptr;
  StatsValueContainer* anyRw = nullptr;

  StatsValueContainer* imuCalibPrio = nullptr;
  StatsValueContainer* camIntrPrio = nullptr;
  StatsValueContainer* imuExtrPrio = nullptr;
  StatsValueContainer* camExtrPrio = nullptr;
  StatsValueContainer* anyPrio = nullptr;
};

void collectStats(small_thing::Optimizer& opt, const StatsRef& ref);

} // namespace visual_inertial_ba
