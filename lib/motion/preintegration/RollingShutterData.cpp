/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <preintegration/RollingShutterData.h>

#include <fmt/format.h>
#include <preintegration/PreIntegration.h>
#include <iostream>

namespace visual_inertial_ba::preintegration {

void RollingShutterData::compute(
    int64_t midpointTimeUs,
    const ImuMeasurementModelParameters& measModel,
    const Vec3& gravityWorld) {
  midpointTimeUs_ = midpointTimeUs;
  gravityWorld_ = gravityWorld;

  sampledRvp_.clear();
  forEachIntegratedMeasurement(
      imuMeas_,
      measModel,
      midpointTimeUs - intervalHalfLengthUs_,
      midpointTimeUs,
      [&](const RotVelPos& rvp, bool atAccelBoundary, bool atGyroBoundary) {
        if (atGyroBoundary) {
          sampledRvp_.push_back(rvp);
        }
      });

  // adjust samples for negative times so that it's relative to midpoint
  RotVelPos start_to_mid = sampledRvp_.back();
  sampledRvp_.pop_back();
  for (size_t i = 0; i < sampledRvp_.size(); i++) {
    RotVelPos start_to_t = sampledRvp_[i];

    // even for negative t, we have that it holds:
    //   start_to_t = combine(start_to_mid, mid_to_t)
    sampledRvp_[i] = uncombineLeft(start_to_t, start_to_mid);
  }

  forEachIntegratedMeasurement(
      imuMeas_,
      measModel,
      midpointTimeUs,
      midpointTimeUs + intervalHalfLengthUs_,
      [&](const RotVelPos& rvp, bool atAccelBoundary, bool atGyroBoundary) {
        if (atGyroBoundary) {
          sampledRvp_.push_back(rvp);
        }
      });

  interp_.clear();
  for (size_t i = 1; i < sampledRvp_.size(); i++) {
    if (sampledRvp_[i - 1].dtSec >= sampledRvp_[i].dtSec) {
      throw std::runtime_error("Wut?");
    }
    auto deltaRvp = uncombineLeft(sampledRvp_[i], sampledRvp_[i - 1]);
    interp_.push_back(differentiate(deltaRvp));
  }
}

RollingShutterEstimate RollingShutterData::getEstimate(
    double tDeltaSec,
    const Vec3& vel_world_atMid,
    const SE3& T_world_bodyImu_atMid,
    bool /* computeVelocities */) const {
  if (sampledRvp_.empty()) {
    throw std::runtime_error("Not initialized RS?");
  }

  auto it = std::upper_bound( // first one strictly >
      sampledRvp_.begin(),
      sampledRvp_.end(),
      tDeltaSec,
      [](double dt, const auto& rvp) -> bool { return (dt < rvp.dtSec); });

  // we raise exception if out of range - keep as is, to signal calibration is drifting way off
  if (it == sampledRvp_.end() || it == sampledRvp_.begin()) {
    throw std::runtime_error(
        fmt::format(
            "RollingShutterData::getEstimate: out of range\n"
            "tDeltaSec: {}ms, tRange: [{}ms..{}ms]",
            tDeltaSec * 1e3,
            sampledRvp_[0].dtSec * 1e3,
            sampledRvp_.back().dtSec * 1e3));
  }
  auto index = std::distance(sampledRvp_.begin(), it);

  const auto& interp = interp_[index - 1];
  const auto& rvpPrev = sampledRvp_[index - 1];
  auto rvpAtT = combine(rvpPrev, integrate(interp, tDeltaSec - rvpPrev.dtSec));

  const SO3 R_bodyImu_world_atMid = T_world_bodyImu_atMid.so3().inverse();
  const Vec3 gravityImuMid = R_bodyImu_world_atMid * gravityWorld_;
  const Vec3 vel_atMid_imuMid = R_bodyImu_world_atMid * vel_world_atMid;
  const Vec3 vel_imuMid_atT = rvpAtT.dV + gravityImuMid * tDeltaSec;
  const Vec3 pos_imuMid_atT =
      rvpAtT.dP + vel_atMid_imuMid * tDeltaSec + gravityImuMid * (0.5 * tDeltaSec * tDeltaSec);

  return RollingShutterEstimate{
      .T_midImu_imuAtT = SE3(rvpAtT.R, pos_imuMid_atT),
      .omega_imuT_atT = interp.gyroRadSec,
      .vel_imuT_atT = rvpAtT.R.inverse() * vel_imuMid_atT,
      .vel_world_atT = R_bodyImu_world_atMid * vel_imuMid_atT,
  };
}

} // namespace visual_inertial_ba::preintegration
