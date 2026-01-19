/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <preintegration/ImuTypes.h>
#include <preintegration/MotionIntegral.h>

namespace visual_inertial_ba::preintegration {

struct RollingShutterEstimate {
  SE3 T_midImu_imuAtT;
  Vec3 omega_imuT_atT;
  Vec3 vel_imuT_atT;
  Vec3 vel_world_atT;
};

struct RollingShutterData {
  RollingShutterData(
      int64_t /* midPointTimestampUs */,
      int64_t intervalHalfLengthUs,
      const std::vector<ImuMeasurement>& imuMeas)
      : intervalHalfLengthUs_(intervalHalfLengthUs), imuMeas_(imuMeas) {}

  void compute(
      int64_t midpointTimeUs,
      const ImuMeasurementModelParameters& measModel,
      const Vec3& gravityWorld);

  RollingShutterEstimate getEstimate(
      double tDeltaSec,
      const Vec3& vel_world_mid,
      const SE3& T_world_bodyImu_mid,
      bool computeVelocities) const;

  const Vec3& gravityWorld() const {
    return gravityWorld_;
  }

  // settings
  const int intervalHalfLengthUs_;
  const std::vector<ImuMeasurement>& imuMeas_;

  // updatable values
  int64_t midpointTimeUs_;
  Vec3 gravityWorld_;
  std::vector<RotVelPos> sampledRvp_;
  std::vector<RVPInterpolationData> interp_;
};

} // namespace visual_inertial_ba::preintegration
