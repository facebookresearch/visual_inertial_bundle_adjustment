/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sophus/se3.hpp>
#include <viba/problem/Types.h>
#include <Eigen/Geometry>

namespace visual_inertial_ba {

struct InertialPoseData {
  SE3 T_bodyImu_world;
  Vec3 vel_world;
  Vec3 omega; // angular velocity (in bodyImu's reference frame)
};

// abstract class that allows to override (inertial)poses with Gt or alternate values
class TrajectoryBase {
 public:
  virtual ~TrajectoryBase() = default;

  virtual bool haveVelocities() const = 0;

  virtual Vec3 gravity() const = 0;

  virtual SE3 T_bodyImu_world(int64_t timestampUs) const = 0;

  virtual InertialPoseData inertialPose(int64_t timestampUs) const = 0;
};

} // namespace visual_inertial_ba
