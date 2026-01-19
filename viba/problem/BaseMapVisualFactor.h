/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <small_thing/Optimizer.h>
#include <viba/problem/Types.h>

namespace visual_inertial_ba {

struct BaseMapCamera {
  SE3 T_Cam_BodyImu;
  CameraModelParam cameraModel;
};
struct BaseMapKeyRig {
  std::vector<BaseMapCamera> cameras;
  SE3 T_bodyImu_world;
};

class BaseMapVisualFactor {
 public:
  BaseMapVisualFactor(
      const BaseMapKeyRig& bmKr,
      int cameraIndex,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes)
      : bmKr_(bmKr),
        cameraIndex_(cameraIndex),
        projBaseRes_(projBaseRes),
        sqrtH_BaseRes_(sqrtH_BaseRes) {}

  std::optional<Vec2> operator()(const Vec3& worldPt, Ref<Mat23>&& worldPt_Jacobian) const;

  const BaseMapKeyRig& bmKr_;
  int cameraIndex_;
  const Vec2 projBaseRes_;
  const Mat22 sqrtH_BaseRes_;
};

} // namespace visual_inertial_ba
