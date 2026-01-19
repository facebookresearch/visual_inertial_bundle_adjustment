/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/problem/BaseMapVisualFactor.h>

#define DEFAULT_LOG_CHANNEL "ViBa::BaseMapVisualFactor"
#include <logging/Log.h>

namespace visual_inertial_ba {

std::optional<Vec2> BaseMapVisualFactor::operator()(
    const Vec3& worldPt,
    Ref<Mat23>&& worldPt_Jacobian) const {
  const auto& camera = bmKr_.cameras[cameraIndex_];
  const Vec3 pointKeyRig = bmKr_.T_bodyImu_world * worldPt;
  const Vec3 pointCamera = camera.T_Cam_BodyImu * pointKeyRig;

  Vec2 proj;
  Mat23 dProj_dPointCam;
  if (!camera.cameraModel.project(pointCamera, proj, dProj_dPointCam)) {
    return {};
  }

  const Vec2 error = proj - projBaseRes_;
  const Vec2 whiteErr = sqrtH_BaseRes_ * error;
  if (!isNull(worldPt_Jacobian)) {
    const Mat23 dWhiteErr_dPointCam = sqrtH_BaseRes_ * dProj_dPointCam;
    worldPt_Jacobian =
        dWhiteErr_dPointCam * (camera.T_Cam_BodyImu.so3() * bmKr_.T_bodyImu_world.so3()).matrix();
  }
  return whiteErr;
}

} // namespace visual_inertial_ba
