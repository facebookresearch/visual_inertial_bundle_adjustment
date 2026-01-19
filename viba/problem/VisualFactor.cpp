/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/problem/SingleSessionProblem.h>

#include <rolling_shutters/RollingShutterData.h>
#include <viba/common/Enumerate.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::VisualFactor"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionProblem::printDetectorBiases(const std::string& label) {
  printDetectorBiases(label, detectorBiases_);
}

void SingleSessionProblem::printDetectorBiases(
    const std::string& label,
    const std::vector<Point2DVariable>& vars) {
  std::stringstream ss;
  ss << "Detector bias (" << label << "):" << std::endl;
  for (const auto& [i, var] : enumerate(vars)) {
    ss << "  Camera n." << i << ": BX=" << var.value[0] << ", BY=" << var.value[1] << std::endl;
  }
  XR_LOGI("{}", ss.str());
}

// Note: don't put in anonymous namespace so we can find the factors via introspection
// @lint-ignore-every CLANGTIDY facebook-hte-ShadowingClass
struct VisualFactor {
  VisualFactor(const Vec2& projBaseRes, const Mat22& sqrtH_BaseRes)
      : projBaseRes_(projBaseRes), sqrtH_BaseRes_(sqrtH_BaseRes) {}

  std::optional<Vec2> operator()(
      const Vec3& worldPt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      Ref<Mat23> worldPt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian) const {
    const Vec3 pointKeyRig = T_bodyImu_world * worldPt;
    const Vec3 pointCamera = T_Cam_BodyImu * pointKeyRig;

    Vec2 proj;
    Mat23 dProj_dPointCam;
    const int nIntrinsics = cameraModel.numModelParameters();
    Mat<2, Eigen::Dynamic> dProj_dCamParams(2, nIntrinsics); // TODO: avoid dynamic allocation
    if (!cameraModel.project(pointCamera, proj, dProj_dPointCam, dProj_dCamParams)) {
      return {};
    }

    const Vec2 error = proj - projBaseRes_;
    const Vec2 whiteErr = sqrtH_BaseRes_ * error;
    const Mat23 dWhiteErr_dPointCam = sqrtH_BaseRes_ * dProj_dPointCam;
    // clang-format off
    if (!isNull(worldPt_Jacobian)) {
      worldPt_Jacobian = //
        dWhiteErr_dPointCam * (T_Cam_BodyImu.so3() * T_bodyImu_world.so3()).matrix();
    }
    if (!isNull(T_bodyImu_world_Jacobian)) {
      const Mat23 dWhiteErr_dRigPtT = dWhiteErr_dPointCam * T_Cam_BodyImu.so3().matrix();
      T_bodyImu_world_Jacobian << //
        dWhiteErr_dRigPtT, dWhiteErr_dRigPtT * SO3::hat(-pointKeyRig);
    }
    if (!isNull(T_Cam_BodyImu_Jacobian)) {
      T_Cam_BodyImu_Jacobian << //
        dWhiteErr_dPointCam, dWhiteErr_dPointCam * SO3::hat(-pointCamera);
    }
    if (!isNull(cameraModel_Jacobian)) {
      cameraModel_Jacobian.leftCols(nIntrinsics) = sqrtH_BaseRes_ * dProj_dCamParams;
    }
    // clang-format on
    return whiteErr;
  }

  std::optional<Vec2> operator()(
      const Vec3& worldPt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      const Vec2& detectorBias,
      Ref<Mat23> worldPt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian,
      Ref<Mat22> detectorBias_Jacobian) const {
    std::optional maybeErr = this->operator()(
        worldPt,
        T_bodyImu_world,
        T_Cam_BodyImu,
        cameraModel,
        std::move(worldPt_Jacobian),
        std::move(T_bodyImu_world_Jacobian),
        std::move(T_Cam_BodyImu_Jacobian),
        std::move(cameraModel_Jacobian));
    if (!maybeErr.has_value()) {
      return {};
    }
    if (!isNull(detectorBias_Jacobian)) {
      detectorBias_Jacobian = sqrtH_BaseRes_;
    }
    return sqrtH_BaseRes_ * detectorBias + *maybeErr;
  }

  double imageRow() const {
    return projBaseRes_[1];
  }

 private:
  Vec2 projBaseRes_;
  Mat22 sqrtH_BaseRes_;
};

struct RollingShutterVisualFactor {
  static constexpr bool kUseAnalyticTimeOffsetJac = false;

  RollingShutterVisualFactor(
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const RollingShutterData& rsData)
      : visFactor_(projBaseRes, sqrtH_BaseRes), rsData_(rsData) {}

  std::optional<Vec2> operator()(
      const Vec3& worldPt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      const Vec3& vel_world,
      Ref<Mat23> worldPt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian,
      Ref<Mat23> vel_world_Jacobian) const {
    const double timeParamFactor = visFactor_.imageRow() / cameraModel.imageHeight() - 0.5;
    const double dtSec =
        cameraModel.readoutTimeSec() * timeParamFactor - cameraModel.timeOffsetSec_Dev_Camera();

    const bool needVelocities = kUseAnalyticTimeOffsetJac && !isNull(cameraModel_Jacobian) &&
        (cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset);

    const RollingShutterEstimate shiftedEst =
        rsData_.getEstimate(dtSec, vel_world, T_bodyImu_world.inverse(), needVelocities);
    const SE3 T_bodyImuAtT_bodyImuMid = shiftedEst.T_midImu_imuAtT.inverse();
    const SE3 T_bodyImuAtT_world = T_bodyImuAtT_bodyImuMid * T_bodyImu_world;

    Mat26 T_bodyImuAtT_world_Jacobian;
    const bool dontNeedPoseJacobian =
        (isNull(cameraModel_Jacobian) ||
         !(cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset)) &&
        isNull(T_bodyImu_world_Jacobian) && isNull(vel_world_Jacobian);
    const auto maybeErr = visFactor_(
        worldPt,
        T_bodyImuAtT_world,
        T_Cam_BodyImu,
        cameraModel,
        worldPt_Jacobian,
        dontNeedPoseJacobian ? NullRef() : Ref<Mat26>(T_bodyImuAtT_world_Jacobian),
        T_Cam_BodyImu_Jacobian,
        cameraModel_Jacobian);

    if (!isNull(cameraModel_Jacobian) &&
        (cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset)) {
      Vec6 dT_bodyImuAtT_world_dT;
      if constexpr (kUseAnalyticTimeOffsetJac) {
        dT_bodyImuAtT_world_dT << -shiftedEst.vel_imuT_atT, -shiftedEst.omega_imuT_atT;
      } else {
        constexpr double kEpsilon = 1e-6;
        const RollingShutterEstimate shiftedEstP =
            rsData_.getEstimate(dtSec + kEpsilon, vel_world, T_bodyImu_world.inverse(), false);
        const SE3 T_bodyImuAtTpEps_bodyImuAtT =
            shiftedEstP.T_midImu_imuAtT.inverse() * shiftedEst.T_midImu_imuAtT;
        dT_bodyImuAtT_world_dT = T_bodyImuAtTpEps_bodyImuAtT.log() / kEpsilon; /* NUMERIC */
      }

      const Vec2 dErr_dT = T_bodyImuAtT_world_Jacobian * dT_bodyImuAtT_world_dT;
      int idx = cameraModel_Jacobian.cols();
      if (cameraModel.estimateTimeOffset) {
        cameraModel_Jacobian.col(--idx) = -dErr_dT;
      }
      if (cameraModel.estimateReadoutTime) {
        cameraModel_Jacobian.col(--idx) = dErr_dT * timeParamFactor;
      }
    }

    if (!isNull(T_bodyImu_world_Jacobian)) {
      Mat66 dT_bodyImuAtT_world_dBodyImuMid = T_bodyImuAtT_bodyImuMid.Adj();
      dT_bodyImuAtT_world_dBodyImuMid.topRightCorner<3, 3>() += SO3::hat(
          T_bodyImuAtT_world.so3() *
          (vel_world * dtSec + 0.5 * dtSec * dtSec * rsData_.gravityWorld()));

      T_bodyImu_world_Jacobian = T_bodyImuAtT_world_Jacobian * dT_bodyImuAtT_world_dBodyImuMid;
    }

    if (!isNull(vel_world_Jacobian)) {
      Mat63 dT_bodyImuAtT_world_dVelW;
      dT_bodyImuAtT_world_dVelW << T_bodyImuAtT_world.so3().matrix() * (-dtSec), Mat33::Zero();

      vel_world_Jacobian = T_bodyImuAtT_world_Jacobian * dT_bodyImuAtT_world_dVelW;
    }

    return maybeErr;
  }

  const VisualFactor visFactor_;
  const RollingShutterData& rsData_;
};

void SingleSessionProblem::addVisualFactor(
    int64_t rigIndex,
    int64_t cameraIndex,
    PointVariable& pointTrackVar,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    const SoftLossType& reprojErrorLoss) {
  const int64_t cameraModelIndex = findOrDie(rigCamToModelIndex_, {rigIndex, cameraIndex});
  const int64_t cameraExtrinsicsIndex = findOrDie(rigCamToExtrIndex_, {rigIndex, cameraIndex});

  // get rig and extrinsic variables
  XR_CHECK_LT(cameraExtrinsicsIndex, Ts_Cam_BodyImu_.size());
  auto& rigVar = findOrDie(inertialPoses_, rigIndex);
  auto& T_Cam_BodyImu_var = Ts_Cam_BodyImu_[cameraExtrinsicsIndex].var;
  auto& camVar = cameraModels_[cameraModelIndex].var;

  small_thing::FactorStoreBase* factorStorePtr;
  int64_t factorIndex;

  if (camVar.value.hasTimeOffset() || camVar.value.isRollingShutter()) {
    XR_CHECK(rigIndex);

    const auto& rsData = findOrDie(rigToRSData_, rigIndex);
    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        RollingShutterVisualFactor(projBaseRes, sqrtH_BaseRes, rsData),
        reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
        pointTrackVar,
        rigVar.T_bodyImu_world,
        T_Cam_BodyImu_var,
        camVar,
        rigVar.vel_world);
  } else {
    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        VisualFactor(projBaseRes, sqrtH_BaseRes),
        reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
        pointTrackVar,
        rigVar.T_bodyImu_world,
        T_Cam_BodyImu_var,
        camVar);
  }

  if (verbosity_ == Verbose) {
    Vec2 err;
    bool hasErr = factorStorePtr->unweightedError(factorIndex, err);
    if (hasErr) {
      XR_LOGI("RIG/CAM: [{}, {}], RES: {}", rigIndex, cameraIndex, err);
    }
  }
}

void SingleSessionProblem::addVisualFactorWithBias(
    int64_t rigIndex,
    int64_t cameraIndex,
    PointVariable& pointTrackVar,
    Point2DVariable& detectorBias,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    const SoftLossType& reprojErrorLoss) {
  const int64_t cameraModelIndex = findOrDie(rigCamToModelIndex_, {rigIndex, cameraIndex});
  const int64_t cameraExtrinsicsIndex = findOrDie(rigCamToExtrIndex_, {rigIndex, cameraIndex});

  // get rig and extrinsic variables
  XR_CHECK_LT(cameraExtrinsicsIndex, Ts_Cam_BodyImu_.size());
  auto& rigVar = findOrDie(inertialPoses_, rigIndex);
  auto& T_Cam_BodyImu_var = Ts_Cam_BodyImu_[cameraExtrinsicsIndex].var;
  auto& camVar = cameraModels_[cameraModelIndex].var;

  XR_CHECK(
      !camVar.value.isRollingShutter(), "Image Bias not supported for rolling shutter cameras");

  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      VisualFactor(projBaseRes, sqrtH_BaseRes),
      reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
      pointTrackVar,
      rigVar.T_bodyImu_world,
      T_Cam_BodyImu_var,
      camVar,
      detectorBias);

  if (verbosity_ == Verbose) {
    auto err = factorStorePtr->rawError(factorIndex);
    if (err.has_value()) {
      XR_LOGI("RIG/CAM: [{}, {}], RES: {}", rigIndex, cameraIndex, err.value());
    }
  }
}

SE3 SingleSessionProblem::T_bodyImu_world_atImageRow(
    int64_t rigIndex,
    int cameraIndex,
    float imageRow) const {
  const auto& rigVar = inertialPose(rigIndex);
  const auto& camVar = cameraModel(rigIndex, cameraIndex).var;
  if (camVar.value.isRollingShutter() || camVar.value.hasTimeOffset()) {
    const auto& cameraModel = camVar.value;
    const double timeParamFactor = imageRow / cameraModel.imageHeight() - 0.5;
    const double dtSec =
        cameraModel.readoutTimeSec() * timeParamFactor - cameraModel.timeOffsetSec_Dev_Camera();

    const auto& rsData = findOrDie(rigToRSData_, rigIndex);
    const RollingShutterEstimate shiftedEst = rsData.getEstimate(
        dtSec,
        rigVar.vel_world.value,
        rigVar.T_bodyImu_world.value.inverse(),
        /* needVelocities = */ false);
    return shiftedEst.T_midImu_imuAtT.inverse() * rigVar.T_bodyImu_world.value;
  }

  return rigVar.T_bodyImu_world.value;
}

} // namespace visual_inertial_ba
