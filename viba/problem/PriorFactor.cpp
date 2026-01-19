/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/problem/SingleSessionProblem.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::PriorFactor"
#include <logging/Log.h>

namespace visual_inertial_ba {

// Note: this is only used when computing covariances
static constexpr double kPositionConstraintInvStdDev = 1e3;
static constexpr double kYawConstraintInvRadStdDev = 1e3;

void SingleSessionProblem::constrainPositionAndYaw(int64_t rigIndex) {
  const auto& T_bodyImu_world = findOrDie(inertialPoses_, rigIndex).T_bodyImu_world.value;

  Mat66 H = Mat66::Zero();
  H.topLeftCorner<3, 3>() =
      Mat33::Identity() * (kPositionConstraintInvStdDev * kPositionConstraintInvStdDev);
  const Vec3 downConstraint =
      T_bodyImu_world.so3() * gravityWorld_->value.vec.normalized() * kYawConstraintInvRadStdDev;
  H.bottomRightCorner<3, 3>() = downConstraint * downConstraint.transpose();

  addPosePrior(rigIndex, T_bodyImu_world, H);
}

void SingleSessionProblem::addPosePrior(
    int64_t rigIndex,
    const SE3& prior_T_bodyImu_world,
    const Mat66& H) {
  const SE3 prior_T_world_rig = prior_T_bodyImu_world.inverse();

  opt_.addFactor(
      [prior_T_world_rig](
          const SE3& T_bodyImu_world, Ref<Mat66>&& T_bodyImu_world_Jacobian) -> Vec6 {
        const SE3 poseErrorAtRig = T_bodyImu_world * prior_T_world_rig;
        Vec6 logErrorAtRig = poseErrorAtRig.log();
        if (!isNull(T_bodyImu_world_Jacobian)) {
          T_bodyImu_world_Jacobian = SE3::leftJacobianInverse(logErrorAtRig.head<6>());
        }
        return logErrorAtRig;
      },
      H,
      findOrDie(inertialPoses_, rigIndex).T_bodyImu_world);
}

// waiting for std::erase_if in C++-20...
template <typename T, typename P>
static size_t erase_if(T& c, P&& pred) {
  auto old_size = c.size();
  for (auto i = c.begin(), last = c.end(); i != last;) {
    if (pred(*i)) {
      i = c.erase(i);
    } else {
      ++i;
    }
  }
  return old_size - c.size();
}

int64_t SingleSessionProblem::clearPosePriors() {
  int64_t numRemoved = 0;
  erase_if(opt_.factorStores.stores, [&](const auto& item) -> bool {
    auto const& [_, factorStore] = item;
    std::string name = factorStore->name();
    if (name.find("visual_inertial_ba::SingleSessionProblem::addPosePrior(") != std::string::npos) {
      numRemoved += factorStore->numCosts();
      return true;
    }
    return false;
  });
  return numRemoved;
}

void SingleSessionProblem::addImuPrior(
    int64_t imuCalibIndex,
    const ImuMeasurementModelParameters& prior,
    const VecX& diagH) {
  auto& var = imuCalibs_[imuCalibIndex].var;
  const int tgDim = var.value.estOpts->errorStateSize();
  XR_CHECK_EQ(tgDim, diagH.size());

  const VecX diagSqrtH = diagH.cwiseSqrt();
  opt_.addFactor(
      [prior, tgDim, diagSqrtH](
          const ImuCalibParam& calib,
          Ref<Mat<kImuCalibTangentMaxDim, Eigen::Dynamic>>&& calibJacobian)
          -> Vec<kImuCalibTangentMaxDim> {
        if (!isNull(calibJacobian)) {
          XR_CHECK_EQ(calibJacobian.cols(), tgDim);
          calibJacobian.setZero();
          calibJacobian.diagonal() = diagSqrtH;
        }

        Map<VecX> deltaCalib((double*)alloca(sizeof(double) * tgDim), tgDim);
        calib.boxMinus(prior, deltaCalib);
        Vec<kImuCalibTangentMaxDim> ret = Vec<kImuCalibTangentMaxDim>::Zero();
        ret.head(tgDim).noalias() = deltaCalib.cwiseProduct(diagSqrtH);
        return ret;
      },
      var);
}

void SingleSessionProblem::addCamIntrinsicsPrior(
    int64_t camModelIndex,
    const CameraModelParam& prior,
    const VecX& diagH) {
  XR_CHECK_LE(diagH.size(), kMaxCamParams);

  VecX diagSqrtH = diagH.cwiseSqrt();
  XR_CHECK_EQ(
      diagSqrtH.size(), ::small_thing::VarSpec<CameraModelParam>::getDynamicTangentDim(prior));
  XR_CHECK_EQ(diagSqrtH.size(), cameraModels_[camModelIndex].var.getTangentDim());

  opt_.addFactor(
      [prior, diagSqrtH](
          const CameraModelParam& calib,
          Ref<Mat<kMaxCamParams, Eigen::Dynamic>>&& calibJacobian) -> Vec<kMaxCamParams> {
        const int nParams = diagSqrtH.size();
        if (!isNull(calibJacobian)) {
          calibJacobian.setZero();
          calibJacobian.topLeftCorner(nParams, nParams).diagonal() = diagSqrtH;
        }

        Vec<kMaxCamParams> params, retv = Vec<kMaxCamParams>::Zero();
        ::small_thing::VarSpec<CameraModelParam>::boxMinus(calib, prior, params.head(nParams));
        retv.head(nParams) = params.head(nParams).cwiseProduct(diagSqrtH);
        return retv;
      },
      cameraModels_[camModelIndex].var);
}

void SingleSessionProblem::addCamExtrinsicsPrior(
    int64_t camExtrinsicsIndex,
    const SE3& prior,
    const Vec6& diagH) {
  const SE3 priorInv = prior.inverse();
  const Vec6 diagSqrtH = diagH.cwiseSqrt();

  opt_.addFactor(
      [priorInv, diagSqrtH](const SE3& extr, Ref<Mat66>&& extrJacobian) -> Vec6 {
        const SE3 errorAtCam = extr * priorInv;
        const Vec6 logErrorAtCam = errorAtCam.log();
        if (!isNull(extrJacobian)) {
          extrJacobian = diagSqrtH.asDiagonal() * SE3::leftJacobianInverse(logErrorAtCam);
        }
        return logErrorAtCam.cwiseProduct(diagSqrtH);
      },
      Ts_Cam_BodyImu_[camExtrinsicsIndex].var);
}

void SingleSessionProblem::addImuExtrinsicsPrior(
    int64_t imuExtrinsicsIndex,
    const SE3& prior,
    const Vec6& diagH) {
  const SE3 priorInv = prior.inverse();
  const Vec6 diagSqrtH = diagH.cwiseSqrt();

  opt_.addFactor(
      [priorInv, diagSqrtH](const SE3& extr, Ref<Mat66>&& extrJacobian) -> Vec6 {
        const SE3 errorAtCam = extr * priorInv;
        const Vec6 logErrorAtCam = errorAtCam.log();
        if (!isNull(extrJacobian)) {
          extrJacobian = diagSqrtH.asDiagonal() * SE3::leftJacobianInverse(logErrorAtCam);
        }
        return logErrorAtCam.cwiseProduct(diagSqrtH);
      },
      Ts_Imu_BodyImu_[imuExtrinsicsIndex].var);
}

} // namespace visual_inertial_ba
