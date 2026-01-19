/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/problem/SingleSessionProblem.h>

#define DEFAULT_LOG_CHANNEL "ViBa::RandomWalkFactors"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionProblem::addImuCalibRWFactor(
    int64_t calibPrevIndex,
    int64_t calibNextIndex,
    const VecX& diagSqrtH) {
  auto& calibPrevVar = imuCalib_at(calibPrevIndex).var;
  const int tgDim = calibPrevVar.value.estOpts->errorStateSize();
  XR_CHECK_EQ(diagSqrtH.size(), tgDim);
  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      [tgDim, diagSqrtH](
          const ImuCalibParam& calibPrev,
          const ImuCalibParam& calibNext,
          Ref<Mat<kImuCalibTangentMaxDim, Eigen::Dynamic>>&& calibPrevJacobian,
          Ref<Mat<kImuCalibTangentMaxDim, Eigen::Dynamic>>&& calibNextJacobian)
          -> Vec<kImuCalibTangentMaxDim> {
        if (!isNull(calibPrevJacobian)) {
          XR_CHECK_EQ(calibPrevJacobian.cols(), tgDim);
          calibPrevJacobian.setZero();
          calibPrevJacobian.diagonal() = -diagSqrtH;
        }
        if (!isNull(calibNextJacobian)) {
          XR_CHECK_EQ(calibNextJacobian.cols(), tgDim);
          calibNextJacobian.setZero();
          calibNextJacobian.diagonal() = diagSqrtH;
        }

        Map<VecX> deltaCalib((double*)alloca(sizeof(double) * tgDim), tgDim);
        calibNext.boxMinus(calibPrev.modelParams, deltaCalib);
        Vec<kImuCalibTangentMaxDim> ret = Vec<kImuCalibTangentMaxDim>::Zero();
        ret.head(tgDim).noalias() = deltaCalib.cwiseProduct(diagSqrtH);
        return ret;
      },
      calibPrevVar,
      imuCalib_at(calibNextIndex).var);

  if (verbosity_ == Verbose) {
    XR_LOGI(
        "RW/ImuBias[{}..{}], RES: {}",
        calibPrevIndex,
        calibNextIndex,
        factorStorePtr->rawError(factorIndex));
  }
}

void SingleSessionProblem::addCamIntrRWFactor(
    int64_t calibPrevIndex,
    int64_t calibNextIndex,
    const VecX& diagSqrtH) {
  auto& prevCamModel = cameraModel_at(calibPrevIndex).var;
  auto& nextCamModel = cameraModel_at(calibNextIndex).var;
  XR_CHECK_EQ(prevCamModel.getTangentDim(), nextCamModel.getTangentDim());
  XR_CHECK_EQ(diagSqrtH.size(), prevCamModel.getTangentDim());

  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      [diagSqrtH](
          const CameraModelParam& calibPrev,
          const CameraModelParam& calibNext,
          Ref<Mat<kMaxCamParams, Eigen::Dynamic>>&& calibPrevJacobian,
          Ref<Mat<kMaxCamParams, Eigen::Dynamic>>&& calibNextJacobian) -> Vec<kMaxCamParams> {
        const int nParams = diagSqrtH.size();
        if (!isNull(calibPrevJacobian)) {
          calibPrevJacobian.setZero();
          calibPrevJacobian.topLeftCorner(nParams, nParams).diagonal() = -diagSqrtH;
        }
        if (!isNull(calibNextJacobian)) {
          calibNextJacobian.setZero();
          calibNextJacobian.topLeftCorner(nParams, nParams).diagonal() = diagSqrtH;
        }

        Vec<kMaxCamParams> params, retv = Vec<kMaxCamParams>::Zero();
        ::small_thing::VarSpec<CameraModelParam>::boxMinus(
            calibNext, calibPrev, params.head(nParams));
        retv.head(nParams) = params.head(nParams).cwiseProduct(diagSqrtH);
        return retv;
      },
      prevCamModel,
      nextCamModel);

  if (verbosity_ == Verbose) {
    XR_LOGI(
        "RW/CamIntr[{}..{}], RES: {}",
        calibPrevIndex,
        calibNextIndex,
        factorStorePtr->rawError(factorIndex));
  }
}

void SingleSessionProblem::addImuExtrRWFactor(
    int64_t calibPrevIndex,
    int64_t calibNextIndex,
    const Vec6& diagSqrtH) {
  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      [diagSqrtH](
          const SE3& extrPrev,
          const SE3& extrNext,
          Ref<Mat66>&& extrPrevJacobian,
          Ref<Mat66>&& extrNextJacobian) -> Vec6 {
        const SE3 errorAtImu = extrNext * extrPrev.inverse();
        const Vec6 logErrorAtCam = errorAtImu.log();
        const Mat66 dLogError_dLeftPoseError = SE3::leftJacobianInverse(logErrorAtCam);
        if (!isNull(extrPrevJacobian)) {
          extrPrevJacobian =
              -(diagSqrtH.asDiagonal() * dLogError_dLeftPoseError * errorAtImu.Adj());
        }
        if (!isNull(extrNextJacobian)) {
          extrNextJacobian = diagSqrtH.asDiagonal() * dLogError_dLeftPoseError;
        }
        return logErrorAtCam.cwiseProduct(diagSqrtH);
      },
      T_Imu_BodyImu_at(calibPrevIndex).var,
      T_Imu_BodyImu_at(calibNextIndex).var);

  if (verbosity_ == Verbose) {
    XR_LOGI(
        "RW/ImuExtr[{}..{}], RES: {}",
        calibPrevIndex,
        calibNextIndex,
        factorStorePtr->rawError(factorIndex));
  }
}

void SingleSessionProblem::addCamExtrRWFactor(
    int64_t calibPrevIndex,
    int64_t calibNextIndex,
    const Vec6& diagSqrtH) {
  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      [diagSqrtH](
          const SE3& extrPrev,
          const SE3& extrNext,
          Ref<Mat66>&& extrPrevJacobian,
          Ref<Mat66>&& extrNextJacobian) -> Vec6 {
        const SE3 errorAtCam = extrNext * extrPrev.inverse();
        const Vec6 logErrorAtCam = errorAtCam.log();
        const Mat66 dLogError_dLeftPoseError = SE3::leftJacobianInverse(logErrorAtCam);
        if (!isNull(extrPrevJacobian)) {
          extrPrevJacobian =
              -(diagSqrtH.asDiagonal() * dLogError_dLeftPoseError * errorAtCam.Adj());
        }
        if (!isNull(extrNextJacobian)) {
          extrNextJacobian = diagSqrtH.asDiagonal() * dLogError_dLeftPoseError;
        }
        return logErrorAtCam.cwiseProduct(diagSqrtH);
      },
      T_Cam_BodyImu_at(calibPrevIndex).var,
      T_Cam_BodyImu_at(calibNextIndex).var);

  if (verbosity_ == Verbose) {
    XR_LOGI(
        "RW/CamExtr[{}..{}], RES: {}",
        calibPrevIndex,
        calibNextIndex,
        factorStorePtr->rawError(factorIndex));
  }
}

} // namespace visual_inertial_ba
